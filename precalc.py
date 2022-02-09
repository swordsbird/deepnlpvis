import os
import numpy as np
import math
import json
from sklearn.metrics.pairwise import euclidean_distances
import sys
from sklearn.cluster import AgglomerativeClustering
sys.path.append('lib')
from data_loader_old import DataLoader
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForSequenceClassification
import nltk.stem
stemmer = nltk.stem.SnowballStemmer('english')
import torch
import random
import yaml
from sklearn.manifold import TSNE
from model.model_elmo import elmo_RNNClassifier
from model.basic_model_helper import preprocess_sentence_batch, set_device

is_sst2_dataset = True
is_lstm = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
home_path = '/home/lizhen/data/'

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# load bert model
bert = BertModel.from_pretrained("bert-base-uncased")
if is_lstm:
    model = elmo_RNNClassifier(yaml.load(open(os.path.join('cache', 'LSTM_basic_4_layer_768_hidden_pool_max.yaml'),'r')))
    model.load_state_dict(torch.load(os.path.join('cache', '_lstm_basic_SST2_2_100_0.8796.param'), map_location='cpu'))
    model.to(device)
else:
    if is_sst2_dataset:
        state_dict = torch.load(home_path + 'model/sst2/bert_sst2t_5_100_0.9323.param',map_location='cpu')
        #state_dict = torch.load(home_path + 'model/sst2/bert_SST2_3_1300_0.9369.param',map_location='cpu')
    else:
        state_dict = torch.load(home_path + 'model/agnews/bert_AGNEWS_1_3700_0.9479.param',map_location='cpu')
    old_keys = []
    new_keys = []
    for key in list(state_dict.keys()):
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    if is_sst2_dataset:
        bert2 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2)
    else:
        bert2 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 4)
    bert2.load_state_dict(state_dict)
    bert2.to(device)
    for param in bert2.parameters():
        param.requires_grad = False
    bert2.eval()

    old_keys = []
    new_keys = []
    for key in list(state_dict.keys()):
        new_key = None
        if 'bert.' in key:
            new_key = key.replace('bert.', '')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    cl_weight = state_dict.pop('classifier.weight')
    cl_bias = state_dict.pop('classifier.bias')
    pooled_weight = state_dict['pooler.dense.weight']
    pooled_bias = state_dict['pooler.dense.bias']
    bert.load_state_dict(state_dict)
    bert.to(device)

    for param in bert.parameters():
        param.requires_grad = False
    bert.eval()

    cl_weight_t = cl_weight.transpose(0,1)
    cl_wt_ = cl_weight_t.cuda(device=device)
    cl_b_ = cl_bias.cuda(device=device)
    pooled_wt_ = pooled_weight.transpose(0,1).cuda(device=device)
    pooled_b_ = pooled_bias.cuda(device=device)
    cl_b = cl_bias.cpu()
    cl_wt = cl_weight_t.cpu()
    pooled_b = pooled_bias.cpu()
    pooled_wt = pooled_weight.transpose(0,1).cpu()

def get_embedding_from_text(text):
    words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
    tokenized_ids = tokenizer.convert_tokens_to_ids(words)
    segment_ids = [0 for _ in range(len(words))]
    token_tensor = torch.tensor([tokenized_ids], device=device)
    segment_tensor = torch.tensor([segment_ids], device=device)
    x = bert.embeddings(token_tensor, segment_tensor)[0]
    return x

def get_logits_from_model(text):
    words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
    tokenized_ids = tokenizer.convert_tokens_to_ids(words)
    mask = [1 for _ in range(len(words))]
    x = torch.tensor([tokenized_ids], dtype = torch.long)
    mask = torch.tensor([mask], dtype = torch.long)
    x = x.cuda()
    mask = mask.cuda()
    ret = bert2(x, attention_mask=mask)
    ret = ret.detach().cpu()
    return ret[0]

def get_embedding_of_layer(x, start_layer=0, end_layer=-1):
    no_batch = len(x.shape) == 2
    if no_batch:
        x = x.unsqueeze(0)
    attention_mask = torch.ones(x.shape[:2]).to(x.device)
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    model_list = bert.encoder.layer[start_layer:end_layer]
    hidden_states = x
    for layer_module in model_list:
        hidden_states = layer_module(hidden_states, extended_attention_mask)
    return hidden_states[0] if no_batch else hidden_states

get_grids_ret = None

sentence_network_cache = {}
word_network_cache = {}

def get_logits_from_CLS(x):
    x = torch.matmul(torch.tensor(x), pooled_wt) + pooled_b
    logits = torch.matmul(x, cl_wt) + cl_b
    logits = torch.softmax(logits * 0.3, dim=0).numpy()
    return logits

def get_logits(x):
    if len(x.shape) == 2:
        x = x.unsqueeze(0)
    x = torch.matmul(x[:,0,:], pooled_wt_) + pooled_b_
    logits = torch.matmul(x, cl_wt_) + cl_b_
    logits = torch.softmax(logits, dim=1)
    logits = logits.mean(axis=0)
    return logits.cpu().numpy()

def get_weight_matrix(m, normalize = True):
    m = np.maximum(np.log2(0.45) - np.log2(1e-4 + m), 0.001)
    if normalize:
        return m.transpose() / m.sum(axis = 1)
    else:
        return m.transpose()

def set_embedding_and_prediction_score(loader):
    train_x = []
    train_y = []
    train_label = []
    print('dataset name', loader.dataset_name, loader.size)
    for i in range(loader.size):
        if 'lstm' in loader.dataset_name:
            v = loader.sample_emb[i]
            logits = loader.sample_logits[i]
            #print(logits.shape)
            logits = torch.softmax(torch.tensor(logits) * 0.6, dim=0).numpy()
        else:
            v = loader.word_activation[i][-1, 0]
            logits = get_logits_from_CLS(v)
        train_x.append(v)
        train_y.append([float(k) for k in logits])
        train_label.append(int(loader.data_label[i]))
        pred_label = loader.data_labels[np.argmax(logits)]
        true_label = loader.data_label[i]
    x_embedded = TSNE(n_components=1, n_jobs=-1, random_state=1).fit_transform(train_x)
    x_min = x_embedded.min()
    x_max = x_embedded.max()
    grid_n = 30
    step = (x_max - x_min + 1e-3) / grid_n
    z = [1] * grid_n
    for i in range(grid_n):
        x0 = x_min + step * i
        x1 = x_min + step * (i + 1)
        t = ((x_embedded >= x0) * (x_embedded < x1)).sum()
        if t == 0:
            z[i] = 0
    for i in range(1, grid_n):
        z[i] += z[i - 1]
    for i in range(grid_n):
        z[i] = i + 1 - z[i]
    for i in range(loader.size):
        t = (x_embedded[i] - x_min) / step
        t = int(t)
        x_embedded[i] = x_embedded[i] - step * z[t]

    avg_x = x_embedded.mean()
    y0 = np.mean([train_y[i][0] for i in range(loader.size) if x_embedded[i] < avg_x])
    y1 = np.mean([train_y[i][0] for i in range(loader.size) if x_embedded[i] >= avg_x])
    if y0 < y1:
        x_embedded = [-x_embedded[i] for i in range(loader.size)]
    data_embed = [float(x_embedded[i]) for i in range(loader.size)]
    data_prob = [train_y[i] for i in range(loader.size)]
    s = json.dumps(data_embed)
    data_embed_file = os.path.join('cache', loader.dataset_name, 'data_embed.json')
    f = open(data_embed_file, 'w')
    f.write(s)
    s = json.dumps(data_prob)
    data_prob_file = os.path.join('cache', loader.dataset_name, 'data_prob.json')
    f = open(data_prob_file, 'w')
    f.write(s)
    loader.set_data_embedding(data_embed)
    loader.set_data_prediction_score(data_prob)
    word_prob = {}
    word_embed = {}
    for i in range(loader.size):
        m = loader.word_entropy[i][-1]
        for j in range(len(loader.data_word[i])):
            word = loader.data_word[i][j]
            if word != '':
                if word not in word_prob:
                    word_prob[word] = []
                    word_embed[word] = []
                word_prob[word].append(loader.data_prediction_score[i])
                word_embed[word].append(loader.data_embedding[i])
    for word in word_prob:
        w = np.array(word_prob[word])
        word_prob[word] = { 'std': float(w.std()), 'mean': float(w.mean()) }
    for word in word_embed:
        w = np.array(word_embed[word])
        word_embed[word] = { 'std': float(w.std()), 'mean': float(w.mean()) }
    s = json.dumps(word_embed)
    word_embed_file = os.path.join('cache', loader.dataset_name, 'word_embed.json')
    f = open(word_embed_file, 'w')
    f.write(s)
    s = json.dumps(word_prob)
    word_prob_file = os.path.join('cache', loader.dataset_name, 'word_prob.json')
    f = open(word_prob_file, 'w')
    f.write(s)
    #loader.set_word_embedding(word_embed)
    #loader.set_word_prediction_score(word_prob)

sentences_keyword_cache = {}
def sentences_keyword_preprocess(idxs, layer):
    for i in idxs:
        print(i)
        m = loader.word_entropy[i][layer]
        w = get_weight_matrix(m).transpose()[0]
        d = []
        for j in range(len(m)):
            word = loader.data_word[i][j]
            key = stemmer.stem(word)
            if key != '' and key.isalpha():
                d.append((key, word, float(w[j])))
        sentences_keyword_cache[f'{i}_{layer}'] = d

def get_sentences_discarded_keywords(idxs, layer, attrs = 'entropy_frequency_uncertainty'):
    key_weight = {}
    key_frequency = {}
    word_frequency = {}
    apply_entropy = 'entropy' in attrs
    apply_frequency = 'frequency' in attrs
    apply_uncertainty = 'uncertainty' in attrs
    for i in idxs:
        m = loader.word_entropy[i][layer]
        w = get_weight_matrix(m).transpose()[0]
        if layer > 0:
            w2 = get_weight_matrix(loader.word_entropy[i][layer-1]).transpose()[0]
        else:
            w2 = np.ones(len(w)) / len(w)
        w = np.maximum(w2 - w, 0)
        for j in range(len(m)):
            word = loader.data_word[i][j]
            key = stemmer.stem(word)
            if key != '' and key.isalpha():
                if key not in key_weight:
                    key_weight[key] = 0
                    key_frequency[key] = 0
                if word not in word_frequency:
                    word_frequency[word] = 0
                key_weight[key] += w[j]
                key_frequency[key] += 1
                word_frequency[word] += 1
    key2word = {}
    for word in word_frequency:
        key = stemmer.stem(word)
        if key not in key2word:
            key2word[key] = (word, word_frequency[word])
        elif word_frequency[word] > key2word[key][1]:
            key2word[key] = (word, word_frequency[word])
    for key in key2word:
        key2word[key] = key2word[key][0]
    words = []
    for key in key_weight:
        w = 1
        if apply_entropy:
            w *= key_weight[key] / key_frequency[key]
        if apply_frequency:
            w *= math.log2(0.5 + key_frequency[key])
        if apply_uncertainty:
            std = loader.word_prediction_score[key2word[key]]['std']
            w *= abs(std) * 0.8 + 0.2
        words.append((key2word[key], w))
    words = sorted(words, key = lambda x: -x[1])
    ret = []
    for x in words[:200]:
        ret.append({
            'word': x[0],
            'weight': x[1],
            'score' : loader.word_prediction_score[x[0]],
            'embedding' : loader.word_embedding[x[0]],
        })
    return ret

def get_sentences_info(idxs):
    info = np.zeros(loader.n_layer)
    for i in idxs:
        info = info + (np.log2(0.5) - np.log2(loader.layer_entropy[i].mean(axis=1)))
    info = info / len(idxs)
    return info

def get_word_info(word):
    info = np.zeros(loader.n_layer)
    for p in loader.inverted_list[word]:
        info = info + loader.layer_entropy[p[0]][:,p[1]]
    info = info / len(loader.inverted_list[word])
    return info

def disturb_input_embed_gaussian(x, disturb_loc, var = 2, n_batch = 50):
    x = x.unsqueeze(0).repeat(n_batch, 1, 1)
    epsilon = torch.randn((n_batch, x.shape[2])).to(device) * var
    x[:, disturb_loc, :] += epsilon
    return x

if is_sst2_dataset:
    if is_lstm:
        train_loader = DataLoader('/home/lizhen/data', 'sst2_lstm', 4000)#, load_activation=False)
    else:
        train_loader = DataLoader('/home/lizhen/data', 'sst2_10k', 4000)#, load_activation=False)
else:
    train_loader = DataLoader('/home/lizhen/data', 'agnews', 2000)#, load_activation=False)
test_loader = None#DataLoader('/home/lizhen/data', 'sst2_10k', datatype='test')#, load_activation=False)
loader = train_loader
#set_embedding_and_prediction_score(loader)

def cos_sim(a, b):
    #sim = a.dot(b) / np.linalg.norm(a) / np.linalg.norm(b)
    #return sim * 0.5 + 0.5
    return ((a - b) ** 2).sum()

next_threshold = 1.25
avg_threshold = 1.2
count_threshold = 4

def get_top_weight(weight, words, top_k):
    args = weight[1:-1].argsort()
    args = args[::-1]
    args = [x + 1 for x in args]
    args = [x for x in args if words[x] != '']
    avg = weight.mean() * avg_threshold
    args = args[:top_k]
    if weight[args[0]] < avg:
        return []
    for k in range(1, len(args)):
        if weight[args[k]] * next_threshold < weight[args[k - 1]] or weight[args[k]] < avg:
            args = args[:k]
            break
    args = sorted(args)
    return [int(x) for x in args]

def get_frequent_pattern(idx):
    words = loader.data_word[idx][:-1]
    n = len(words)
    pattern_count = {}
    pattern_item = {}
    for layer in range(loader.n_layer):
        head_n = min(4, (layer + 1) // 2 + 1)
        if layer == loader.n_layer - 1:
            head_n = max(head_n, n // 2)
        entropys = get_weight_matrix(loader.word_entropy[idx][layer]).transpose()
        for j in range(1, n):
            args = get_top_weight(entropys[j], words, head_n)
            if len(args) >= 2:
                p = '_'.join([words[x] for x in args])
                if p not in pattern_count:
                    pattern_count[p] = 1
                    pattern_item[p] = [int(x) for x in args]
                else:
                    pattern_count[p] += 1
    pattern_count = [(x, pattern_count[x]) for x in pattern_count]
    patterns = sorted(pattern_count, key=lambda x:-x[1])
    return [(x[0], set(pattern_item[x[0]])) for x in patterns if x[1] >= count_threshold]

def getdist(a, b):
    c = a - b
    return np.sqrt(c.dot(c))

def get_layer_clustering(idx):
    weights = []
    for layer in range(loader.n_layer - 1):
        w = get_weight_matrix(loader.word_entropy[idx][layer][:-1, :-1]).transpose()
        weights.append(w)
    weights = [euclidean_distances(w) for w in weights]
    weights = np.array(weights)
    n = weights.shape[1]
    e = []
    for i in range(1, n):
        for j in range(i + 1, n):
            argmin = weights[:, i, j].argmin()
            e.append((i, j, argmin, weights[argmin, i, j]))
    for i in range(1, n):
        if loader.data_word[idx][i] == '':
            e.append((i - 1, i, 0, 0))
    e = sorted(e, key = lambda x: x[3])

    group = [[i] for i in range(len(loader.data_word[idx]))]
    parent = [-1] * len(group)
    def find_parent(x):
        while parent[x] != -1:
            x = parent[x]
        return x
    def merge(x, y):
        x = find_parent(x)
        y = find_parent(y)
        if x != y:
            if x > y:
                t = x
                x = y
                y = t
            group[x] = group[x] + group[y]
            group[y] = []
            group[x] = sorted(group[x])
            parent[y] = x
            return True
        else:
            return False
    #for layer in range(loader.n_layer):
    tree_edges = set()
    max_tree_edge = 0
    relations = []
    e2 = []
    for p in e:
        if merge(p[0], p[1]):
            tree_edges.add((p[0], p[1]))
            max_tree_edge = max(max_tree_edge, p[3])
            e2.append(p)
        else:
            relations.append(p)
    relations = [x for x in relations if x[3] < max_tree_edge]
    relations = sorted(relations, key = lambda x: x[3])
    relations = relations[:n]

    e = e2
    e = sorted(e, key = lambda x: x[2])
    group = [[i] for i in range(n)]
    parent = [-1] * len(group)
    labels = [None] * loader.n_layer
    #relations = [[] for i in range(n)]

    for i in range(1, n):
        if loader.data_word[idx][i] == '':
            merge(i - 1, i)
    g = [w for w in group if len(w) != 0]
    label0 = [0] * n
    s = []
    for j, cl in enumerate(g):
        for k in cl:
            label0[k] = j
    cnt = 0
    for i, p in enumerate(e):
        if cnt + 2 + 1 < n:
            merge(p[0], p[1])
            cnt += 1
        if i + 1 == len(e) or p[2] != e[i + 1][2]:
            g = [w for w in group if len(w) != 0]
            label = [0] * n
            s = []
            for j, cl in enumerate(g):
                for k in cl:
                    label[k] = j
                s.append(' '.join([loader.data_word[idx][k] for k in cl]))
            labels[p[2]] = label
    labels[0] = label0
    for i in range(1, loader.n_layer):
        if labels[i] == None:
            labels[i] = labels[i - 1]
    return labels, relations

def multi_disturb_input_embed_gaussian(x, disturb_locs, var = 2.5, n_batch = 120):
    x = x.unsqueeze(0).repeat(n_batch, 1, 1)
    for i in disturb_locs:
        epsilon = torch.randn((n_batch, x.shape[2])).to(device) * var
        x[:, i, :] += epsilon
    return x

def get_network_layout_cache(idx):
    labels, relations = get_layer_clustering(idx)
    text = loader.data_text[idx]
    contri_matrix = []
    if is_lstm:
        x0, Phi, tokenized = preprocess_sentence_batch(model=model, types='rnn', sent1=text, target_id = -1, target_layer=6, device=device)
        logits0 = Phi.get_hidden(x0).cpu()
        logits0 = torch.softmax(logits0 * 0.6, dim=1).detach().numpy()
        logits0 = logits0.reshape((logits0.shape[-1]))
        prob = logits0
        prob = [float(t) for t in prob]
        x0 = x0[0]
        d = [0] * (x0.shape[0] - 1)
        for i in range(x0.shape[0] - 1):
            idxs = [i]
            x1 = multi_disturb_input_embed_gaussian(x0, idxs, var=2)
            logits = Phi.get_hidden(x1).cpu()
            logits = torch.softmax(logits * 0.6, dim=1).detach().numpy()
            logits = logits.mean(axis = 0)
            d[i] = [float(t) for t in logits]
        for layer in range(loader.n_layer):
            contri_matrix.append(d)
    else:
        x = get_embedding_from_text(text)
        xe = get_embedding_of_layer(x, start_layer=0, end_layer=loader.n_layer)
        prob = get_logits(xe)
        prob = [float(t) for t in prob]
        for layer in range(loader.n_layer):
            x0 = get_embedding_of_layer(x, start_layer=0, end_layer=layer)
            m = int(max(labels[layer])) + 1
            d = [0] * (x.shape[0] - 1)
            for i in range(x.shape[0] - 1):
                idxs = [i]
                x1 = multi_disturb_input_embed_gaussian(x0, idxs, var=2.5)
                x2 = get_embedding_of_layer(x1, start_layer=layer, end_layer=loader.n_layer)
                d[i] = [float(t) for t in get_logits(x2)]
            contri_matrix.append(d)
    s = json.dumps({ 'logits': prob, 'contris': contri_matrix })
    cache_dir = os.path.join('cache', loader.dataset_name)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    cache_file = os.path.join(cache_dir, f'{idx}.json')
    f = open(cache_file, 'w')
    f.write(s)
    f.close()

def get_phrases(weight, weight0, current, patterns, words, head_n):
    args = get_top_weight(weight, words, head_n)
    retained_keywords = []
    for l in args:
        retained_keywords.append([l, weight0[l] * weight[l], 'self' if l in current else ''])
    wordset = set(args)
    left_keywords = retained_keywords
    for p in patterns:
        flag = False
        if p[1].issubset(wordset) or len(p[1]) >= 4 and len(p[1].intersection(wordset)) + 1 >= len(p[1]):
            flag = True
        if not flag:
            continue
        keywords1 = [x for x in left_keywords if x[0] not in p[1]]
        keywords2 = [x for x in retained_keywords if x[0] in p[1]]
        #wordset = wordset.difference(p[1])
        left_keywords = keywords1 + [(p[0], float(np.mean([x[1] for x in keywords2])), 'self' if [x[2] for x in keywords2].count('self') > 0 else '')]
    left_keywords = [((words[x[0]] if type(x[0]) == int else x[0]), float(x[1]), x[2]) for x in left_keywords]
    return left_keywords

def get_label(arr):
    stk = []
    ret = ''
    def short(stk):
        if len(stk) == 1:
            return str(stk[0])
        elif len(stk) == 2:
            return str(stk[0]) + ',' + str(stk[1])
        else:
            return str(stk[0]) + '-' + str(stk[-1])
    for x in arr:
        if len(stk) > 0 and x > stk[-1] + 1:
            ret += ',' + short(stk)
            stk = []
        stk.append(x)
    if len(stk) > 0:
        ret += ',' + short(stk)
    return ret[1:]

for layer in range(loader.n_layer):
    sentences_keyword_preprocess(range(loader.size), layer)
s = json.dumps(sentences_keyword_cache)
f = open(os.path.join('cache', loader.dataset_name, 'word.json'), 'w')
f.write(s)
f.close()

for i in range(145, 146):
    print(f'processing sample {i}')
    get_network_layout_cache(i)
