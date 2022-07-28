import os
import numpy as np
import json
from pytorch_pretrained_bert import BertModel, BertTokenizer, BertForSequenceClassification
import torch
import yaml
from sklearn.manifold import TSNE
from model.model_elmo import elmo_RNNClassifier
from model.basic_model_helper import preprocess_sentence_batch
from loader import init_loader
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
home_path = '/home/lizhen/data/'

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

if config.model_name == 'lstm':
    model = elmo_RNNClassifier(yaml.load(open(os.path.join('cache', 'LSTM_basic_4_layer_768_hidden_pool_max.yaml'),'r')))
    model.load_state_dict(torch.load(os.path.join('cache', '_lstm_basic_SST2_2_100_0.8796.param'), map_location='cpu'))
    model.to(device)
else:
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    if config.data_name == 'sst2':
        state_dict = torch.load(home_path + 'model/sst2/bert_sst2t_5_100_0.9323.param',map_location='cpu')
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

    if config.data_name == 'sst2':
        bert_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2)
    else:
        bert_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 4)
    bert_classifier.load_state_dict(state_dict)
    bert_classifier.to(device)
    for param in bert_classifier.parameters():
        param.requires_grad = False
    bert_classifier.eval()

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
    bert_model.load_state_dict(state_dict)
    bert_model.to(device)

    for param in bert_model.parameters():
        param.requires_grad = False
    bert_model.eval()

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
    x = bert_model.embeddings(token_tensor, segment_tensor)[0]
    return x

def get_logits_from_model(text):
    words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
    tokenized_ids = tokenizer.convert_tokens_to_ids(words)
    mask = [1 for _ in range(len(words))]
    x = torch.tensor([tokenized_ids], dtype = torch.long)
    mask = torch.tensor([mask], dtype = torch.long)
    x = x.cuda()
    mask = mask.cuda()
    ret = bert_classifier(x, attention_mask=mask)
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
    model_list = bert_model.encoder.layer[start_layer:end_layer]
    hidden_states = x
    for layer_module in model_list:
        hidden_states = layer_module(hidden_states, extended_attention_mask)
    return hidden_states[0] if no_batch else hidden_states

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
            logits = torch.softmax(torch.tensor(logits) * 0.6, dim=0).numpy()
        else:
            v = loader.word_activation[i][-1, 0]
            logits = get_logits_from_CLS(v)
        train_x.append(v)
        train_y.append([float(k) for k in logits])
        train_label.append(int(loader.data_label[i]))
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

loader = init_loader()
def cos_sim(a, b):
    return ((a - b) ** 2).sum()

def multi_disturb_input_embed_gaussian(x, disturb_locs, var = 2.5, n_batch = 120):
    x = x.unsqueeze(0).repeat(n_batch, 1, 1)
    for i in disturb_locs:
        epsilon = torch.randn((n_batch, x.shape[2])).to(device) * var
        x[:, i, :] += epsilon
    return x

def get_network_layout_cache(idx):
    text = loader.data_text[idx]
    contri_matrix = []
    if config.model_name == 'lstm':
        x0, Phi, _ = preprocess_sentence_batch(model=model, types='rnn', sent1=text, target_id = -1, target_layer=6, device=device)
        prob = Phi.get_hidden(x0).cpu()
        prob = torch.softmax(prob * 0.6, dim=1).detach().numpy()
        prob = prob.reshape((prob.shape[-1]))
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
            d = [0] * (x.shape[0] - 1)
            for i in range(x.shape[0] - 1):
                idxs = [i]
                x1 = multi_disturb_input_embed_gaussian(x0, idxs, var=2.5)
                x2 = get_embedding_of_layer(x1, start_layer=layer, end_layer=loader.n_layer)
                d[i] = [float(t) for t in get_logits(x2)]
            contri_matrix.append(d)
    s = json.dumps({ 'logits': prob, 'contris': contri_matrix })
    return
    cache_dir = os.path.join('cache', loader.dataset_name)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    cache_file = os.path.join(cache_dir, f'{idx}.json')
    f = open(cache_file, 'w')
    f.write(s)
    f.close()

for i in range(loader.size):
    print(f'processing sample {i}')
    get_network_layout_cache(i)
