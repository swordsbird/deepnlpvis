from tqdm import tqdm
import numpy as np
import os
import nltk.stem
import pandas as pd
import random
import config
from utils import clear_mat, entropy_to_contribution
from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

stemmer = nltk.stem.SnowballStemmer('english')
project_name = 'DeepNLPVis'


class DataLoader:
    def __init__(self, home_path, dataset_name, model_name, size=-1, load_activation=True, datatype='train'):
        self.home_path = os.path.join(home_path, 'data')
        self.dataset_name = dataset_name
        self.project_name = project_name
        self.model_name = model_name
        self.color_scheme = None
        self.xi_values = None
        self.dataset_path = os.path.join(self.home_path, 'dataset')

        if datatype == 'train':
            self.data_path = os.path.join(
                self.dataset_path, self.dataset_name + '_train.tsv')
            self.all_data_path = os.path.join(
                self.dataset_path, self.dataset_name + '_train_all.tsv')
            self.all_label_path = os.path.join(
                self.dataset_path, self.dataset_name + '_train_all.npy')
            self.activation_path = os.path.join(
                self.home_path, 'activation', self.dataset_name + '_sample_activation.npz')
        elif datatype == 'test':
            self.data_path = os.path.join(
                self.dataset_path, self.dataset_name + '_dev.tsv')
            self.activation_path = os.path.join(
                self.home_path, 'activation', self.dataset_name + '_dev.npz')
        self.layer_activation_path = os.path.join(
            self.home_path, 'entropy', self.dataset_name, datatype, 'activation')
        self.layer_entropy_path = os.path.join(
            self.home_path, 'entropy', self.dataset_name, datatype, 'layer')
        self.word_entropy_path = os.path.join(
            self.home_path, 'entropy', self.dataset_name, datatype, 'word')
        self.layer_weight_path = os.path.join(
            self.home_path, 'entropy', self.dataset_name, datatype, 'weight')
        self.contri_path = os.path.join(
            self.home_path, 'entropy', self.dataset_name, datatype, 'contri')
        self.direction_path = os.path.join(
            self.home_path, 'entropy', self.dataset_name, datatype, 'direction')

        self.size = size
        self.load_activation = load_activation
        self.init_cache()
        self.init_data()

    def init_data(self):
        if self.dataset_name[:4] == 'sst2':
            data = open(self.data_path, 'r').read().split('\n')
            print('data_path', self.data_path)
            data = data[1:]
            if self.size == -1:
                self.size = len(data)
            else:
                data = data[:self.size]
            data = [x.split('\t') for x in data]
            data = [x for x in data if len(x) == 2]
            self.size = len(data)
            self.data_text = [x[0] for x in data]
            self.data_label = [x[1] for x in data]

            data = open(self.all_data_path, 'r').read().split('\n')
            data = data[1:]
            data = [x.split('\t') for x in data]
            data = [x for x in data if len(x) == 2]
            self.all_data_text = [x[0] for x in data]
            self.all_data_label = [x[1] for x in data]
            self.all_data_text = self.all_data_text
            self.all_data_label = self.all_data_label
            self.data_labels = ['0', '1']
            self.data_label_name = ['Negative', 'Positive']
            self.label_index = {}
            for i in range(len(self.data_labels)):
                self.label_index[self.data_labels[i]] = i
        elif self.dataset_name[:6] == 'agnews':
            table = pd.read_csv(self.data_path, sep='\t')
            title = [x for x in table['Title']]
            description = [x for x in table['Description']]
            #text = [title[i] + ' ' + description[i] for i in range(table.shape[0])]
            text = [title[i] + (' ' if len(tokenizer.tokenize(title[i] + ' ' + description[i])) == len(tokenizer.tokenize(title[i] + description[i])) else '') + description[i] for i in range(table.shape[0])]
            labels = [str(x) for x in table['Class Index']]
            if self.size == -1:
                self.size = len(text)
            else:
                text = text[:self.size]
            self.size = len(text)
            self.data_text = text
            self.data_label = labels
            self.data_pred = np.load(self.all_label_path)
            table = pd.read_csv(self.all_data_path, sep='\t')
            title = [x for x in table['Title']]
            description = [x for x in table['Description']]
            text = [title[i] + ' ' + description[i] for i in range(table.shape[0])]
            labels = [x for x in table['Class Index']]
            self.all_data_text = text
            self.all_data_label = labels
            self.all_data_text = self.all_data_text[:20000]
            self.all_data_label = self.all_data_label[:20000]
            self.data_labels = ['1', '2', '3', '4']
            self.data_label_name = ['World', 'Sports', 'Business', 'Sci/Tech']
            self.label_index = {}
            for i in range(len(self.data_labels)):
                self.label_index[self.data_labels[i]] = i
        self.n_classes = len(self.data_labels)
        print('loading entropy for each layer')
        self.layer_entropy = []
        self.layer_contri = []
        self.word_entropy = []
        self.contri_value = []
        self.sigma = []
        self.logits = []
        self.layer_weight = []
        self.word_context = []
        self.sample_logits = []
        self.sample_emb = []
        self.word_info_by_layer = {}
        self.n_layer = 0
        if 'lstm' in self.dataset_name:
            self.n_layer = 4

        print('self.load_activation', self.load_activation)
        for i in tqdm(range(self.size)):
            file_path_1 = os.path.join(self.layer_entropy_path, f'{i}.npy')
            file_path_2 = os.path.join(
                self.word_entropy_path, f'{i}.npy')
            file_path_3 = os.path.join(
                self.layer_activation_path, f'{i}logits.npy')
            file_path_4 = os.path.join(
                self.contri_path, f'{i}.npy')
            file_path_6 = os.path.join(
                self.layer_weight_path, f'{i}.npy')
            file_path_7 = os.path.join(self.layer_activation_path, f'{i}.npy')
            file_path_8 = os.path.join(
                self.layer_activation_path, f'{i}logits.npy')
            file_path_9 = os.path.join(
                self.layer_activation_path, f'{i}emb.npy')
            file_paths = [file_path_1, file_path_2, file_path_3,
                          file_path_4, file_path_6]

            flag = False
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    flag = True

            if flag:
                self.data_text[i] = ''
                self.data_label[i] = ''
                self.size -= 1
                continue

            entropy = np.load(file_path_1)
            self.layer_entropy.append(entropy)

            entropy = np.load(file_path_2, allow_pickle=True)
            n_layer = entropy.shape[0] // entropy.shape[1]
            if self.n_layer == 0:
                self.n_layer = n_layer
            entropy = entropy.reshape(
                (n_layer, entropy.shape[-1], entropy.shape[-1]))
            self.word_entropy.append(entropy)
            
            if self.model_name == 'bert':
                self.layer_entropy[-1] = np.concatenate(
                    (self.layer_entropy[-1], entropy[-1, :1]), axis=0
                )
            self.layer_contri.append(entropy_to_contribution(self.layer_entropy[-1]))
            
            entropy = np.load(file_path_3)
            self.logits.append(entropy)

            entropy = np.load(file_path_4)
            self.sigma.append(entropy)
            self.contri_value.append(entropy_to_contribution(entropy.copy()))

            entropy = np.load(file_path_6, allow_pickle=True)
            n_layer = entropy.shape[0] // entropy.shape[1]
            if self.n_layer == 0:
                self.n_layer = n_layer
            entropy = entropy.reshape(
                (n_layer, entropy.shape[-1], entropy.shape[-1]))
            self.layer_weight.append(entropy)
            
            if self.load_activation:
                entropy = np.load(file_path_7)
                self.word_context.append(entropy)
                entropy = np.load(file_path_8)
                if entropy.shape[0] == 1:
                    entropy = entropy.reshape((entropy.shape[1]))
                self.sample_logits.append(entropy)
                if os.path.exists(file_path_9):
                    entropy = np.load(file_path_9)
                    if entropy.shape[0] == 1:
                        entropy = entropy.reshape((entropy.shape[1]))
                    self.sample_emb.append(entropy)
        self.data_text = [x for x in self.data_text if x != '']
        self.data_label = [str(x) for x in self.data_label if x != '']
        self.data_token = []
        self.data_word = []
        self.data_word_lens = []
        self.inverted_list = {}
        self.original_inverted_list = {}
        print('tokenize all sentences')
        max_len = 0
        for i in tqdm(range(self.size)):
            tokens = []
            tokens_ = ["[CLS]"] + \
                tokenizer.tokenize(self.data_text[i]) + ["[SEP]"]
            words = [''] * len(tokens_)
            lens = [0] * len(tokens_)
            max_len = max(max_len, len(tokens_))
            w = ''
            idxs = []
            for j in range(len(tokens_) - 2, 0, -1):
                t = tokens_[j]
                idxs.append(j)
                if t[:2] == '##':
                    lens[j] = len(t) - 2
                else:
                    lens[j] = len(t)
                if t[:2] == '##':
                    w = t[2:] + w
                    continue
                elif w != '':
                    t = t + w
                    w = ''
                else:
                    w = ''
                words[j] = t
                tokens.append((t, tuple(idxs)))
                idxs = []
            tokens = tokens[::-1]
            for j, w in enumerate(words):
                k = stemmer.stem(w)
                if w == '' or j == 0 or j == len(tokens_) - 1:
                    continue
                if k not in self.inverted_list:
                    self.inverted_list[k] = []
                if w not in self.original_inverted_list:
                    self.original_inverted_list[w] = []
                self.inverted_list[k].append((i, j))
                self.original_inverted_list[w].append((i, j))
            self.data_token.append(tokens)
            self.data_word.append(words)
            self.data_word_lens.append(lens)
        print('average length', np.mean([len(x) for x in self.data_text]))

        self.all_inverted_list = {}
        self.all_word_labels = {}
        self.all_data_word = []
        for i in tqdm(range(len(self.all_data_text))):
            tokens = []
            tokens_ = ["[CLS]"] + \
                tokenizer.tokenize(self.all_data_text[i]) + ["[SEP]"]
            words = [''] * len(tokens_)
            lens = [0] * len(tokens_)
            w = ''
            idxs = []
            for j in range(len(tokens_) - 2, 0, -1):
                t = tokens_[j]
                idxs.append(j)
                if t[:2] == '##':
                    lens[j] = len(t) - 2
                else:
                    lens[j] = len(t)
                if t[:2] == '##':
                    w = t[2:] + w
                    continue
                elif w != '':
                    t = t + w
                    w = ''
                else:
                    w = ''
                words[j] = t
                tokens.append((t, tuple(idxs)))
                idxs = []
            tokens = tokens[::-1]
            for j, w in enumerate(words):
                if w == '' or j == 0 or j == len(tokens_) - 1:
                    continue
                k = w
                if k not in self.all_inverted_list:
                    self.all_inverted_list[k] = []
                    self.all_word_labels[k] = []
                self.all_inverted_list[k].append((i, j))
                self.all_word_labels[k].append(self.all_data_label[i])
            self.all_data_word.append(words)

        self.word_frequency = self.get_word_frequency(range(self.size))
        self.all_layer_entropy = [self.get_word_layer_entropy(
            range(self.size), layer) for layer in range(self.n_layer)]
        print('max_len', max_len)

    def init_cache(self):
        if not os.path.exists('data/cache'):
            os.mkdir('data/cache', parents=True)
        self.cache = {}
        self.cache['linechart_idxes'] = None
        self.cache['linechart_attrs'] = None

    def set_pred_label(self, labels):
        self.pred_label = labels
        cnt = 0
        for i in range(self.size):
            if labels[i] == self.data_label[i]:
                cnt += 1
        print('Accuracy on Training Data', round(cnt / self.size * 100, 2))

    def get_confusion_matrix(self):
        n = len(self.data_labels)
        data = []
        for i in range(n):
            data.append([0 for _ in range(n)])
        idx = {}
        for i, x in enumerate(self.data_labels):
            idx[x] = i
        for i in range(self.size):
            data[idx[self.data_label[i]]][idx[self.pred_label[i]]] += 1
        return {'data': data, 'labels': self.data_label_name}

    # data embedding -- instance's x coordinate
    # data prediction score -- instance's y coordinate
    # word embedding -- word's x coordinate
    # word prediction score -- word's y coordinate
    def set_word_prediction_score(self, scores):
        self.word_prediction_score = scores

    def set_instance_prediction_score(self, scores):
        self.data_prediction_score = scores

    def set_word_embedding(self, v):
        self.word_embedding = v

    def set_instance_embedding(self, v):
        self.data_embedding = v

    def set_main_index(self, v):
        self.main_index = v

    def set_second_index(self, v):
        self.second_index = v

    def selected_idxes(self):
        return [self.main_index, self.second_index]

    def selected_labels(self):
        return [self.data_labels[idx] for idx in self.selected_idxes()]

    def set_threshold_xi(self, v):
        self.threshold_xi = v

    def set_model_name(self, v):
        self.model_name = v

    def calc_polarity(self):
        self.polarity_mat = []
        for idx in range(self.size):
            old_s = self.all_old_s[idx]
            delta_mat = []
            agg_ns = []
            for layer in range(self.n_layer):
                for _, new_s in enumerate(self.all_new_s[idx][layer]):
                    ns = new_s[self.main_index] / (new_s[self.main_index] + new_s[self.second_index])
                    agg_ns.append(ns)
            std_ns = np.std(agg_ns)
            for layer in range(self.n_layer):
                s = old_s[self.main_index] / (old_s[self.main_index] + old_s[self.second_index])
                delta_row = []
                for _, new_s in enumerate(self.all_new_s[idx][layer]):
                    ns = new_s[self.main_index] / (new_s[self.main_index] + new_s[self.second_index])
                    ds = ns - s
                    delta_row.append(ds / std_ns)
                delta_mat.append(delta_row)
            self.polarity_mat.append(np.array(delta_mat))

    def calc_distribution(self):
        if self.xi_values != None:
            return
        xi_values = []
        for i in range(len(self.polarity_mat)):
            shape = self.polarity_mat[i].shape
            xi_values = xi_values + self.polarity_mat[i].reshape((shape[0] * shape[1])).tolist()
        xi_values = random.sample(xi_values, config.n_distribution_sample)
        self.xi_values = [float(x) for x in xi_values]

    def calc_all_layer_delta_s(self):
        clear_mat(self)
        dicts = [self.get_word_delta_s_by_layer(
            range(self.size), layer) for layer in range(self.n_layer)]
        self.layer_word_contri = dicts

    def get_word_delta_s_by_layer(self, idxs, layer = -1, is_stem = False):
        delta_s = {}
        for i in idxs:
            for t in self.data_token[i]:
                w = stemmer.stem(t[0]) if is_stem else t[0]
                if w not in delta_s:
                    delta_s[w] = []
                if len(t[1]) == 1:
                    ds = self.polarity_mat[i][layer,t[1][0]]
                else:
                    ds = self.polarity_mat[i][layer,t[1]].sum()
                delta_s[w].append(ds)
        for t in delta_s:
            tot = len(delta_s[t])
            neg = len([x for x in delta_s[t] if x > self.threshold_xi])
            pos = len([x for x in delta_s[t] if x < -self.threshold_xi])
            delta_s[t] = {
                'avg': float(np.mean(delta_s[t])),
                'neg': neg,
                'pos': pos,
                'neu': tot - neg - pos,
            }

        return delta_s

    def get_word_layer_entropy(self, idxs, layer=-1):
        entropy = {}
        for i in idxs:
            for t in self.data_token[i]:
                if t[0] not in entropy:
                    entropy[t[0]] = []
                e = self.layer_entropy[i][layer, t[1]].min()
                entropy[t[0]].append(e)
        for t in entropy:
            entropy[t] = float(np.mean(entropy[t]))
        return entropy

    def get_word_frequency(self, idxs):
        frequency = {}
        for i in idxs:
            for t in self.data_token[i]:
                frequency[t[0]] = frequency.get(t[0], 0) + 1
        return frequency

    def is_overview(self, idxs):
        return len(idxs) > self.size // 10

    def is_correct(self, i):
        return self.data_label[i] == self.pred_label[i]
