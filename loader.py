
import json
import os
import numpy as np
from data_loader import DataLoader
from nltk.corpus import stopwords
import config
from config import stop_words as custom_stop_words
from utils import get_weight_matrix, entropy_to_contribution, stem 

# determining the coordinates of each instance in the distribution view


def set_instance_coordinates(loader):
    data_embed_file = os.path.join(loader.home_path, 'cache', loader.dataset_name, 'data_embed.json')
    f = open(data_embed_file, 'r')
    s = f.read()
    data_embed = json.loads(s)
    # data embedding -- instance's x coordinate
    loader.set_instance_embedding(data_embed)
    data_prob_file = os.path.join(loader.home_path, 'cache', loader.dataset_name, 'data_prob.json')
    f = open(data_prob_file, 'r')
    s = f.read()
    data_prob = json.loads(s)
    # data prediction score -- instance's y coordinate
    loader.set_instance_prediction_score(data_prob)
    pred_labels = []
    for i in range(loader.size):
        pred_labels.append(loader.data_labels[np.argmax(data_prob[i])])
    loader.set_pred_label(pred_labels)

# determining the coordinates of each word in the distribution view


def set_word_coordinates(loader):
    word_embed_file = os.path.join(loader.home_path, 'cache', loader.dataset_name, 'word_embed.json')
    f = open(word_embed_file, 'r')
    s = f.read()
    word_embed = json.loads(s)
    # word embedding -- word's x coordinate
    loader.set_word_embedding(word_embed)
    word_prob_file = os.path.join(loader.home_path, 'cache', loader.dataset_name, 'word_prob.json')
    f = open(word_prob_file, 'r')
    s = f.read()
    word_prob = json.loads(s)
    # word prediction score -- word's y coordinate
    loader.set_word_prediction_score(word_prob)


def set_prediction(loader):
    loader.all_new_s = []
    loader.all_old_s = []
    for idx in range(loader.size):
        cache_dir = os.path.join(loader.home_path, 'cache', loader.dataset_name)
        cache_file = os.path.join(cache_dir, f'{idx}.json')
        cache_data = open(cache_file, 'r').read()
        cache_data = json.loads(cache_data)
        new_scores = cache_data['contris']
        old_s = cache_data['logits']
        instance_new_s = []
        for layer in range(loader.n_layer):
            layer_new_s = []
            for new_s in new_scores[layer]:
                layer_new_s.append(new_s)
            instance_new_s.append(layer_new_s)
        loader.all_new_s.append(instance_new_s)
        loader.all_old_s.append(old_s)


def layer_weight_preprocess(loader, idxs, layer):
    if layer not in loader.word_info_by_layer:
        loader.word_info_by_layer[layer] = {}
    for i in idxs:
        weight = np.array(loader.layer_entropy[i][layer + 1])
        mat = get_weight_matrix(weight, normalize=False).transpose()
        mat = mat / mat.sum()
        row = []
        for t in loader.data_token[i]:
            if len(t[1]) == 1:
                contribution = entropy_to_contribution(weight[t[1][0]])
            else:
                contribution = entropy_to_contribution(np.min(weight[t[1], ]))
            contribution = float(contribution)
            norm_contribution = np.sum(mat[t[1], ])
            norm_contribution = float(norm_contribution)
            word = t[0]
            stem_word = stem(word)
            if stem_word != '' and stem_word.isalpha():
                row.append((stem_word, word, contribution, norm_contribution))
        loader.word_info_by_layer[layer][i] = row


def weight_preprocess(loader):
    for layer in range(loader.n_layer):
        layer_weight_preprocess(loader, range(loader.size), layer)

def update_loader(loader, indexes = None):
    if not indexes is None:
        [main_index, second_index] = indexes
        loader.set_main_index(main_index)
        loader.set_second_index(second_index)
    loader.calc_polarity()
    loader.calc_distribution()
    loader.calc_all_layer_delta_s()

def basic_init_loader():
    loader = DataLoader(config.home_path, config.dataset_name, config.model_name, config.n_samples)
    return loader

def init_loader():
    loader = DataLoader(config.home_path, config.dataset_name, config.model_name, config.n_samples)
    set_instance_coordinates(loader)
    set_word_coordinates(loader)
    loader.threshold_xi = config.threshold_xi
    stop_words = []
    #stop_words = stopwords.words('english')
    stop_words = set(stop_words + custom_stop_words)
    loader.stop_words = stop_words
    set_prediction(loader)
    weight_preprocess(loader)
    update_loader(loader, [1, 0])
    return loader
