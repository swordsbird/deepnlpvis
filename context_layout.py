import numpy as np
import random
import config
from utils import get_weight_matrix, stem, cross
from sklearn.cluster import AgglomerativeClustering
from keyword_extractor import get_sentences_keywords


def clustering_for_word(loader, word, inverted_list):
    distance_step = 1.1
    labels = []
    thres = config.context_distance_threshold

    word_index = {}
    word_curr_index = 0
    for p in inverted_list:
        for q in loader.data_token[p[0]]:
            w = stem(q[0])
            if w not in word_index:
                word_index[w] = word_curr_index
                word_curr_index += 1
    print('total words on', word, word_curr_index)
    min_clusters = config.context_min_n_clusters + 1
    max_clusters = config.context_max_n_clusters
    deep_layer = min(loader.n_layer - 2, loader.n_layer * 3 // 4)

    for layer in range(loader.n_layer):
        x = []
        tot = 0
        if layer < deep_layer:
            for p in inverted_list:
                if p[0] < len(loader.word_context):
                    vec = np.concatenate((loader.word_context[p[0]][layer, p[1]], np.array(
                        (0.0 * loader.data_prediction_score[p[0]][loader.main_index], 0.02 * loader.data_embedding[p[0]]))))
                    x.append(vec)
        else:
            for p in inverted_list:
                if p[0] < len(loader.word_context):
                    vec = np.concatenate((loader.word_context[p[0]][layer, p[1]], np.array(
                        (0.1 * loader.data_prediction_score[p[0]][loader.main_index], 0.02 * loader.data_embedding[p[0]]))))
                    x.append(vec)
                tot += loader.data_prediction_score[p[0]][loader.main_index]
        tot /= len(inverted_list)
        n_clusters = max_clusters + 1
        if layer == deep_layer:
            min_clusters -= 1
        thres *= distance_step
        loop = 0

        while n_clusters > max_clusters or n_clusters < min_clusters:
            if n_clusters > max_clusters:
                thres *= distance_step
            else:
                thres /= distance_step
            clustering = AgglomerativeClustering(
                distance_threshold=thres, n_clusters=None, linkage="ward").fit(x)
            n_clusters = clustering.labels_.max() + 1
            loop += 1
        if layer == loader.n_layer - 1 and n_clusters > min_clusters + 1:
            if tot < 0.15 or tot > 0.85:
                clustering = AgglomerativeClustering(n_clusters=1, linkage="ward").fit(x)
                n_clusters = clustering.labels_.max() + 1
            else:
                clustering = AgglomerativeClustering(n_clusters=min_clusters, linkage="ward").fit(x)
                n_clusters = clustering.labels_.max() + 1
        labels.append(clustering.labels_)

    return labels




def get_frequent_pattern_for_keyword(loader, idx, n_layer, keyword):
    words = loader.data_word[idx]
    n = len(words)
    pattern_count = {}
    pattern_item = {}
    for layer in range(n_layer):
        head_n = 3
        entropys = get_weight_matrix(
            loader.word_entropy[idx][layer], normalize=True).transpose()
        for j in range(1, n):
            entropy = entropys[j]
            args = entropy[1:-1].argsort()
            args = args[::-1]
            args = [x + 1 for x in args]
            args = [x for x in args if words[x] != '']
            avg = entropy.mean() * (config.frequent_phrase_step + 0.2)
            args = args[:head_n]
            for k in range(2, len(args)):
                if entropy[args[k]] * config.frequent_phrase_step < entropy[args[k - 1]] or entropy[args[k]] < avg:
                    args = args[:k]
                    break
            args = sorted(args)
            args = [int(x) for x in args if words[x].isalpha()]
            if len(args) >= 2:
                p = '_'.join([words[x] for x in args])
                if keyword not in p:
                    continue
                if p not in pattern_count:
                    pattern_count[p] = 1
                    pattern_item[p] = args
                else:
                    pattern_count[p] += 1
    pattern_count = [(x, pattern_count[x]) for x in pattern_count]
    patterns = sorted(pattern_count, key=lambda x: -x[1])
    return [x for x in patterns if x[1] >= 1][:1]


def get_wordcontext_layout(loader, layers, word):
    if loader.n_layer <= config.word_context_max_layers:
        layers = [i for i in range(loader.n_layer)]
    else:
        layers = [min(i, loader.n_layer - 1) for i in range(0, loader.n_layer + 1,
                                                            loader.n_layer // (config.word_context_max_layers - 1))]
    original_word = word
    stemmed = False
    if word in loader.original_inverted_list and len(loader.original_inverted_list[word]) > 20:
        pass
    else:
        word = stem(word)
        stemmed = True
    inverted_list = loader.inverted_list[word] if stemmed else loader.original_inverted_list[word]
    print(stemmed, inverted_list)
    labels = clustering_for_word(loader, word, inverted_list)
    m = len(labels[0])
    neurons = []
    edges = []
    neuron_id = 0
    layer_neuron = {}
    neuron_dict = {}
    max_position = max([p[1] for p in inverted_list]) + 1
    position_sum = [0] * max_position
    for p in inverted_list:
        position_sum[p[1]] += 1
    for i in range(1, len(position_sum)):
        position_sum[i] += position_sum[i - 1]
    n_clusters = config.context_max_n_clusters - 1
    i = 0
    j = 1
    position_clusters = []
    current_label = 0
    for l in range(m):
        labels[0][l] = -1
    for k in range(n_clusters):
        while position_sum[i] < m * (k + 1) / n_clusters:
            i += 1
        if i < j:
            continue

        if i == j:
            position_clusters.append([f'{original_word}$in position {j}'])
        else:
            position_clusters.append(
                [f'{original_word}$in position {j} to {i}'])
        for l in range(m):
            p = inverted_list[l]
            if p[1] >= j and p[1] <= i:
                labels[0][l] = current_label
        j = i + 1
        current_label += 1

    for lid, layer in enumerate(layers[:-1]):
        n1 = int(labels[layer].max() + 1)
        n2 = int(labels[layers[lid + 1]].max() + 1)
        weight = np.zeros((n1, n2))
        for i in range(len(labels[layer])):
            weight[labels[layer][i], labels[layers[lid + 1]][i]] += 1
        positions = [i for i in range(n2)]
        for try_times in range(100):
            if n2 < 2:
                continue
            if n2 == 2:
                k = 0
            else:
                k = random.randint(0, n2 - 2)
            v1 = weight[:, positions[k]]
            v2 = weight[:, positions[k + 1]]
            c1 = cross(v1, v2)
            c2 = cross(v2, v1)
            if c1 > c2:
                t = positions[k]
                positions[k] = positions[k + 1]
                positions[k + 1] = t
            if n2 == 2:
                break
        labels_ = np.zeros(m)
        for k in range(n2):
            for i in range(m):
                if labels[layers[lid + 1]][i] == positions[k]:
                    labels_[i] = k
        for i in range(m):
            labels[layers[lid + 1]][i] = labels_[i]

    for lid, layer in enumerate(layers):
        layer_neuron[layer] = []
        n_clusters = labels[layer].max() + 1
        for k in range(n_clusters):
            patterns = []
            size = 1
            idxs = [inverted_list[i][0]
                    for i in range(m) if labels[layer][i] == k]
            for i in idxs:
                size += 1
                pattern = get_frequent_pattern_for_keyword(
                    loader, i, n_layer=layer, keyword=word)
                pattern = [x for x in pattern if word in x[0]]
                patterns += pattern
            pattern_count = {}
            for p in patterns:
                if p[0] not in pattern_count:
                    pattern_count[p[0]] = 0
                pattern_count[p[0]] += p[1]
            patterns = [(x, pattern_count[x]) for x in pattern_count]
            patterns = sorted(patterns, key=lambda x: -x[1])
            new_patterns = []
            for i in range(len(patterns)):
                flag = True
                for j in range(i - 1):
                    if patterns[i][0] in patterns[j][0]:
                        flag = False
                    break
                if flag:
                    new_patterns.append(patterns[i])
            patterns = new_patterns[:4]
            pattern_str = '_'.join([x[0] for x in patterns])
            patterns = [(x[0], x[1], 'self') for i, x in enumerate(patterns)]
            if layer == 0:
                patterns.append((position_clusters[k][0], 1, 'self'))
            else:
                keywords = get_sentences_keywords(loader, idxs, layer)
                ratio = patterns[0][1] / keywords[0]['value'] * \
                    0.75 if len(patterns) > 0 else 1
                keywords = [(x['word'], x['value'] * ratio, 'self') for x in keywords if x['entropy'] > .55][:15]
                keywords = [x for x in keywords if x[0] not in pattern_str]
                patterns = patterns + keywords

            name = original_word + ' - ' + str(neuron_id)
            if layer == 0:
                name = position_clusters[k][0]
            if len(idxs) == 0:
                continue
            if layer > 0:
                contris = loader.get_word_delta_s_by_layer(idxs, layer, is_stem = True)
            scores = []
            for i in idxs:
                s1 = loader.data_prediction_score[i][loader.main_index]
                s2 = loader.data_prediction_score[i][loader.second_index]
                sum_s = s1 + s2
                s1 /= sum_s
                s2 /= sum_s
                scores.append(s1)
            neuron = {
                'id': neuron_id,
                'rank': [k],
                'idxs': idxs,
                'label': name,
                'token': name,
                'layer_id': layer + 1,
                'retained_keywords': patterns,
                'discarded_keywords': [],
                'in_edges': [],
                'out_edges': [],
                'prediction_score': float(np.mean(scores)),
                'contri': contris[stem(word)] if layer > 0 else {'pos': 0, 'neg': 0, 'neu': len(idxs), 'avg': 0},
                'size': size / m,
            }
            neurons.append(neuron)
            layer_neuron[layer].append(neuron_id)
            neuron_id += 1
            neuron_dict[neuron['id']] = neuron
        if layer < config.flow_deep_layer:
            continue
        max_count = {}
        for k in range(n_clusters):
            for p in neurons[-k]['retained_keywords']:
                if p[0] not in max_count or max_count[p[0]] < p[1]:
                    max_count[p[0]] = p[1]
    for neuron in neurons:
        neuron['retained_keywords'] = [
            (x[0], x[1], x[2]) for x in neuron['retained_keywords']]

    edges = []
    nid = 0
    for lid, layer in enumerate(layers[:-1]):
        n1 = int(labels[layer].max() + 1)
        n2 = int(labels[layers[lid + 1]].max() + 1)
        weight = np.zeros((n1, n2))
        for i in range(len(labels[layer])):
            weight[labels[layer][i], labels[layers[lid + 1]][i]] += 1
        for i in range(n1):
            for j in range(n2):
                if weight[i, j] > 0:
                    eid = len(edges)
                    w = weight[i, j]
                    e = {
                        'id': eid,
                        'source': nid + i,
                        'target': nid + n1 + j,
                        'weight': float(w),
                    }
                    edges.append(e)
                    neuron_dict[nid + i]['out_edges'].append(eid)
                    neuron_dict[nid + n1 + j]['in_edges'].append(eid)
        nid += n1
    return neurons, edges
