import os
import json
import config
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from utils import get_weight_matrix


max_int = 1e8

def aggregate(m, idxs, index=-1):
    shape = list(m.shape)
    shape[index] = 1
    ret = []
    for t in idxs:
        if len(shape) == 2:
            if index == 1 or index == -1:
                ret.append(m[:, t].mean(axis=index).reshape(shape))
            else:
                ret.append(m[t, :].mean(axis=index).reshape(shape))
        else:
            if index == 2 or index == -1:
                ret.append(m[:, :, t].mean(axis=index).reshape(shape))
            elif index == 1:
                ret.append(m[:, t, :].mean(axis=index).reshape(shape))
            else:
                ret.append(m[t, :, :].mean(axis=index).reshape(shape))
    return np.concatenate(ret, axis=index)


def get_word_preclustering(loader, idx):
    layer = loader.n_layer // 2
    norm_contri = get_weight_matrix(
        loader.word_entropy[idx][layer][:-1, :-1]).transpose()
    dist = euclidean_distances(norm_contri)
    idxs = [[i] for i in range(norm_contri.shape[0])]
    for _ in range(norm_contri.shape[0] - config.flow_n_lines):
        min_dist = 5
        k = -1
        for i in range(1, len(idxs) - 1):
            if len(idxs[i]) + len(idxs[i + 1]) > config.flow_max_phrase_len:
                continue
            if dist[idxs[i][0], idxs[i + 1][-1]] < min_dist:
                min_dist = dist[idxs[i][0], idxs[i + 1][-1]]
                k = i
        if k == -1:
            break
        idxs = idxs[:k] + [idxs[k] + idxs[k + 1]] + idxs[k + 2:]
    return idxs


def get_layer_clustering(loader, word, weights):
    weights = [euclidean_distances(weights[i])
               for i in range(weights.shape[0])]
    weights = np.array(weights)
    n = weights.shape[1]
    thres_factor = 1.2
    conns = []
    for i in range(1, weights.shape[1] - 1):
        conns.append((weights[-1, i, i + 1], i))

    conns = sorted(conns, key=lambda x: x[0])

    top_50_percent = int(len(conns) * 0.5)
    top_75_percent = int(len(conns) * 0.75)
    top_50_weight = conns[top_50_percent][0]
    #print(top_50_weight, top_75_weight)
    merge_points = []
    merge_mark = [False] * (weights.shape[1] - 1)
    for i in range(len(conns)):
        x = conns[i][1]
        if i > top_75_percent or i > top_50_percent and conns[i][0] / conns[i - 1][0] > 1.2:
            break
        if word[x + 1] == "" or word[x] != ',' and word[x] != '.' and word[x + 1] != ',' and word[x + 1] != '.':
            y = x - 1
            m = 1
            while y >= 0 and merge_mark[y]:
                y -= 1
                m += 1
            y = x + 1
            while y < len(merge_mark) and merge_mark[y]:
                y += 1
                m += 1
            # if m > len(merge_mark) * 0.33:
            #    continue
            merge_points.append(x)
            merge_mark[x] = True
    e = [[]]
    for l in range(loader.n_layer):
        e.append([])
    if len(merge_points) + 5 > n:
        merge_points = merge_points[:n - 5]
    for i in merge_points:
        argmin = weights[:, i, i + 1].argmin()
        thres = weights[argmin, i, i + 1] * thres_factor
        if thres < top_50_weight:
            thres = top_50_weight
        for l in range(weights.shape[0]):
            if weights[l, i, i + 1] < thres:
                e[l + 1].append(i)
                break

    group = [[i] for i in range(n)]
    parent = [-1] * n

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

    labels = []
    for l in range(loader.n_layer):
        for i in e[l]:
            merge(i, i + 1)
        g = [x for x in group if len(x) != 0]
        label = [0] * n
        label.append(-1)
        for i in range(len(g)):
            for x in g[i]:
                label[x] = i
        labels.append(label)
    group = [x for x in group if len(x) != 0]
    for g in group:
        print(' '.join([word[i] for i in g]))
    return labels


def get_non_adjacent_relations(word, norm_contri, contri, polarity, labels):
    word[0] = 'CLS'
    norm_contri = np.array(norm_contri)
    labels = np.array(labels)
    display = np.ones(contri.shape) > 0
    for i in range(0, contri.shape[0] - 2):
        layer = contri.shape[0] - 1 - i
        for j in range(1, contri.shape[1]):
            print(layer, j)
            if (i == 0 or (not display[layer + 1, j])) and polarity[layer, j] == 0:
                display[layer, j] = False

    has_key_points = np.zeros(contri.shape) > 0
    def is_legal(x): return not (x == '.' or x == ',' or x == '')
    n_points = np.sum([np.sum(display[:, i])
               for i in range(contri.shape[1] - 1) if is_legal(word[i])])
    n_edges = int(n_points) // 4

    swap_points = []
    cross_count = {}
    indegree_count = {}
    outdegree_count = {}
    maxdegree = config.flow_max_degree
    maxcross = config.flow_max_cross
    k = contri.shape[0] - 1
    has_key_points[k, 0] = True
    swap_points.append((int(contri[k - 1, 1:].argmax()) + 1, 0, k - 1, 1, 10))
    for _ in range(n_edges):
        max_delta = 0
        max_item = None
        for i in range(0, contri.shape[1]):
            start_point = (contri.shape[0] // 3) if i == 0 else 1
            for left in range(start_point, contri.shape[0]):
                if not display[left, i]:
                    break
                for right in range(left + 1, contri.shape[0]):
                    if has_key_points[right, i] or not display[right, i]:
                        break
                    delta = abs(contri[left, i] - contri[right, i])
                    if delta > max_delta:
                        max_delta = delta
                        max_item = (i, left, right)
        if max_item == None:
            break
        max_delta = -1
        i, left, right = max_item
        for k in range(left + 1, right + 1):
            delta = abs(contri[k, i] - contri[k - 1, i])
            if polarity[k, i] * polarity[k - 1, i] == -1:
                delta *= 2
            if delta > max_delta:
                max_delta = delta
                max_item = k
        k = max_item
        has_key_points[k, i] = True
        added_j = set()
        default_max = 0.05
        while True:
            max_delta = default_max
            if contri[k, i] < contri[k - 1, i] and polarity[k, i] * polarity[k - 1, i] == 1 and i != 0:
                for j in range(1, contri.shape[1]):
                    if not is_legal(word[j]) or abs(i - j) <= 1 or not display[k, j]:
                        continue
                    if labels[k - 1][j] == labels[k - 1][i]:
                        continue
                    if polarity[k, i] == 0 and polarity[k - 1, i] == 0:
                        continue
                    if polarity[k - 1, j] == polarity[k, j] and polarity[k - 1, i] == polarity[k, i] and polarity[k, j] != polarity[k, i]:
                        continue
                    if j in added_j:
                        continue
                    delta = norm_contri[k - 1, j, i]
                    if delta > max_delta:
                        max_delta = delta
                        max_item = j
                if max_delta == default_max:
                    break
                # k += 1
                j = max_item
                (i, j) = (j, i)
                max_dist_delta = max_delta
            else:
                for j in range(1, contri.shape[1]):
                    if not is_legal(word[j]) or abs(i - j) <= 1 or not display[k - 1, j]:
                        continue
                    if labels[k - 1][j] == labels[k - 1][i]:
                        continue
                    if polarity[k - 1, j] == 0 and polarity[k, j] == 0:
                        continue
                    if polarity[k - 1, j] == polarity[k, j] and polarity[k - 1, i] == polarity[k, i] and polarity[k, j] != polarity[k, i]:
                        continue
                    if j in added_j:
                        continue
                    delta = norm_contri[k - 1, i, j]
                    if delta > max_delta:
                        max_delta = delta
                        max_item = j
                if max_delta == default_max:
                    break
                j = max_item
                max_dist_delta = max_delta
            if j >= len(word) or i >= len(word) or j < 0 or i < 0:
                break
            added_j.add(j)
            if cross_count.get((j, i), 0) < maxcross and outdegree_count.get((k - 1, j), 0) < maxdegree and indegree_count.get((k, i), 0) < maxdegree:
                cross_count[(j, i)] = cross_count.get((j, i), 0) + 1
                outdegree_count[(k - 1, j)
                                ] = outdegree_count.get((k - 1, j), 0) + 1
                indegree_count[(k, i)] = indegree_count.get((k, i), 0) + 1
                swap_points.append((j, i, k - 1, max_dist_delta, contri[k - 1, j] * contri[k, i]))
            if len(added_j) > 2:
                break
    return swap_points, display


def get_network_layout(loader, idx):
    idxs = get_word_preclustering(loader, idx)
    cache_dir = os.path.join('cache', loader.dataset_name)
    cache_file = os.path.join(cache_dir, f'{idx}.json')
    cache_data = open(cache_file, 'r').read()
    cache_data = json.loads(cache_data)
    contri = np.array(loader.contri[idx])[:loader.n_layer, :-1]
    contri = aggregate(contri, idxs)
    delta_s = np.array(loader.all_delta_s[idx])
    delta_s = aggregate(delta_s, idxs)
    for i in range(delta_s.shape[0]):
        for j in range(delta_s.shape[1]):
            if delta_s[i, j] < -loader.threshold_xi:
                delta_s[i, j] = -1
            elif delta_s[i, j] > +loader.threshold_xi:
                delta_s[i, j] = 1
            else:
                delta_s[i, j] = 0
    polarity = delta_s
    
    m = contri.shape[0]
    max_pos = contri[:, 0].argmax()
    contri[max_pos, 0] = contri.max()
    for i in range(max_pos, m):
        contri[i, 0] = contri[max_pos, 0]
    for i in range(max_pos - 1, -1, -1):
        if contri[i, 0] > contri[i + 1, 0]:
            contri[i, 0] = contri[i + 1, 0]
    for j in range(1, contri.shape[1]):
        max_pos = contri[:, j].argmax()
        for i in range(1, max_pos):
            if contri[i, j] < contri[i - 1, j]:
                contri[i, j] = contri[i - 1, j]
    polarity[:, 0] = -1 if loader.pred_label[idx] == loader.data_labels[loader.main_index] else 1
    # contri = contri * polarity
    word = []
    word_len = []
    for t in idxs:
        word.append(' '.join([loader.data_word[idx][i]
                    for i in t if loader.data_word[idx][i] != '']))
        l = 0
        for i in t:
            l += loader.data_word_lens[idx][i] + 1
        word_len.append(l - 1)
    weights = []
    for layer in range(loader.n_layer):
        mat = loader.layer_weight[idx][layer][:-1, :-1]
        w = get_weight_matrix(mat, normalize=True).transpose()
        weights.append(w)
    weights = np.array(weights)
    weights = aggregate(weights, idxs, 1)
    weights = aggregate(weights, idxs, 2)

    labels = get_layer_clustering(loader, word, weights)

    dist = []
    n = len(word)
    grid_n = max(config.flow_expected_min_grids, n * config.flow_word_per_grids)
    if grid_n > config.flow_expected_max_grids:
        grid_n = n * (config.flow_word_per_grids - 1)
    grid_step = 1.0 / grid_n
    for layer in range(loader.n_layer):
        w = weights[layer]
        pairwise_dist = []
        for i in range(n):
            d = []
            for j in range(n):
                t = w[i] - w[j]
                t = t.dot(t)
                d.append(t)
            pairwise_dist.append(d)
        pairwise_dist = np.array(pairwise_dist)
        if len(dist) > 0:
            pairwise_dist = np.minimum(pairwise_dist, dist[-1])
        dist.append(pairwise_dist)
    for layer in range(loader.n_layer):
        diag_len = 0
        for i in range(2, n):
            diag_len += dist[layer][i - 1, i]
        dist[layer] /= diag_len
        if layer > 0:
            ratio = 1.0
            for i in range(1, n):
                t = dist[layer][i - 1, i] / dist[layer - 1][i - 1, i]
                if t > ratio:
                    ratio = t
            if ratio > 1.0:
                dist[layer] /= ratio

    y = np.zeros((n, loader.n_layer + 1))
    for i in range(n):
        t = int((1 - i / n) / grid_step + 1e-5)
        y[i, 0] = t * grid_step
    layer_diag_len = []
    for layer in range(loader.n_layer):
        left = 0
        total = 0
        last = 2
        for i in range(2, n):
            delta = dist[layer][i - 1, i]
            if labels[layer][i] == labels[layer][i - 1]:
                left += delta * (1 - config.information_flow_alpha * 0.5)
                dist[layer][i - 1, i] = delta * config.information_flow_alpha * 0.5
            elif layer + 1 < loader.n_layer and labels[layer + 1][i] == labels[layer + 1][i - 1]:
                left += delta * (1 - config.information_flow_alpha)
                dist[layer][i - 1, i] = delta * config.information_flow_alpha
            else:
                dist[layer][last - 1, last] += left / 2
                dist[layer][i - 1, i] += left / 2
                left = 0
                last = i
                total += 1
        delta = left / total
        for i in range(1, n):
            if labels[layer][i] == labels[layer][i - 1]:
                pass
            elif layer + 1 < loader.n_layer and labels[layer + 1][i] == labels[layer + 1][i - 1]:
                pass
            elif layer + 2 < loader.n_layer and labels[layer + 2][i] == labels[layer + 2][i - 1]:
                pass
            else:
                dist[layer][i - 1, i] += delta
    for layer in range(loader.n_layer):
        left = 0
        if layer > 0:
            for i in range(1, n):
                dist[layer][i - 1, i] = (dist[layer][i - 1, i] + dist[layer - 1][i - 1, i]) / 2
        diag_len = 0
        for i in range(1, n):
            diag_len += dist[layer][i - 1, i]
        layer_diag_len.append(diag_len)

    #print(len(word), n, word)
    for layer in range(loader.n_layer):
        j = layer + 1
        f = np.ones((n, grid_n)) * max_int
        pre = []
        for i in range(n):
            pre.append([0] * grid_n)

        for k in range(grid_n):
            t = k * grid_step
            f[0, k] = max_int
        if loader.n_layer > config.flow_deep_layer:
            K = int((config.information_flow_alpha * max(0, layer - config.flow_deep_layer) / (loader.n_layer - config.flow_deep_layer)) / 2 * grid_n)
        else:
            K = int((config.information_flow_alpha * max(0, layer) / (loader.n_layer)) / 2 * grid_n)
        k = grid_n - 1 - K
        for k2 in range(0, k):
            t = k2 * grid_step
            f[0, k2] = (t - y[0, 0]) ** 2
        for i in range(n):
            for k in range(grid_n):
                t = k * grid_step
                ft = (t - y[i, j - 1]) ** 2 * config.information_flow_alpha
                if ft == 0:
                    ft = -config.information_flow_beta * grid_step ** 2
                delta = 0 if i == 0 else dist[layer][i - 1, i]
                ed = grid_n
                if i > 0 and labels[layer][i - 1] == labels[layer][i]:
                    st = k + 1
                    ed = min(grid_n, k + 4)
                    ft = 0
                elif i > 0 and layer + 1 < loader.n_layer and labels[layer + 1][i - 1] == labels[layer + 1][i]:
                    st = k + 1
                    ed = min(grid_n, k + 5)
                    ft = 0
                else:
                    st = min(grid_n, k + 3)
                for l in range(st, ed):
                    d = (l - k) * grid_step
                    d2 = f[i - 1, l] + ft + (abs(d - delta) ** 2)
                    if d2 < f[i, k]:
                        f[i, k] = d2
                        pre[i][k] = l
        k = max(K // 4, int(y[n - 1, j - 1] // grid_step))
        for i in range(k + 1, K + 10):
            if f[n - 1][i] < f[n - 1][k]:
                k = i
        #k = K
        cy = []
        for i in range(n - 1, -1, -1):
            cy.append(k)
            k = pre[i][k]
        cy = cy[::-1]
        cy = np.array(cy) * grid_step
        for i in range(n):
            y[i, j] = cy[i]
    for layer in range(1, loader.n_layer - 1):
        for i in range(n):
            if y[i, layer] < y[i, layer - 1] and y[i, layer] < y[i, layer + 1]:
                y[i, layer] = min(y[i, layer - 1], y[i, layer + 1])
            elif y[i, layer] > y[i, layer - 1] and y[i, layer] > y[i, layer + 1]:
                y[i, layer] = max(y[i, layer - 1], y[i, layer + 1])

    lines = []
    norm_weight = loader.layer_entropy[idx]
    norm_weight = aggregate(norm_weight, idxs)
    norm_weight = get_weight_matrix(norm_weight).transpose()
    relations, is_displayed = get_non_adjacent_relations(word, weights, contri, polarity, labels)

    word[0] = 'CLS'
    word_len[0] = 3
    phrases = []
    for i in range(n):
        print(word[i], contri[:, i], polarity[:, i])
    contri = contri * polarity
    show_label_set = set()
    for i in range(y.shape[0]):
        t = []
        for j in range(y.shape[1] - 1):
            weight = norm_weight[j, i]
            curr_contri = contri[j, i]
            if j > 0:
                layer = j - 1
                k = i + 1
                if layer > 0 and k <= n:
                    if labels[layer][k] == labels[layer][i]:
                        if labels[layer - 1][k] != labels[layer - 1][i]:
                            st = i
                            ed = i + 1
                            while st > 0 and labels[layer][st - 1] == labels[layer][st]:
                                st -= 1
                            while ed < n and labels[layer][ed + 1] == labels[layer][ed]:
                                ed += 1
                            #print(st, ed, loader.data_word[idx][st + 1: ed + 1])
                            idxs = range(st, ed)
                            idxs = [i for i in idxs if word[i] != '']
                            if len(idxs) > 3:
                                idxs = sorted(idxs, key=lambda x: -norm_weight[-1][x])
                                idxs = idxs[:3]
                                idxs = sorted(idxs)
                            for l in range(st, ed):
                                show_label_set.add((j, l))
                                print('label', j, word[l])
                            phrases.append({
                                'top': st, 'bottom': ed - 1,
                                'y1': float(y[st, j]),
                                'y2': float(y[ed - 1, j]),
                                'left': float(y[ed - 1, j - 1]),
                                'layer': j,
                                'type': 'phrase',
                                'text': ' '.join([word[i] for i in idxs]),
                            })
            t.append({
                'layer': j,
                'display': True if is_displayed[j, i] != 0 else False,
                'position': round(float(y[i, j]), 4),
                'show_label': True if (j, i) in show_label_set else False,
                'size': round(float(weight), 4),
                'label': labels[j - 1][i] if j > 0 else i,
                'contri': round(float(curr_contri), 4),
            })
        lines.append({'text': word[i],  'len': word_len[i], 'line': t})
    cnt = 0

    for p in relations:
        top, bottom, j, _, weight = p
        if j + 1 >= y.shape[1]:
            continue
        if top >= n or bottom >= n:
            continue
        y1 = float(y[top, j])
        y3 = float(y[top, j - 1])
        y2 = float(y[bottom, j + 1])
        y4 = float(y[bottom, j])
        y1 = (y1 + y3) / 2
        y2 = (y2 + y4) / 2
        text = word[top] + ' ' + word[bottom]
        cnt += 1
        phrases.append({
            'top': top,
            'bottom': bottom,
            'weight': float(weight),
            'y1': y1,
            'y2': y2,
            'layer': int(j + 1),
            'type': 'relation',
            'text': text,
        })
    ret = {'lines': lines, 'phrases': phrases}
    return ret
