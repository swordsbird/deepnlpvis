import numpy as np
import json
import random

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from flask_compress import Compress
from pytorch_pretrained_bert import BertTokenizer
from utils import entropy_to_contribution, word_ordering, stem
from loader import init_loader, update_select_index
from flow_layout import get_network_layout
from keyword_extractor import get_sentences_keywords
from context_layout import get_wordcontext_layout
from config import n_percentile, n_percentile_words, word_contribution_max_layers

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
get_grids_ret = None
loader = init_loader()
update_select_index(loader, [1, 0])

sentence_network_cache = {}
word_network_cache = {}


def get_word_info(loader, word):
    info = np.zeros(loader.n_layer)
    for p in loader.inverted_list[word]:
        info = info + loader.layer_entropy[p[0]][-loader.n_layer:, p[1]]
    info = info / len(loader.inverted_list[word])
    return info


def get_word_contribution(loader, idxs, attrs={}):
    selected_labels = loader.selected_labels()
    idxs = [i for i in idxs if loader.data_label[i] in selected_labels and loader.pred_label[i] in selected_labels]
    loader.cache['linechart_idxes'] = idxs
    loader.cache['linechart_attrs'] = attrs
    represents = {}
    wordkey = {}
    norm_contri = {}
    frequency = {}
    last_layer = loader.n_layer - 1
    total_words = 0

    polarity_by_layer = []
    for layer in range(loader.n_layer):
        polarity = loader.get_word_delta_s_by_layer(idxs, layer)
        polarity_by_layer.append(polarity)

    for i in idxs:
        sent = loader.word_info_by_layer[last_layer][i]
        instance_weight = 1
        if loader.is_correct(i) and not loader.is_overview(idxs):
            instance_weight = 5
        for p in sent:
            key, word, tf, importance = p
            if key not in represents:
                represents[key] = {}
            represents[key][word] = represents[key].get(
                word, 0) + instance_weight
            frequency[key] = frequency.get(key, 0) + instance_weight
            norm_contri[key] = norm_contri.get(
                key, 0) + importance * instance_weight
            if word not in loader.stop_words:
                total_words += 1

    for key in norm_contri:
        norm_contri[key] = norm_contri[key] / frequency[key]
    words = [(key, norm_contri[key]) for key in norm_contri]
    words = sorted(words, key=lambda x: -x[1])

    tmp = []
    tmp2 = []
    for key in represents:
        max_count = 0
        represent_word = ''
        for word in represents[key]:
            if represents[key][word] > max_count:
                max_count = represents[key][word]
                represent_word = word
        tmp.append((key, represent_word))
        tmp2.append((represent_word, key))
    represents = dict(tmp)
    wordkey = dict(tmp2)

    layers = []
    last_value = 0
    norm_contri = {}
    previous_contri = {}
    layer_words = []
    layers.append({
        'words': None,
        'info': 0,
        'index': 0,
    })

    word_ranks = {}
    last_value = 1
    for layer in range(0, loader.n_layer):
        norm_contri = {}
        all_entropy = []
        all_contri = []
        contri = {}
        curr_words = 0
        v_distribution = []
        for i in idxs:
            instance_weight = 1
            if loader.is_correct(i) and not loader.is_overview(idxs):
                instance_weight = 5
            for index, p in enumerate(loader.word_info_by_layer[layer][i]):
                key, word, contri0, norm_contri0 = p
                if key not in represents:
                    represents[key] = {}
                norm_contri[key] = norm_contri.get(
                    key, 0) + norm_contri0 * instance_weight
                contri[key] = contri.get(key, 0) + contri0 * instance_weight
                if word not in loader.stop_words:
                    all_entropy.append(contri0)
                    all_contri.append(loader.delta_s_value[i][layer, index])
                    v_distribution.append(contri0)
                    if contri0 > loader.threshold_gamma:
                        curr_words += 1
        for key in norm_contri:
            norm_contri[key] = norm_contri[key] / frequency[key]
            contri[key] = contri[key] / frequency[key]
        layer_value = float(curr_words / total_words)
        if last_value < layer_value:
            layer_value = last_value
        last_value = layer_value
        all_entropy = sorted(all_entropy)
        all_contri = sorted(all_contri)

        n_ticks = 100
        entropy_ticks = []
        contri_ticks = []
        for i in range(n_ticks):
            contri_ticks.append(all_contri[len(all_contri) * i // n_ticks])
        for i in range(n_ticks):
            entropy_ticks.append(all_entropy[len(all_entropy) * i // n_ticks])

        def get_yaxis(x):
            left = 0
            right = n_ticks - 1
            while left < right:
                mid = (left + right) // 2
                if entropy_ticks[mid] >= x:
                    right = mid
                else:
                    left = mid + 1
            return left
        for key in norm_contri:
            if key not in word_ranks:
                word_ranks[key] = []
            yaxis = get_yaxis(contri[key])
            word_ranks[key].append(yaxis)
        words = [(represents[key], norm_contri[key], contri[key], frequency[key])
                 for key in norm_contri]
        words = word_ordering(words)
        all_words = []
        yaxis_count = [0] * n_percentile
        max_yaxis_count = n_percentile_words
        for x in words:
            word, importance, tf, contri0, _ = x
            if word in loader.stop_words:
                continue
            yaxis = word_ranks[stem(word)][-1] / n_ticks
            t = int(yaxis * n_ticks / n_percentile)
            if yaxis_count[t] > max_yaxis_count:
                continue
            yaxis_count[t] += 1
            status = 'retained'
            if contri0 < loader.threshold_gamma:
                old_e = previous_contri.get(wordkey[word], 1e10)
                if old_e < loader.threshold_gamma:
                    status = 'old_discarded'
                else:
                    status = 'new_discarded'
            all_words.append({
                'word': word,
                'value': round(float(importance), 4),
                'frequency': round(float(tf), 4),
                'entropy': round(float(contri0), 4),
                'contri': polarity_by_layer[layer][word],
                'score': loader.word_prediction_score[word],
                'yaxis': yaxis,
                'status': status,
            })

        layer_words.append(all_words)
        layers.append({
            'word_range': [],
            'words': None,
            'info': 1.0 - layer_value,
            'index': layer + 1,
        })
        previous_contri = contri

    cloud_layers = []
    cloud_range = []
    if loader.n_layer <= word_contribution_max_layers:
        step = 1
        cloud_range = [(start, start + step)
                       for start in range(loader.n_layer)]
    else:
        step = loader.n_layer // word_contribution_max_layers
        cloud_range = [(start, start + step)
                       for start in range(0, loader.n_layer, step)]
    cloud_layers = [r[1] - 1 for r in cloud_range]
    for j, i in enumerate(cloud_layers):
        layers[i + 1]['words'] = layer_words[i]
        layers[i + 1]['word_range'] = cloud_range[j]
    last_discards = []
    last_words = {}
    all_words = set()
    for layer in layers:
        if layer['words'] != None:
            for x in layer['words']:
                all_words.add(x['word'])
                if x['word'] in last_words:
                    y = last_words[x['word']]
                    x['yaxis'] = (x['yaxis'] + y['yaxis']) / 2
                    x['entropy'] = (x['entropy'] + y['entropy']) / 2
                    x['status'] = y['status']
                    x['contri'] = y['contri']
                if x['word'] in last_discards:
                    x['status'] = 'old_discarded'
                elif x['status'] == 'old_discarded':
                    x['status'] = 'new_discarded'
                x['line'] = word_ranks[stem(x['word'])]
            last_discards = set(
                [x['word'] for x in layer['words'] if x['status'] != 'retained'])
            last_words = {}
        else:
            last_words = {}
    return layers


layer_info = get_word_contribution(loader, range(loader.size))

app = Flask(__name__, template_folder='static',
            static_folder='static', static_url_path='/static')
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
Compress(app)


@app.route('/api/word', methods=['POST'])
def get_word_idxs():
    data = json.loads(request.get_data(as_text=True))
    word = data['word']
    word = stem(word)
    if word in loader.inverted_list:
        idxs = [p[0] for p in loader.inverted_list[word]]
        pos = [p[1] for p in loader.inverted_list[word]]
        info = get_word_info(loader, word)
        info = [float(entropy_to_contribution(x)) for x in info]
    else:
        idxs = []
        pos = []
        info = []
    return jsonify({'idxs': idxs, 'pos': pos, 'info': info})


@app.route('/api/networks', methods=['POST'])
def get_network():
    data = json.loads(request.get_data(as_text=True))
    if data['level'] == 'sentence':
        idx = data['idx']
        layers = data['layers']
        if idx not in sentence_network_cache:
            layout = get_network_layout(loader, idx)
            loader.cache['linechart_attrs']['highlight'] = loader.data_word[idx]
            linechart = get_word_contribution(
                loader,
                loader.cache['linechart_idxes'],
                loader.cache['linechart_attrs']
            )
            ret = {'layout': layout, 'linechart': linechart}
            sentence_network_cache[idx] = ret
        else:
            ret = sentence_network_cache[idx]
        return jsonify(ret)
    elif data['level'] == 'word':
        word = data['word']
        layers = data['layers']
        layers = [x for x in layers]
        if word not in word_network_cache:
            neurons, edges = get_wordcontext_layout(loader, layers, word)
            word_network_cache[word] = [neurons, edges]
        else:
            neurons, edges = word_network_cache[word]
        ret = {'neuron_clusters': neurons, 'edges': edges}
        return jsonify(ret)
    else:
        ret = {'neuron_clusters': None, 'edges': None}
        return jsonify(ret)


@app.route('/api/scatterplot', methods=['GET', 'POST'])
def get_scatterplot():
    selected_labels = loader.selected_labels()
    idxs = [i for i in range(loader.size) if loader.data_label[i] in selected_labels and loader.pred_label[i] in selected_labels]
    keywords = get_sentences_keywords(loader, idxs, -1)
    scatters = []
    for i in idxs:
        x1 = loader.data_prediction_score[i][loader.main_index]
        x2 = loader.data_prediction_score[i][loader.second_index]
        scatters.append({
            'id': i,
            'x': round(x1 / (x1 + x2), 4),
            'y': float(loader.data_embedding[i]),
            'label': loader.data_label[i],
            'pred': loader.pred_label[i],
            'is_correct': loader.data_label[i] == loader.pred_label[i],
        })
    scatters = sorted(scatters, key=lambda x: x['y'])
    ys = [t['y'] for t in scatters]
    y1 = float(np.max(ys))
    y0 = float(np.min(ys))
    for t in scatters:
        t['y'] = round((t['y'] - y0) / (y1 - y0), 4)
    delta = 0
    max_gap = 0.03
    for i in range(1, len(scatters)):
        scatters[i]['y'] -= delta
        curr_d = scatters[i]['y'] - scatters[i - 1]['y']
        if curr_d > max_gap:
            delta += curr_d - max_gap
            scatters[i]['y'] -= curr_d - max_gap
    ret = {
        'keywords': keywords,
        'scatters': scatters,
        'dataset': loader.dataset_name,
        'n_layer': loader.n_layer
    }
    return jsonify(ret)


@app.route('/api/layers', methods=['POST'])
def get_layers():
    data = json.loads(request.get_data(as_text=True))
    idxs = data.get('idxs', None)
    attrs = data.get('attrs', {})
    if idxs == None or len(idxs) == 0:
        ret = get_word_contribution(loader, range(loader.size), attrs)
    else:
        ret = get_word_contribution(loader, idxs, attrs)
    return jsonify(ret)


@app.route('/api/sentences', methods=['POST'])
def get_sentences():
    data = json.loads(request.get_data(as_text=True))
    idxs = data.get('idxs', None)
    if idxs == None or len(idxs) == 0:
        idxs = range(loader.size)
    samples = []
    for i in idxs:
        samples.append({
            'index': i,
            'embedding': loader.data_embedding[i],
            'score': loader.data_prediction_score[i][loader.main_index],
            'label': loader.data_label[i],
            'pred': loader.pred_label[i],
            'is_correct': loader.data_label[i] == loader.pred_label[i],
        })
    samples = sorted(samples, key=lambda x: x['embedding'])
    ret = samples

    for x in ret:
        x['text'] = loader.data_text[x['index']]
        x['wrong'] = x['pred'] != x['label']
    random.shuffle(ret)
    ret = sorted(ret, key=lambda x: 0 if len(x['text']) >= 20 else 1)
    ret = {
        'sentences': ret,
    }
    return jsonify(ret)


@app.route('/api/word_sentences', methods=['POST'])
def get_word_sentences():
    data = json.loads(request.get_data(as_text=True))
    word = data.get('word', None)
    word = stem(word)
    if word == None or word not in loader.all_word_labels:
        ret = {'sentences': [], 'pos': 0, 'neg': 0, 'tot': 0}
    else:
        labels = loader.all_word_labels[word]
        idxs = [x[0] for x in loader.all_inverted_list[word]]
        ret = {
            'sentences': [{'text': loader.all_data_text[i], 'label': loader.all_data_label[i], 'index': i} for i in idxs],
            'pos': len([x for x in labels if x == loader.data_labels[0]]),
            'neg': len([x for x in labels if x != loader.data_labels[0]]),
        }
    return jsonify(ret)


@app.route('/api/confusion_matrix', methods=['POST'])
def get_confusion_matrix():
    ret = {
        'labels': loader.data_labels,
        'label_names': loader.data_label_name,
        'selection': [loader.main_index, loader.second_index],
        'matrix': loader.get_confusion_matrix(),
    }
    return jsonify(ret)


@app.route('/api/select_class', methods=['POST'])
def select_class():
    ret = {
        'selection': [loader.main_index, loader.second_index],
        'matrix': loader.get_confusion_matrix(),
    }
    return jsonify(ret)


@app.route('/api/all_sentences', methods=['POST'])
def get_all_sentences():
    data = json.loads(request.get_data(as_text=True))
    idxs = data.get('idxs', None)
    if idxs == None or len(idxs) == 0:
        idxs = range(loader.size)
    ret = []
    for i in idxs:
        ret.append({
            'index': i,
            'embedding': round(loader.data_embedding[i], 4),
            'score': [round(k, 4) for k in loader.data_prediction_score[i]],
            'pred': loader.pred_label[i],
            'label': loader.data_label[i],
        })
    for x in ret:
        x['text'] = loader.data_text[x['index']]
        x['wrong'] = x['label'] != x['pred']

    ret = {
        'sentences': ret,
    }
    return jsonify(ret)


@app.route('/')
def index():
    return render_template('index.html')
