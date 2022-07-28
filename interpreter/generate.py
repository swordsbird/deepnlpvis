from multiprocessing import Pool, Process, set_start_method
import time
import sys
import json
import torch
import os
import random
import pandas as pd
import numpy as np
from layer_parser import parser_text
from model_helper import convert_state_dict
from pytorch_pretrained_bert import BertModel, BertTokenizer
from model_basic import basic_RNNClassifier, basic_CNNClassifier, global_tokenizer
import yaml
from basic_model_helper import load_basic_model, preprocess_sentence_batch, set_device

if __name__ != "__main__":
    exit()

cfg = open('config.json', 'r')
cfg = cfg.read()
cfg = json.loads(cfg)

has_regular = False
if cfg['type'] == 'bert':
    model = BertModel.from_pretrained('bert-base-uncased')
    state_dict = torch.load(cfg['dataset'], map_location='cpu')
    state_dict, (weight, bias) = convert_state_dict(state_dict)
    model.load_state_dict(state_dict)
    weight = weight.transpose(0,1)
else:
    model = basic_RNNClassifier(yaml.load(open(os.path.join('data', 'LSTM_basic_4_layer_768_hidden_pool_max.yaml'),'r')))
    model.load_state_dict(torch.load(os.path.join('data', 'lstm_basic_SST2_5_300_0.8486.param'), map_location='cpu'))
    has_regular = True
    LSTM_regular = json.load(open('data/lstm_sst.json'))
    regularizations = LSTM_regular['-1']
    regularizations = [(int(x), regularizations[x]) for x in regularizations]
    regularizations = sorted(regularizations, key = lambda x: x[0])
    regularizations = [x[1] for x in regularizations]
    weight = None
    bias = None

table = pd.read_csv(cfg['datapath'], sep='\t')
if cfg['datatype'] == 'sst2':
    text = [x for x in table['sentence']]
elif cfg['datatype'] == 'agnews':
    title = [x for x in table['Title']]
    description = [x for x in table['Description']]
    text = [title[i] + description[i] for i in range(table.shape[0])]

if cfg["prediction"] and cfg["type"] == "rnn":
    texts = [x for x in table['sentence']]
    labels = [x for x in table['label']]
    n = len(text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    for i in range(n):
        text = texts[i]
        label = labels[i]
        x, Phi, tokenized = preprocess_sentence_batch(model=model, types='rnn', sent1=text, target_id = -1, target_layer=6, device=device)
        logits = Phi.get_hidden(x).cpu().numpy()
        out_label = int(logits.argmax())
        if out_label == label:
            continue
            print(i, 'pass')
        else:
            print(i, text, out_label, label)
    exit()

n_layer = cfg['n_layer']
n_neuron  = cfg['n_neuron']

if not has_regular:
    if not os.path.exists('data/regular.npy'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        def get_embedding(x):
            global model
            x = x.unsqueeze(0)
            attention_mask = torch.ones(x.shape[:2]).to(x.device)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(
                dtype=torch.float)
            extended_attention_mask = (
                1.0 - extended_attention_mask) * -10000.0
            # extract the 3rd layer
            model_list = model.encoder.layer
            hidden_states = x
            ret = np.array(x.cpu())
            for layer_module in model_list:
                hidden_states = layer_module(
                    hidden_states, extended_attention_mask)
                x2 = np.array(hidden_states.cpu())
                ret = np.concatenate([ret, x2])
            return ret

        def get_embedding_from_text(text):
            words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
            tokenized_ids = tokenizer.convert_tokens_to_ids(words)
            segment_ids = [0 for _ in range(len(words))]
            token_tensor = torch.tensor([tokenized_ids], device=device)
            segment_tensor = torch.tensor([segment_ids], device=device)
            x = model.embeddings(token_tensor, segment_tensor)[0]
            return get_embedding(x)

        sampled_text = random.sample(text, 1000)
        sampled_s = [[] for i in range(n_layer + 1)]

        it = 0
        for t in sampled_text:
            if it % 100 == 0:
                print(f'{it} iters')
            it += 1
            s0 = get_embedding_from_text(t)
            for k in range(len(s0)):
                s = s0[k]
                s = s.mean(axis=0)
                sampled_s[k].append(s.tolist())
        for k in range(len(sampled_s)):
            sampled_s[k] = np.array(sampled_s[k])
            sampled_s[k] = np.std(sampled_s[k], axis=0)

        sampled_s = np.array(sampled_s)
        np.save('data/regular.npy', sampled_s)

        exit()

    regularizations = np.load('data/regular.npy')
text = [(text[k], k) for k in range(len(text))]
if cfg['test']:
    text = [text[i] for i in cfg['ids']]
else:
    text = text[:cfg['n']]
#random.shuffle(text)

neurons = []
for i in range(n_layer):
    clusters = [range(n_neuron)]
    neurons.append(clusters)

modes = [cfg['mode']]#[['contri']]#[['weight', 'word', 'contri']]
task_per_gpu = cfg['task_per_gpu']
gpus = cfg['gpus'].split(',')
n = len(gpus)
m = n * task_per_gpu
task_texts = [[] for i in range(m)]
for i in range(len(text)):
    k = i % m
    task_texts[k].append(text[i])

if not os.path.exists('output'):
    os.mkdir('output')

for mode in modes:
    for x in mode:
        if not os.path.exists('output/' + x):
            os.mkdir('output/' + x)
    tasks = []
    for k in range(n):
        for i in range(task_per_gpu):
            j = k * task_per_gpu + i
            p = Process(target=parser_text, args=(
                task_texts[j], gpus[k], n_layer, regularizations, model, (weight, bias), mode))
            tasks.append(p)
            p.start()

    for p in tasks:
        p.join()
