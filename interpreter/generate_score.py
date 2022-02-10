import multiprocessing
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

if __name__ != "__main__":
    exit()

cfg = open('config.json', 'r')
cfg = cfg.read()
cfg = json.loads(cfg)

if cfg['type'] == 'bert':
    model = BertModel.from_pretrained('bert-base-uncased')
    state_dict = torch.load(cfg['dataset'], map_location='cpu')
    state_dict, (weight, bias) = convert_state_dict(state_dict)
    model.load_state_dict(state_dict)
    weight = weight.transpose(0,1)
else:
    pass

table = pd.read_csv(cfg['fulldatapath'], sep='\t')
if cfg['datatype'] == 'sst2':
    text = [x for x in table['sentence']]
elif cfg['datatype'] == 'agnews':
    title = [x for x in table['Title']]
    description = [x for x in table['Description']]
    text = [title[i] + description[i] for i in range(table.shape[0])]

n_layer = cfg['n_layer']
n_neuron  = cfg['n_neuron']

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
    for layer_module in model_list:
        hidden_states = layer_module(
            hidden_states, extended_attention_mask)
    return hidden_states[0]

def get_embedding_from_text(text):
    words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
    tokenized_ids = tokenizer.convert_tokens_to_ids(words)
    segment_ids = [0 for _ in range(len(words))]
    token_tensor = torch.tensor([tokenized_ids], device=device)
    segment_tensor = torch.tensor([segment_ids], device=device)
    x = model.embeddings(token_tensor, segment_tensor)[0]
    return get_embedding(x)

it = 0
weight = torch.tensor(weight, device = device)
bias = torch.tensor(bias, device = device)
logits = []

for t in text:
    if it % 100 == 0:
        print(f'{it} iters')
    it += 1
    x = get_embedding_from_text(t)
    x = x.unsqueeze(0)
    logit = torch.matmul(x[:,0,:], weight) + bias
    logit = torch.softmax(logit, dim=1)
    logit = logit.mean(axis=0)
    logits.append(logit)
logits = np.array(logits)

np.save('output/all_labels.npy', logits)

