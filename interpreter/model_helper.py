import torch
import os
from pytorch_pretrained_bert import BertModel

home_path = '/home/lizhen/data'
path_to_cola = os.path.join(home_path, 'model/cola')
path_to_sst2 = os.path.join(home_path, 'model/sst2')

def convert_state_dict(state_dict):
    old_keys = []
    new_keys = []
    for key in list(state_dict.keys()):
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if 'bert.' in key:
            if new_key == None:
                new_key = key.replace('bert.', '')
            else:
                new_key = new_key.replace('bert.', '')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)
    weight = state_dict['classifier.weight']
    bias = state_dict['classifier.bias']
    state_dict.pop('classifier.weight')
    state_dict.pop('classifier.bias')
    return state_dict, (weight, bias)

def load_cola_model(number):
    bert = BertModel.from_pretrained('bert-base-uncased')
    if number == 0:
        state_dict = bert.state_dict()
    elif number == 1:
        state_dict = torch.load(os.path.join(path_to_cola, 'bert_CoLA_0_0_0.3116.param'), map_location='cpu')
    elif number == 2:
        state_dict = torch.load(os.path.join(path_to_cola, 'bert_CoLA_0_100_0.7814.param'), map_location='cpu')
    elif number == 3:
        state_dict = torch.load(os.path.join(path_to_cola, 'bert_CoLA_0_200_0.7967.param'), map_location='cpu')
    elif number == 4:
        state_dict = torch.load(os.path.join(path_to_cola, 'bert_CoLA_1_0_0.8102.param'), map_location='cpu')
    elif number == 5:
        state_dict = torch.load(os.path.join(path_to_cola, 'bert_CoLA_1_100_0.8226.param'), map_location='cpu')
    elif number == 6:
        state_dict = torch.load(os.path.join(path_to_cola, 'bert_CoLA_2_0_0.8303.param'), map_location='cpu')
    elif number == 7:
        state_dict = torch.load(os.path.join(path_to_cola, 'bert_CoLA_5_0_0.8399.param'), map_location='cpu')
    elif number == -1 or number == 8:
        state_dict = torch.load(os.path.join(path_to_cola, 'bert_CoLA_8_100_0.8408.param'), map_location='cpu')
    state_dict = convert_state_dict(state_dict)
    bert.load_state_dict(state_dict)
    return bert

def load_sst2_model(number):
    bert = BertModel.from_pretrained('bert-base-uncased')
    if number == 0:
        state_dict = bert.state_dict()
    elif number == 1:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_SST2_3_1300_0.9369.param'),map_location='cpu')
    elif number == 2:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_sst2t_0_200_0.8761.param'),map_location='cpu')
    elif number == 3:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_sst2t_0_300_0.8968.param'),map_location='cpu')
    elif number == 4:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_sst2t_0_400_0.8979.param'),map_location='cpu')
    elif number == 5:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_sst2t_0_600_0.9060.param'),map_location='cpu')
    elif number == 6:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_sst2t_0_800_0.9140.param'),map_location='cpu')
    elif number == 7:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_sst2t_0_1400_0.9151.param'),map_location='cpu')
    elif number == 8:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_sst2t_0_1600_0.9197.param'),map_location='cpu')
    elif number == 9:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_sst2t_0_1900_0.9209.param'),map_location='cpu')
    elif number == 10:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_sst2t_1_500_0.9289.param'),map_location='cpu')
    elif number == 11:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_sst2t_1_1700_0.9312.param'),map_location='cpu')
    elif number == -1 or number == 12:
        state_dict = torch.load(os.path.join(path_to_sst2, 'bert_sst2t_5_100_0.9323.param'),map_location='cpu')
    state_dict = convert_state_dict(state_dict)
    bert.load_state_dict(state_dict)
    return bert
