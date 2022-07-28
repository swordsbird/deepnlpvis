from .model_elmo import elmo_RNNClassifier, basic_CNNClassifier, global_tokenizer
from pytorch_pretrained_bert import BertTokenizer
from .dataset import InputExample, dataset
import torch
import yaml
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_device(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cuda_ = "cuda:" + str(gpu_id)
    global device
    device = torch.device(cuda_ if torch.cuda.is_available() else "cpu")

def load_basic_model(type='lstm', task='sst2', number = -1):
    """
    This function is used to load all the trained model

    @params:
    - type: Str, can be choosen from ['lstm','gru','cnn']
    - att: Bool, indicating whether the model use attention
    - task: Str, current tasks.
    """
    if task == 'sst2':
        if type == 'lstm':
            path_att_lstm = './model/lstm/sst2/'
            model = elmo_RNNClassifier(yaml.load(open(os.path.join(path_att_lstm, 'LSTM_basic_4_layer_768_hidden_pool_max.yaml'),'r')))
            # 0, 100, 200, 500, 700, 1000, 1400, 3300, 5800
            if number == 0:
                model.load_state_dict(torch.load(os.path.join(path_att_lstm, 'lstm_basic_SST2_0_0_0.5092.param'), map_location='cpu'))
            elif number == 1:
                model.load_state_dict(torch.load(os.path.join(path_att_lstm, 'lstm_basic_SST2_0_100_0.6514.param'), map_location='cpu'))
            elif number == 2:
                model.load_state_dict(torch.load(os.path.join(path_att_lstm, 'lstm_basic_SST2_0_200_0.7225.param'), map_location='cpu'))
            elif number == 3:
                model.load_state_dict(torch.load(os.path.join(path_att_lstm, 'lstm_basic_SST2_0_500_0.7718.param'), map_location='cpu'))
            elif number == 4:
                model.load_state_dict(torch.load(os.path.join(path_att_lstm, 'lstm_basic_SST2_0_700_0.7901.param'), map_location='cpu'))
            elif number == 5:
                model.load_state_dict(torch.load(os.path.join(path_att_lstm, 'lstm_basic_SST2_0_1000_0.8108.param'), map_location='cpu'))
            elif number == 6:
                model.load_state_dict(torch.load(os.path.join(path_att_lstm, 'lstm_basic_SST2_1_300_0.8326.param'), map_location='cpu'))
            elif number == 7:
                model.load_state_dict(torch.load(os.path.join(path_att_lstm, 'lstm_basic_SST2_3_0_0.8429.param'), map_location='cpu'))
            elif number == 8 or number == -1:
                model.load_state_dict(torch.load(os.path.join(path_att_lstm, 'lstm_basic_SST2_5_300_0.8486.param'), map_location='cpu'))
            else:
                raise ValueError("currently not available")
        elif type == 'cnn':
            path_cnn = './model/cnn/sst2/'
            model = basic_CNNClassifier(yaml.load(open(os.path.join(path_cnn,'CNN_new.yaml'),'r')))
            if number == 0:
                model.load_state_dict(torch.load(os.path.join(path_cnn,'cnn_basic_SST2_0_0_0.4920.param'), map_location='cpu'))
            elif number == 1:
                model.load_state_dict(torch.load(os.path.join(path_cnn,'cnn_basic_SST2_0_100_0.6261.param'), map_location='cpu'))
            elif number == 2:
                model.load_state_dict(torch.load(os.path.join(path_cnn,'cnn_basic_SST2_0_400_0.7190.param'), map_location='cpu'))
            elif number == 3:
                model.load_state_dict(torch.load(os.path.join(path_cnn,'cnn_basic_SST2_1_0_0.7603.param'), map_location='cpu'))
            elif number == 4:
                model.load_state_dict(torch.load(os.path.join(path_cnn,'cnn_basic_SST2_1_500_0.7764.param'), map_location='cpu'))
            elif number == 5:
                model.load_state_dict(torch.load(os.path.join(path_cnn,'cnn_basic_SST2_2_200_0.7936.param'), map_location='cpu'))
            elif number == 6:
                model.load_state_dict(torch.load(os.path.join(path_cnn,'cnn_basic_SST2_3_700_0.8131.param'), map_location='cpu'))
            elif number == -1 or number == 7:
                model.load_state_dict(torch.load(os.path.join(path_cnn,'cnn_basic_SST2_6_600_0.8200.param'), map_location='cpu'))

        else:
            raise ValueError("currently not available")
    elif task == 'cola':
        if type == 'lstm':
            path_lstm = './model/lstm/cola'
            model = elmo_RNNClassifier(yaml.load(open(os.path.join(path_lstm, 'LSTM_basic_4_layer_768_hidden_pool_max.yaml'),'r')))
            # 0, 850, 1000, 1150, 1450, 1750, 1950
            if number == 0:
                model.load_state_dict(torch.load(os.path.join(path_lstm, 'lstm_basic_max_cola_0_0_0.0000.param'), map_location='cpu'))
            elif number == 1:
                model.load_state_dict(torch.load(os.path.join(path_lstm, 'lstm_basic_max_cola_5_100_0.0276.param'), map_location='cpu'))
            elif number == 2:
                model.load_state_dict(torch.load(os.path.join(path_lstm, 'lstm_basic_max_cola_6_100_0.0478.param'), map_location='cpu'))
            elif number == 3:
                model.load_state_dict(torch.load(os.path.join(path_lstm, 'lstm_basic_max_cola_7_100_0.0666.param'), map_location='cpu'))
            elif number == 4:
                model.load_state_dict(torch.load(os.path.join(path_lstm, 'lstm_basic_max_cola_9_100_0.0819.param'), map_location='cpu'))
            elif number == 5:
                model.load_state_dict(torch.load(os.path.join(path_lstm, 'lstm_basic_max_cola_11_100_0.1054.param'), map_location='cpu'))
            elif number == 6 or number == -1:
                model.load_state_dict(torch.load(os.path.join(path_lstm, 'lstm_basic_max_cola_13_0_0.1296.param'), map_location='cpu'))
            else:
                raise ValueError("currently not available")
        elif type == 'cnn':
            path_cnn = './model/cnn/cola'
            model = basic_CNNClassifier(yaml.load(open(os.path.join(path_cnn,'CNN_new.yaml'),'r')))
            # 0, 600, 750, 3450, 3750, 4350, 4650, 39150
            if number == 0:
                model.load_state_dict(torch.load(os.path.join(path_cnn, 'cnn_basic_cola_0_0_0.0000.param'), map_location='cpu'))
            elif number == 1:
                model.load_state_dict(torch.load(os.path.join(path_cnn, 'cnn_basic_cola_4_0_0.0464.param'), map_location='cpu'))
            elif number == 2:
                model.load_state_dict(torch.load(os.path.join(path_cnn, 'cnn_basic_cola_5_0_0.0656.param'), map_location='cpu'))
            elif number == 3:
                model.load_state_dict(torch.load(os.path.join(path_cnn, 'cnn_basic_cola_23_0_0.0774.param'), map_location='cpu'))
            elif number == 4:
                model.load_state_dict(torch.load(os.path.join(path_cnn, 'cnn_basic_cola_25_0_0.0882.param'), map_location='cpu'))
            elif number == 5:
                model.load_state_dict(torch.load(os.path.join(path_cnn, 'cnn_basic_cola_29_0_0.1051.param'), map_location='cpu'))
            elif number == 6:
                model.load_state_dict(torch.load(os.path.join(path_cnn, 'cnn_basic_cola_31_0_0.1124.param'), map_location='cpu'))
            elif number == 7 or number == -1:
                model.load_state_dict(torch.load(os.path.join(path_cnn, 'cnn_basic_cola_261_0_0.1146.param'), map_location='cpu'))
            elif number == -9:
                model.load_state_dict(torch.load(os.path.join(path_cnn, 'cnn_basic_cola_9_0_0.0985.param'), map_location='cpu'))
            else:
                raise ValueError("currently not available")
    else:
        raise ValueError("currently not available")
    if USE_CUDA:
        model.to(device)
    for par in model.parameters():
        par.requires_grad = False
    model.eval()
    return model

def get_basic_embedding(model, types, Dataset: dataset, batchsize=-1, use_cuda = 'auto'):
    """
        get the embedding of RNNClassifier, CNNClassifier

        @params:
        - model: RNNClassifier/CNNClassifier, the loaded model
        - types: Str, the model type, can be choosen in ['rnn','cnn']
        - Dataset: Dataset, the dataset we need to get embedding
        - word_embedding_flag: Bool, whether to return word embedding or layer embedding. Set to True if you want the actual word embedding.
        - batchsize [optional]: Int, if the length of instance list is too large, we will batch them to get result. Set to -1 to ignore this option.
        - use_cuda: Bool/Str, whether to use cuda. Set to 'auto' for auto decision

        @return:
        a Torch.FloatTensor of shape (Batch, SeqLen, EmbDim)
    """
    if use_cuda == 'auto':
        use_cuda = torch.cuda.is_available()
    # first, we need to proceed the input example
    if types == 'rnn':
        model: elmo_RNNClassifier
    elif types == 'cnn':
        model: basic_CNNClassifier
    embed_layer = model.encoder.embed_layer
    embed_layer.to(device)

    if batchsize < 0: batchsize = len(Dataset.datasetList)

    all_embedding = torch.tensor(())
    if use_cuda:
        all_embedding = all_embedding.to(device)

    for data in Dataset.make_input(batchsize, False):
        sent_tensor = torch.tensor(data['sent'])
        mask_tensor = torch.tensor(data['mask'])
        if use_cuda:
            sent_tensor = sent_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
        emb = embed_layer(sent_tensor)
        all_embedding = torch.cat([all_embedding, emb], dim=0)

    return all_embedding

class cnn_basic_certain_hidden:
    def __init__(self, model: basic_CNNClassifier, embedding, target_id, cell_num, device, isBatch=False):
        if not isinstance(model, basic_CNNClassifier):
            raise ValueError("Only support basic CNNClassifier!")
        self.model = model
        model: basic_CNNClassifier
        self.isBatch = isBatch
        self.target_id = target_id
        self.cell_num = cell_num
    def get_hidden(self, embedding):
        if self.cell_num == 0:
            return embedding
        inp = embedding.unsqueeze(1)
        if self.cell_num <= len(self.model.encoder.cnn_module):
            moduleL = self.model.encoder.cnn_module[self.cell_num - 1:self.cell_num]
            for module in moduleL:
                cur_output = module(inp)
                cur_output = self.model.encoder.activation(cur_output)
                pooled = torch.max(cur_output, dim=2)[0]
                pooled = pooled.view(pooled.size(0), pooled.size(1))
                return pooled if self.isBatch else pooled[0]
        last_ = torch.tensor(()).to(device)
        for module in self.model.encoder.cnn_module:
            cur_output = module(inp)          # B * c * S * 1
            cur_output = self.model.encoder.activation(cur_output)
            pooled = torch.max(cur_output, dim=2)[0]
            pooled = pooled.view(pooled.size(0), pooled.size(1))
            last_ = torch.cat([last_, pooled], dim=1)
        if self.cell_num == len(self.model.encoder.cnn_module) + 1:
            return last_ if self.isBatch else last_[0]
        logit = self.model.decoder(last_)
        if self.cell_num == len(self.model.encoder.cnn_module) + 2:
            return logit if self.isBatch else logit[0]
        else:
            raise ValueError("Layer number exceed!!!")

class lstm_basic_certain_hidden:
    """
    params:
    - model  : type of model in model.py
    - embedding : the embedding of word
    - target_id : the target id of certain hidden layer. -1 stands for all.
    - layer_num : the target layer number. Start at 1.
    - word_embeding_flag : indicating whether the embedding is the word embeding.
    - isBatch : whether keep batch info.
    """
    def __init__(self, model, embedding, target_id, layer_num, device, isBatch=False):
        if not isinstance(model, elmo_RNNClassifier):
            raise ValueError("can only support lstm custom hiddens")
        model: elmo_RNNClassifier
        self.model = model
        encoder = model.encoder
        rnn = encoder.rnn
        self.embedding = embedding
        self.isBatch = isBatch
        self.layer_num = layer_num
        self.target_id = target_id
        params = rnn.state_dict()
        cur_params = {}
        keys = ['weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0', 'weight_ih_l0_reverse', 'weight_hh_l0_reverse', 'bias_ih_l0_reverse', 'bias_hh_l0_reverse']
        for i in range(min(layer_num, rnn.num_layers)):
            for k in keys:
                cur_params[k.replace('0',str(i))] = params[k.replace('0',str(i))]
        if isinstance(encoder.rnn, torch.nn.LSTM):
            cell = torch.nn.LSTM(rnn.input_size, rnn.hidden_size, min(layer_num, rnn.num_layers), batch_first=True, bidirectional=True).to(embedding)
        elif isinstance(encoder.rnn, torch.nn.GRU):
            cell = torch.nn.GRU(rnn.input_size, rnn.hidden_size, min(layer_num, rnn.num_layers), batch_first=True, bidirectional=True).to(embedding)
        cell.load_state_dict(cur_params)
        for param in cell.parameters():
            param.requires_grad = False
        self.cell = cell

    def get_hidden(self, embedding):
        cur_output = embedding
        if self.layer_num == 0:
            return cur_output
        elif self.layer_num >= self.model.encoder.layer_num:
            cur_output,_ = self.cell(cur_output)
            if self.layer_num >= self.model.encoder.layer_num + 1:
                pooled = self.model.pooler(cur_output, mask=torch.ones(cur_output.shape[0], cur_output.shape[1]).to(device))
                if self.layer_num == self.model.encoder.layer_num + 1:
                    return pooled if self.isBatch else pooled[0]
            if self.layer_num >= self.model.encoder.layer_num + 2:
                logit = self.model.decoder(pooled)
                if self.layer_num == self.model.encoder.layer_num + 2:
                    return logit if self.isBatch else logit[0]
                else:
                    raise ValueError("Invalid layer number!")
        else:
            cur_output,_ = self.cell(cur_output)
        if isinstance(self.target_id, int):
            if self.target_id == -1:
                return cur_output if self.isBatch else cur_output[0]
            else:
                return cur_output[:,self.target_id] if self.isBatch else cur_output[0, self.target_id]
        else:
            target_id = torch.tensor(self.target_id, dtype=torch.long).to(device)
            target_id = target_id.unsqueeze(-1)
            target_id = target_id.expand(target_id.size(0), cur_output.size(2))
            cur_output = cur_output.gather(1, target_id.unsqueeze(1))
            cur_output = cur_output.view(cur_output.size(0), cur_output.size(2))
            return cur_output

def preprocess_sentence_batch(model, types, sent1, target_id, target_layer, device, sent2=None):
    """
    - model: basic_RNNClassifier or basic_CNNClassifier
    - types: selected from ['rnn','cnn']

    - return: word_embedding [torch.FloatTensor], embed_to_hidden [class instance], tokenized [List[Str]]
    """
    example = [InputExample(0, sent1, sent2)]
    curD = dataset(example, global_tokenizer)
    part1 = ['[CLS]'] + global_tokenizer.tokenize(sent1) + ['[SEP]']
    #segId1 = [0 for _ in range(len(part1))]
    if sent2 is not None:
        part2 = global_tokenizer.tokenize(sent2) + ['[SEP]']
        #segId2 = [1 for _ in range(len(part2))]
    else:
        part2 = []
        #segId2 = []
    tokenized = part1 + part2
    #segIds = segId1 + segId2
    emb = get_basic_embedding(model, types, curD)
    if types == 'rnn':
        eth = lstm_basic_certain_hidden(model, emb, target_id, target_layer, device, isBatch=True)
    else:
        eth = cnn_basic_certain_hidden(model, emb, target_id, target_layer, device, isBatch=True)
    return emb, eth, tokenized#, segIds

def preprocess_sentence_basic(model, types, sent1, target_id, target_layer, sent2=None):
    """
    - model: basic_RNNClassifier or basic_CNNClassifier
    - types: selected from ['rnn','cnn']

    - return: word_embedding [torch.FloatTensor], embed_to_hidden [class instance], tokenized [List[Str]]
    """
    example = [InputExample(0, sent1, sent2)]
    curD = dataset(example, global_tokenizer)
    part1 = ['[CLS]'] + global_tokenizer.tokenize(sent1) + ['[SEP]']
    #segId1 = [0 for _ in range(len(part1))]
    if sent2 is not None:
        part2 = global_tokenizer.tokenize(sent2) + ['[SEP]']
        #segId2 = [1 for _ in range(len(part2))]
    else:
        part2 = []
        #segId2 = []
    tokenized = part1 + part2
    #segIds = segId1 + segId2
    emb = get_basic_embedding(model, types, curD)
    if types == 'rnn':
        eth = lstm_basic_certain_hidden(model, emb, target_id, target_layer, isBatch=False)
    else:
        eth = cnn_basic_certain_hidden(model, emb, target_id, target_layer, isBatch=False)
    return emb, eth, tokenized#, segIds


if __name__ == '__main__':
    import argparse
    from bert_helper import tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('sentence',type=str)
    parser.add_argument('--model',default='lstm',type=str)
    parser.add_argument('--task',default='sst2',type=str)
    args = parser.parse_args()
    model = load_basic_model(args.model, args.task, -1)
    sent = args.sentence
    tokenized = ['[CLS]'] + tokenizer.tokenize(sent) + ['[SEP]']
    idx_sequence = tokenizer.convert_tokens_to_ids(tokenized)
    input_tensor = torch.tensor([idx_sequence], dtype=torch.long)
    result = model(input_tensor, mask=torch.ones_like(input_tensor, dtype=torch.long))
    result = result.tolist()[0]
    print('The logit is ' + str(result))
