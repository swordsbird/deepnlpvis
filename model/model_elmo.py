import torch
from torch import nn
from torch.nn import functional as F
import yaml
from pytorch_pretrained_bert import BertTokenizer
from allennlp.modules.elmo import Elmo, batch_to_ids

global_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
total_word = len(global_tokenizer.vocab)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class elmo_WordEmbedLayer(nn.Module):
    """
    Word embedding layer. Consist of word, segment and position embedding. Just return the sum of all these embeddings.

    @init params:
    - config: Dict,
        - hidden_size: the hidden size of word, segment, position embedding
    
    @input:
    - input_ids: Torch.LongTensor, (Batch, SeqLen). The ids of input words.
    - token_type_ids (optional): Torch.LongTensor, (Batch, SeqLen). The ids of input segments.
    - attention_mask (optional): Torch.LongTensor, (Batch, SeqLen). A mask indicating where is word.
    
    @output:
    Torch.FloatTensor, (Batch, SeqLen, EmbDim). The embedded tensor.
    """
    def __init__(self, config):
        super(elmo_WordEmbedLayer, self).__init__()
        if isinstance(config, str):
            config = yaml.load(config)
        self.elmo = Elmo('~/data/large/elmo/options.json', '~/data/large/elmo/weights.hdf5', 1, dropout=0)
        #global_tokenizer.vocab
        self.elmo_emb_size = 1024
        self.word_embed_layer = nn.Embedding(total_word, config['hidden_size'])
    
    def forward(self, input_ids):
        sentences = []
        for i in range(input_ids.shape[0]):
            sentences.append(global_tokenizer.convert_ids_to_tokens(input_ids[i].tolist()))
        character_ids = batch_to_ids(sentences)
        character_ids = character_ids.cuda()
        elmo_emb = self.elmo(character_ids)['elmo_representations'][0]
        emb = self.word_embed_layer(input_ids)
        return torch.cat([emb, elmo_emb], dim=2)

class basic_Pooler(nn.Module):
    ''' Do pooling, possibly with a projection beforehand '''

    def __init__(self, d_inp, project=True, d_proj=512, pool_type='max'):
        super(basic_Pooler, self).__init__()
        self.project = nn.Linear(d_inp, d_proj) if project else lambda x: x
        self.pool_type = pool_type

    def forward(self, sequence, mask):
        if len(mask.size()) < 3:
            mask = mask.unsqueeze(dim=-1)
        pad_mask = (mask == 0)
        proj_seq = self.project(sequence)  # linear project each hid state
        if self.pool_type == 'max':
            proj_seq = proj_seq.masked_fill(pad_mask, -float('inf'))
            seq_emb = proj_seq.max(dim=1)[0]
        elif self.pool_type == 'mean':
            proj_seq = proj_seq.masked_fill(pad_mask, 0)
            seq_emb = proj_seq.sum(dim=1) / mask.sum(dim=1).float()
        elif self.pool_type == 'final':
            idxs = mask.expand_as(proj_seq).sum(dim=1, keepdim=True).long() - 1
            seq_emb = proj_seq.gather(dim=1, index=idxs)
            seq_emb = seq_emb.view(seq_emb.size(0), seq_emb.size(2))
        return seq_emb

class elmo_RNNEncoder(nn.Module):
    def __init__(self, config, embedder=None):
        super(elmo_RNNEncoder, self).__init__()
        if isinstance(config, str):
            config = yaml.load(open(config))
        self.hidden_size = config['hidden_size']
        self.hidden_state_size = config['hidden_state_size']
        self.word_vocab = total_word
        if embedder is None:
            self.embed_layer = elmo_WordEmbedLayer(config)
        else:
            self.embed_layer = embedder
        self.rnn_type = config['rnn_type']
        self.layer_num = config['layer_num']
        self.dropout_rate = config['dropout']
        self.bidirectional = config['bidirectional_flag']
        self.elmo_emb_size = 1024
        #self.attention = config['use_attention']
        self.config = config
        if self.rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_state_size, num_layers=self.layer_num, batch_first=True, dropout=self.dropout_rate, bidirectional=self.bidirectional)
        elif self.rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(input_size=self.hidden_size+self.elmo_emb_size, hidden_size=self.hidden_state_size, num_layers=self.layer_num, batch_first=True, dropout=self.dropout_rate, bidirectional=self.bidirectional)

    def forward(self, input_ids, all_output = True):
        embed = self.embed_layer(input_ids)
        final_output, _ = self.rnn(embed)
        if all_output:
            return final_output
        else:
            return final_output[:,0]

    @classmethod
    def from_pretrained(config, model_path = None):
        model = elmo_RNNEncoder(config)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        return model

class elmo_RNNClassifier(nn.Module):
    def __init__(self, config, label_number = 2):
        super(elmo_RNNClassifier, self).__init__()
        self.encoder = elmo_RNNEncoder(config)
        self.label_number = label_number
        self.pooler = basic_Pooler((1 + self.encoder.bidirectional) * self.encoder.hidden_size, config['project'], config['dim_project'], config['pool_type'])
        self.decoder = nn.Linear(config['decode_input_dimension'], self.label_number)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_ids, mask = None, labels = None):        
        output = self.encoder(input_ids)
        pooled = self.pooler(output, mask)
        if labels is None:
            return self.decoder(pooled)
        
        logits = self.decoder(pooled)
        loss = self.loss_function(logits.view(-1, self.label_number), labels.view(-1))
        return loss, logits
    
    @classmethod
    def from_pretrained(config, model_path = None, label_number = 2):
        model = elmo_RNNClassifier(config, label_number)
        if model_path is None:
            model.encoder.load_state_dict(torch.load(model_path))
        return model

class basic_CNNEncoder(nn.Module):
    def __init__(self,config, embedder=None):
        super(basic_CNNEncoder, self).__init__()
        if isinstance(config, str):
            config = yaml.load(open(config))
        self.hidden_size = config['hidden_size']
        if embedder is None:
            self.embed_layer = elmo_WordEmbedLayer(config)
        else:
            self.embed_layer = embedder
        self.cnn_module = nn.ModuleList()
        for i, ele in enumerate(config['cnn_kernel']):
            self.cnn_module.add_module('cnn_%d' % (i), nn.Conv2d(1, ele[0], (ele[1], self.hidden_size), padding=(ele[1] // 2, 0)))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, input_ids, mask):
        embedding = self.embed_layer(input_ids)
        embedding = embedding.unsqueeze(1)          # B * 1 * S * D
        last_ = torch.tensor(()).to(device)
        for module in self.cnn_module:
            cur_output = module(embedding)          # B * c * S * 1
            cur_output = self.activation(cur_output)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(1).unsqueeze(-1)   # B * 1 * S * 1
            mask_pad = (mask == 0)
            cur_output = cur_output.masked_fill(mask_pad, -float('inf'))
            pooled = torch.max(cur_output, dim=2)[0]
            pooled = pooled.view(pooled.size(0), pooled.size(1))
            last_ = torch.cat([last_, pooled], dim=1)
        return self.dropout(last_)
    
class basic_CNNClassifier(nn.Module):
    def __init__(self, config, label_number=2):
        super(basic_CNNClassifier, self).__init__()
        if isinstance(config, str):
            config = yaml.load(open(config))
        self.encoder = basic_CNNEncoder(config)
        self.label_number = label_number
        self.decoder = nn.Linear(config['decode_input_dimension'], self.label_number)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_ids, mask=None, labels=None):
        if mask is None:
            mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
        output = self.encoder(input_ids, mask)
        if labels is None:
            return self.decoder(output)
        
        logits = self.decoder(output)
        loss = self.loss_function(logits.view(-1, self.label_number), labels.view(-1))
        return loss, logits

def calculate_attn(seq_1, seq_2, mask_1, mask_2):
    """
    Calculate Bi-attention of seq1 and seq2
    """
    attn = seq_1 @ seq_2.transpose(1,2) # B * S1 * S2
    #mask_attn = mask_1.unsqueeze(2).float() @ mask_2.unsqueeze(1).transpose(1,2).float()    # B * S1 * S2
    attn[mask_1 == 0] == - 1000.
    attn[mask_2.unsqueeze(1).expand_as(attn) == 0] == - 1000.
    #attn[mask_attn < 0.5] == - 1000.
    re_seq_2 = torch.softmax(attn, dim=1).transpose(1,2) @ seq_1   # B * S2 * D
    re_seq_1 = torch.softmax(attn, dim=2) @ seq_2   # B * S1 * D
    return torch.cat([seq_1, re_seq_1], dim=2), torch.cat([seq_2, re_seq_2], dim=2)

class basic_RNNPairClassifier(nn.Module):
    def __init__(self, config, label_number=2):
        super(basic_RNNPairClassifier, self).__init__()
        if isinstance(config, str):
            config = yaml.load(open(config))
        self.encoder = elmo_RNNEncoder(config['encoder'])
        self.label_number = label_number
        self.pooler = basic_Pooler(config['encoder']['hidden_state_size'] * 4, config['project'], config['dim_project'], config['pool_type'])
        self.decoder_1 = nn.Linear(config['decode_input_dimension'], config['decode_input_dimension'] // 4)
        self.decoder_2 = nn.Linear(config['decode_input_dimension'] // 4, self.label_number)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, input_a, input_b, mask_a=None, mask_b=None, labels=None):
        if mask_a is None:
            mask_a = torch.ones(input_a.shape, dtype=torch.long, device=device)
        if mask_b is None:
            mask_b = torch.ones(input_b.shape, dtype=torch.long, device=device)

        seq_1 = self.encoder(input_a)
        seq_2 = self.encoder(input_b)

        seq_1, seq_2 = calculate_attn(seq_1, seq_2, mask_a, mask_b)

        seq_1 = self.pooler(seq_1, mask_a)
        seq_2 = self.pooler(seq_2, mask_b)

        xxx = torch.cat([seq_1, seq_2, torch.abs(seq_1 - seq_2), seq_1 * seq_2], dim=1)
        logit = self.decoder_2(self.decoder_1((xxx)))

        if labels is None:
            return logit
        
        loss = self.loss_function(logit.view(-1, self.label_number), labels.view(-1))
        return loss, logit

class basic_CNNPairClassifier(nn.Module):
    def __init__(self, config, label_number=2):
        super(basic_CNNPairClassifier, self).__init__()
        if isinstance(config, str):
            config = yaml.load(open(config))
        self.encoder = basic_CNNEncoder(config['encoder'])
        self.label_number = label_number
        self.decoder_1 = nn.Linear(config['decode_input_dimension'], config['decode_input_dimension'] // 4)
        self.decoder_2 = nn.Linear(config['decode_input_dimension'] // 4, self.label_number)
        self.loss_function = nn.CrossEntropyLoss()
    
    def forward(self, input_a, input_b, mask_a=None, mask_b=None, labels=None):
        if mask_a is None:
            mask_a = torch.ones(input_a.shape, dtype=torch.long, device=device)
        if mask_b is None:
            mask_b = torch.ones(input_b.shape, dtype=torch.long, device=device)

        pooled_1 = self.encoder(input_a, mask_a)
        pooled_2 = self.encoder(input_b, mask_b)
        
        xxx = torch.cat([pooled_1, pooled_2, torch.abs(pooled_1 - pooled_2), pooled_1 * pooled_2], dim=1)
        logit = self.decoder_2(self.decoder_1((xxx)))

        if labels is None:
            return logit
        
        loss = self.loss_function(logit.view(-1, self.label_number), labels.view(-1))
        return loss, logit