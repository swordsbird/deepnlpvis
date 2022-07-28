from pytorch_pretrained_bert import BertTokenizer
import random

max_length = 64

def CoLA_processor(path_to_dataset):

    all_content = open(path_to_dataset,'r',encoding='utf8').readlines()
    dataset = []
    label_dict = {'0':0,'1':1}
    for idx,line in enumerate(all_content):
        line = line.strip().split('\t')
        dataset.append(InputExample(guid = idx, text_a = line[3], text_b = None, label = label_dict[line[1]]))
    
    return dataset, 2

def QQP_processor(path_to_dataset):

    all_content = open(path_to_dataset, 'r', encoding='utf8').readlines()[1:]
    dataset = []
    label_dict = {'0':0,'1':1}
    for idx,line in enumerate(all_content):
        line = line.strip().split('\t')
        if len(line) < 6:
            print(idx,line)
            exit(-1)
        dataset.append(InputExample(guid = idx, text_a = line[3], text_b = line[4], label = label_dict[line[5]]))
    return dataset, 2

def SST_2_processor(path_to_dataset):

    all_content = open(path_to_dataset, 'r', encoding='utf8').readlines()[1:]
    dataset = []
    label_dict = {'0':0,'1':1}
    for idx, line in enumerate(all_content):
        line = line.strip().split('\t')
        dataset.append(InputExample(guid = idx, text_a = line[0], text_b = None, label = label_dict[line[1]]))
    return dataset, 2

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) int. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class pair_dataset:
    def __init__(self, instanceList: list, tokenizer: BertTokenizer):
        self.datasetList = [ele for ele in instanceList]
        self.tokenizer = tokenizer
    
    def make_input(self, batch_size, rand_flag=True):
        """
        make batch input
        -param batch_size: the batch size of this data
        -param rand_flag: whether the input data is random shuffled
        -return: yield return. Return a dict with key 'sent', 'mask', 'label', 'segment'
        """
        if rand_flag:
            random.shuffle(self.datasetList)
        cur_idx = 0
        while cur_idx < len(self.datasetList):
            cur_batch = self.datasetList[cur_idx:cur_idx + batch_size]
            tokenized_a, tokenized_b = [], []
            max_a = 0
            max_b = 0
            label_batch = []
            for instance in cur_batch:
                instance: InputExample
                text_a = ['[CLS]'] + self.tokenizer.tokenize(instance.text_a) + ['[SEP]']
                text_b = ['[CLS]'] + self.tokenizer.tokenize(instance.text_b) + ['[SEP]']
                tokenized_a.append(text_a)
                tokenized_b.append(text_b)
                if len(text_a) > max_a:
                    max_a = len(text_a)
                if len(text_b) > max_b:
                    max_b = len(text_b)
                label_batch.append(instance.label)

            sent_a, sent_b, mask_a, mask_b = [], [], [], []

            max_a = max_a if max_a < max_length else max_length
            max_b = max_b if max_b < max_length else max_length

            for idx in range(len(tokenized_a)):
                aaa = tokenized_a[idx][:max_a]
                bbb = tokenized_b[idx][:max_b]
                sent_a.append(self.tokenizer.convert_tokens_to_ids(aaa + ['[PAD]' for _ in range(max_a - len(aaa))]))
                sent_b.append(self.tokenizer.convert_tokens_to_ids(bbb + ['[PAD]' for _ in range(max_b - len(bbb))]))
                mask_a.append([1 for _ in range(len(aaa))] + [0 for _ in range(max_a - len(aaa))])
                mask_b.append([1 for _ in range(len(bbb))] + [0 for _ in range(max_b - len(bbb))])
            
            cur_idx += batch_size

            yield {
                'sent_a': sent_a,
                'sent_b': sent_b,
                'mask_a': mask_a,
                'mask_b': mask_b,
                'label': label_batch
            }

    def all_input(self, batch_size, rand_flag=True):
        """
        make batch input
        -param batch_size: the batch size of this data
        -param rand_flag: whether the input data is random shuffled
        -return: yield return. Return a dict with key 'sent', 'mask', 'label', 'segment'
        """
        if rand_flag:
            random.shuffle(self.datasetList)
        cur_idx = 0

        cur_batch = self.datasetList[cur_idx:cur_idx + batch_size]
        tokenized_a, tokenized_b = [], []
        max_a = 0
        max_b = 0
        label_batch = []
        for instance in cur_batch:
            instance: InputExample
            text_a = ['[CLS]'] + self.tokenizer.tokenize(instance.text_a) + ['[SEP]']
            text_b = ['[CLS]'] + self.tokenizer.tokenize(instance.text_b) + ['[SEP]']
            tokenized_a.append(text_a)
            tokenized_b.append(text_b)
            if len(text_a[-1]) > max_a:
                max_a = len(text_a[-1])
            if len(text_b[-1]) > max_b:
                max_b = len(text_b[-1])
            label_batch.append(instance.label)

        sent_a, sent_b, mask_a, mask_b = [], [], [], []

        max_a = max_a if max_a < max_length else max_length
        max_b = max_b if max_b < max_length else max_length

        for idx in range(len(text_a)):
            aaa = text_a[idx][:max_a]
            bbb = text_b[idx][:max_b]
            sent_a.append(self.tokenizer.convert_tokens_to_ids(aaa + ['[PAD]' for _ in range(max_a - len(aaa))]))
            sent_b.append(self.tokenizer.convert_tokens_to_ids(bbb + ['[PAD]' for _ in range(max_b - len(bbb))]))
            mask_a.append([1 for _ in range(len(aaa))] + [0 for _ in range(max_a - len(aaa))])
            mask_b.append([1 for _ in range(len(bbb))] + [0 for _ in range(max_b - len(bbb))])
        
        cur_idx += batch_size

        return {
            'sent_a': sent_a,
            'sent_b': sent_b,
            'mask_a': mask_a,
            'mask_b': mask_b,
            'label': label_batch
        }

def _truncate_pair_sentence(token_a, token_b, max_length):
    while len(token_a) + len(token_b) > max_length:
        if len(token_a) >= len(token_b):
            token_a.pop()
        else:
            token_b.pop()

class dataset:
    def __init__(self, instanceList: list, tokenizer: BertTokenizer, fix_length=-1):
        self.datasetList = [ele for ele in instanceList]
        self.tokenizer = tokenizer
        self.fix_length = fix_length

    def make_input(self, batch_size, rand_flag=True):
        """
        make batch input
        -param batch_size: the batch size of this data
        -param rand_flag: whether the input data is random shuffled
        -return: yield return. Return a dict with key 'sent', 'mask', 'label', 'segment'
        """
        if rand_flag:
            random.shuffle(self.datasetList)
        cur_idx = 0
        while cur_idx < len(self.datasetList):
            cur_batch = self.datasetList[cur_idx:cur_idx + batch_size]
            sent_batch = []
            label_batch = []
            seg_batch = []
            if self.fix_length > 0:
                max_len = self.fix_length
                max_length = max_len
            else:
                max_len = 0
                max_length = 10000
            for instance in cur_batch:
                instance: InputExample
                #text_a = ['[CLS]'] + self.tokenizer.tokenize(instance.text_a) + ['[SEP]']
                text_a = self.tokenizer.tokenize(instance.text_a)
                if instance.text_b is not None:
                    text_b = self.tokenizer.tokenize(instance.text_b)
                else:
                    text_b = []
                _truncate_pair_sentence(text_a, text_b, max_length - 3)
                text_a = ['[CLS]'] + text_a + ['[SEP]']
                if instance.text_b is not None:
                    text_b = text_b + ['[SEP]']
                seg_a = [0 for _ in range(len(text_a))]
                seg_b = [1 for _ in range(len(text_b))]
                all_tokenized_text = text_a + text_b
                seg_ins = seg_a + seg_b
                seg_batch.append(seg_ins)
                sent_batch.append(all_tokenized_text)
                label_batch.append(instance.label)
                if self.fix_length <= 0:
                    if len(sent_batch[-1]) > max_len:
                        max_len = len(sent_batch[-1])
            sent_id_batch = []
            mask_batch = []
            
            for idx, sent in enumerate(sent_batch):
                sent_id_batch.append(self.tokenizer.convert_tokens_to_ids(sent + ['[PAD]' for _ in range(max_len - len(sent))]))
                seg_batch[idx] = seg_batch[idx] + [0 for _ in range(max_len - len(sent))]
                mask_batch.append([1 for _ in range(len(sent))] + [0 for _ in range(max_len - len(sent))])
            
            # sent_id_batch is B * S
            # mask_batch is B * S

            cur_idx += batch_size

            yield {
                'sent': sent_id_batch,
                'mask': mask_batch,
                'segment': seg_batch,
                'label': label_batch
            }

    def all_input(self, batch_size, rand_flag=True):
        """
        select some input of batch_size
        - param batch_size: the batch size of this data
        - param rand_flag: whether the input data is random shuffled
        - return: yield return. Return a dict with key 'sent', 'mask', 'label', 'segment'
        """
        if rand_flag:
            random.shuffle(self.datasetList)
        cur_idx = 0
        cur_batch = self.datasetList[cur_idx:cur_idx + batch_size]
        sent_batch = []
        label_batch = []
        seg_batch = []
        max_len = 0
        for instance in cur_batch:
            instance: InputExample
            text_a = ['[CLS]'] + self.tokenizer.tokenize(instance.text_a) + ['[SEP]']
            seg_a = [0 for _ in range(len(text_a))]
            if instance.text_b is not None:
                text_b = self.tokenizer.tokenize(instance.text_b)
            else:
                text_b = []
            seg_b = [1 for _ in range(len(text_b))]
            all_tokenized_text = text_a + text_b
            seg_ins = seg_a + seg_b
            seg_batch.append(seg_ins)
            sent_batch.append(all_tokenized_text)
            label_batch.append(instance.label)
            if len(sent_batch[-1]) > max_len:
                max_len = len(sent_batch[-1])
        sent_id_batch = []
        mask_batch = []

        for idx, sent in enumerate(sent_batch):
            sent_id_batch.append(
                self.tokenizer.convert_tokens_to_ids(sent + ['[PAD]' for _ in range(max_len - len(sent))]))
            seg_batch[idx] = seg_batch[idx] + [0 for _ in range(max_len - len(sent))]
            mask_batch.append([1 for _ in range(len(sent))] + [0 for _ in range(max_len - len(sent))])

        # sent_id_batch is B * S
        # mask_batch is B * S

        return {
            'sent': sent_id_batch,
            'mask': mask_batch,
            'segment': seg_batch,
            'label': label_batch
        }
