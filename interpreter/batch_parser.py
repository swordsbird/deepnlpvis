import multiprocessing
import time
import sys
import random
from layer_parser import parser_text
import argparse
import model_helper
import torch
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some args.')
    # parser.add_argument(
    #    "-f", "--file", default='../../../dataset/CoLA/train.tsv')
    parser.add_argument(
        "-f", "--file", default='../../dataset/sst2_10k_train.tsv')
    parser.add_argument("-m", "--model", default='sst2_12')
    parser.add_argument("-r", "--regular",
                        action='store_true', help='generate regular')
    parser.add_argument("-neuron", "--neuron",
                        action='store_true', help='use neuron clusters')
    parser.add_argument("-word", "--word_level",
                        action='store_true', help='is word level')
    parser.add_argument("-sl", "--single_layer",
                        action='store_true', help='is single layer')
    parser.add_argument("-al", "--all_layer",
                        action='store_true', help='is all layer')
    parser.add_argument("-gpu", "--gpus", default='0', type=str)
    parser.add_argument("-t", "--task_per_gpu", default=1, type=int)
    parser.add_argument("-n_layer", "--n_layer", default=12, type=int)
    parser.add_argument("-n_neuron", "--n_neuron", default=768, type=int)
    parser.add_argument("-start", "--start", default=0, type=int)
    parser.add_argument("-end", "--end", default=-1, type=int)
    parser.add_argument("-ids", "--ids", default='', type=str)
    args = parser.parse_args(sys.argv[1:])
    gpus = args.gpus.split(',')
    n = len(gpus)
    task_per_gpu = args.task_per_gpu
    start = args.start
    end = args.end
    n_layer = args.n_layer
    n_neuron = args.n_neuron
    infile = args.file

    texts = open(infile, 'r').read()
    texts = texts.split('\n')[1:]
    texts = [x.split('\t')[0] for x in texts if len(x.split('\t')) == 2]

    model_name = args.model.split('_')[0]
    model_number = int(args.model.split('_')[1])
    if model_name == 'sst2':
        model = model_helper.load_sst2_model(model_number)
    else:
        model = model_helper.load_cola_model(model_number)

    if args.regular:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        from pytorch_pretrained_bert import BertTokenizer
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

        texts = random.sample(texts, 1000)
        sample_s = [[] for i in range(n_layer + 1)]

        it = 0
        for t in texts:
            if it % 100 == 0:
                print(f'{it} iters')
            it += 1
            s0 = get_embedding_from_text(t)
            for k in range(len(s0)):
                s = s0[k]
                s = s.mean(axis=0)
                sample_s[k].append(s.tolist())
        for k in range(len(sample_s)):
            sample_s[k] = np.array(sample_s[k])
            sample_s[k] = np.std(sample_s[k], axis=0)

        np.save('regular.npy', np.array(sample_s))
        exit()

    regularizations = np.load('regular.npy')
    texts = [(texts[k], k) for k in range(len(texts))]
    if args.ids:
        ids = [int(i) for i in args.ids.split(',')]
        texts = [texts[i] for i in range(len(texts)) if i in ids]
    else:
        texts = texts[start:end]
    random.shuffle(texts)

    neurons = []
    if args.neuron:
        neuron_labels = np.load('label.npy')
        for i in range(n_layer):
            n_cluster = int(neuron_labels[i].max()) + 1
            clusters = []
            for k in range(n_cluster):
                clusters.append([j for j in range(n_neuron)
                                 if neuron_labels[i][j] == k])
            neurons.append(clusters)
    else:
        for i in range(n_layer):
            clusters = [range(n_neuron)]
            neurons.append(clusters)

    tasks = []
    for k in range(n):
        for i in range(task_per_gpu):
            j = k * task_per_gpu + i
            start = len(texts)*j // (n*task_per_gpu)
            end = len(texts)*(j+1) // (n*task_per_gpu)
            textsplit = [texts[k] for k in range(start, end)]
            p = multiprocessing.Process(target=parser_text, args=(
                textsplit, gpus[k], neurons, regularizations, model, args.word_level, args.single_layer, args.all_layer))
            tasks.append(p)
            p.start()

    for p in tasks:
        p.join()
