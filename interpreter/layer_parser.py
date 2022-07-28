def parser_text(texts, gpu_id, n_layer, regularizations, model, classifier, modes):
    from pytorch_pretrained_bert import BertTokenizer
    import torch
    import os
    import numpy as np
    import logging
    from basic_model_helper import set_device

    set_device(str(gpu_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.getLogger().setLevel(logging.WARNING)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    home_path = '/home/lizhen/'
    def multi_disturb_input_embed_gaussian(x, disturb_locs, var = 2.5, n_batch = 120):
        x = x.unsqueeze(0).repeat(n_batch, 1, 1)
        for i in disturb_locs:
            epsilon = torch.randn((n_batch, x.shape[2])).to(device) * var
            x[:, i, :] += epsilon
        return x

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.to(device)
    weight, bias = classifier
    if weight is not None:
        weight = torch.tensor(weight, device = device)
    if bias is not None:
        bias = torch.tensor(bias, device = device)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    def get_logits(x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        return logits[0]

    def get_real_logits(x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        logits = torch.matmul(x[:,0,:], weight) + bias
        logits = torch.softmax(logits, dim=1)
        logits = logits.mean(axis=0)
        return logits


    def get_embedding_from_text(text):
        words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]
        tokenized_ids = tokenizer.convert_tokens_to_ids(words)
        segment_ids = [0 for _ in range(len(words))]
        token_tensor = torch.tensor([tokenized_ids], device=device)
        segment_tensor = torch.tensor([segment_ids], device=device)
        x = model.embeddings(token_tensor, segment_tensor)[0]
        return x

    from Interpreter import Interpreter as InterpreterBatch

    batch_size = 12

    def get_embedding_from_layer(x, layer=-1):
        attention_mask = torch.ones(x.shape[:2]).to(x.device)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        model_list = model.encoder.layer[layer:]
        hidden_states = x
        for layer_module in model_list:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states

    def get_embedding(x, start, end):
        x = x.unsqueeze(0)
        attention_mask = torch.ones(x.shape[:2]).to(x.device)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        model_list = model.encoder.layer[start:end]
        hidden_states = x
        for layer_module in model_list:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states[0]

    def get_embedding_of_layer(x, layer=-1):
        x = x.unsqueeze(0)
        attention_mask = torch.ones(x.shape[:2]).to(x.device)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        model_list = model.encoder.layer[:layer]
        hidden_states = x
        for layer_module in model_list:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states[0]

    def get_sigma(text, layer, n_iter=80, lr=0.012, select_n=None, to_prediction=False, to_weight=False):
        x = get_embedding_from_text(text)
        if to_prediction:
            x = get_embedding_of_layer(x, layer)
            regular = (0.1, regularizations[layer])
            rate = 0.1
            scale = 0.5
            phi = get_phi_start_from(layer)
        elif to_weight:
            x = get_embedding_of_layer(x, layer)
            regular = (0.1, regularizations[layer])
            rate = 0.1
            scale = 0.5
            phi = get_single_layer_phi(layer)
        else:
            regular = regularizations[layer]
            rate = 0.1
            scale = 0.5
            phi = get_phi_from_input_to(layer)
        interpreter = InterpreterBatch(x=x, Phi=phi, scale=scale, n=batch_size, rate=rate, dim=select_n, regularization=regular).to(
            device
        )
        interpreter.optimize(iteration=n_iter , lr=lr, show_progress=False)
        return interpreter.get_sigma()

    text = "rare bird has more than enough charm to make it memorable."

    def get_phi_start_from(layer):
        # extract the Phi we need to explain
        def Phi(x):
            no_batch = len(x.shape) == 2
            if no_batch:
                x = x.unsqueeze(0)
            #print(x.shape)
            attention_mask = torch.ones(x.shape[:2]).to(x.device)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # extract the 3rd layer
            model_list = model.encoder.layer[layer:]
            hidden_states = x
            for layer_module in model_list:
                hidden_states = layer_module(hidden_states, extended_attention_mask)
            #print(hidden_states.shape)
            hidden_states = get_logits(hidden_states)
            # print('shape of hidden state', hidden_states.shape)
            if no_batch:
                ret = hidden_states[0]
            else:
                ret = hidden_states
            return ret
        return Phi

    def get_single_layer_phi(layer):
        # extract the Phi we need to explain
        def Phi(x):
            no_batch = len(x.shape) == 2
            if no_batch:
                x = x.unsqueeze(0)
            #print(x.shape)
            attention_mask = torch.ones(x.shape[:2]).to(x.device)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # extract the 3rd layer
            model_list = model.encoder.layer[layer:layer+1]
            hidden_states = x
            for layer_module in model_list:
                hidden_states = layer_module(hidden_states, extended_attention_mask)
            if no_batch:
                ret = hidden_states[0]
            else:
                ret = hidden_states
            return ret
        return Phi

    def get_phi_from_input_to(layer):
        # extract the Phi we need to explain
        def Phi(x):
            no_batch = len(x.shape) == 2
            if no_batch:
                x = x.unsqueeze(0)
            #print(x.shape)
            attention_mask = torch.ones(x.shape[:2]).to(x.device)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            # extract the 3rd layer
            model_list = model.encoder.layer[:layer]
            hidden_states = x
            for layer_module in model_list:
                hidden_states = layer_module(hidden_states, extended_attention_mask)
            if no_batch:
                ret = hidden_states[0]
            else:
                ret = hidden_states
            return ret
        return Phi

    print('run task on ', len(texts), 'sentences')
    for text in texts:
        for mode in modes:
            file_name = 'output/' + mode + '/' + str(text[1]) + '.npy'
            if os.path.exists(file_name):
                print(f'{file_name} was already existed. skip.')
                continue
            agg_sigma = []
            words = ["[CLS]"] + tokenizer.tokenize(text[0]) + ["[SEP]"]
            if mode == 'layer':
                for layer in range(n_layer):
                    sigma = get_sigma(text[0], layer + 1, to_prediction = False)
                    agg_sigma.append(sigma)
                agg_sigma = np.array(agg_sigma)
            elif mode == 'contri':
                for layer in range(n_layer):
                    sigma = get_sigma(text[0], layer, to_prediction = True)
                    agg_sigma.append(sigma)
                agg_sigma = np.array(agg_sigma)
            elif mode == 'weight':
                for layer in range(n_layer):
                    for i in range(len(words)):
                        sigma = get_sigma(text[0], layer, select_n = i, to_prediction = False, to_weight = True)
                        agg_sigma.append(sigma)
                agg_sigma = np.array(agg_sigma)
            elif mode == 'word':
                for layer in range(n_layer):
                    for i in range(len(words)):
                        sigma = get_sigma(text[0], layer + 1, select_n = i, to_prediction = False)
                        agg_sigma.append(sigma)
                agg_sigma = np.array(agg_sigma)
            elif mode == 'activation':
                x = get_embedding_from_text(text[0])
                for layer in range(n_layer + 1):
                    x = get_embedding(x, layer - 1, layer)
                    agg_sigma.append(x.cpu().numpy())
                agg_sigma = np.array(agg_sigma)
                logits_file_name = 'output/' + mode + '/' + str(text[1]) + 'logits.npy'
                logits = get_real_logits(x).cpu().numpy()
                np.save(logits_file_name, np.array(logits))
            elif mode == 'direction':
                x = get_embedding_from_text(text[0])
                agg_sigma = []
                for layer in range(n_layer):
                    x1 = get_embedding_of_layer(x, layer)
                    layer_ret = []
                    for i in range(len(words)):
                        x2 = multi_disturb_input_embed_gaussian(x1, [i])
                        x2 = get_embedding_from_layer(x2, layer)
                        logits = get_real_logits(x2)
                        layer_ret.append(logits.cpu().numpy())
                    agg_sigma.append(layer_ret)
                agg_sigma = np.array(agg_sigma)

if __name__ == '__main__':
    parser_text(["rare bird has more than enough charm to make it memorable."], 0, range(1, 13))
