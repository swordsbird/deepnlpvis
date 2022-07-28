# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Toolkit that enables you to explain every hidden state in your model

import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt

def calculate_regularization(sampled_x, Phi, reduced_axes=None, device=None):
    """ Calculate the variance that is used for Interpreter

    Args:
        sample_x (list of torch.FloatTensor):
            A list of sampled input embeddings $x$, each $x$ is of shape ``[length, dimension]``. All the $x$s can have different length,
            but should have the same dimension. Sampled number should be higher to get a good estimation.
        Phi (function):
            The $Phi$ we studied. A function whose input is x (element in the first parameter) and returns a hidden state (of type
            ``torch.FloatTensor``, of any shape)
        reduced_axes (list of ints, Optional):
            The axes that is variable in Phi (e.g., the sentence length axis). We will reduce these axes by mean along them.

    Returns:
        torch.FloatTensor: The regularization term calculated

    """
    sample_num = len(sampled_x)
    sample_s = []
    for n in range(sample_num):
        x = sampled_x[n]
        if device is not None:
            x = x.to(device)
        s = Phi(x)
        if reduced_axes is not None:
            for axis in reduced_axes:
                assert axis < len(s.shape)
                s = s.mean(dim=axis, keepdim=True)
        sample_s.append(s.tolist())
    sample_s = np.array(sample_s)
    return np.std(sample_s, axis=0)

class Interpreter(nn.Module):
    """ Interpreter for interpret one instance.

    It will minimize the loss in Eqn.(7):

        $L(sigma) = (||Phi(embed + epsilon) - Phi(embed)||_2^2) // (regularization^2) - rate * log(sigma)$

    In our implementation, we use reparameterization trick to represent epsilon ~ N(0, sigma^2 I), i.e. epsilon = scale * ratio * noise.
    Where noise ~ N(0, 1), scale is a hyper-parameter that controls the maximum value of sigma^2, and ratio in (0, 1) is the learnable parameter.

    """
    def __init__(self, x, Phi, scale=0.5, rate=0.1, dim=None, n=32, init_ratio = None, regularization=None, words=None):
        """ Initialize an interpreter class.

        Args:
            x (torch.FloatTensor): Of shape ``[length, dimension]``.
                The $x$ we studied. i.e. The input word embeddings.
            Phi (function):
                The $Phi$ we studied. A function whose input is x (the first parameter) and returns a hidden state (of type ``torch.FloatTensor``, of any shape)
            scale (float):
                The maximum size of sigma. A hyper-parameter in reparameterization trick. The recommended value is 10 * Std[word_embedding_weight],
                where word_embedding_weight is the word embedding weight in the model interpreted. Larger scale will give more salient result, Default: 0.5.
            rate (float):
                A hyper-parameter that balance the MLE Loss and Maximum Entropy Loss. Larger rate will result in larger information loss. Default: 0.1.
            regularization (Torch.FloatTensor or np.ndarray):
                The regularization term, should be of the same shape as (or broadcastable to) the output of Phi. If None is given, method will use the output to
                regularize itself. Default: None.
            words (List[Str]):
                The input sentence, used for visualizing. If None is given, method will not show the words.

        """
        super(Interpreter, self).__init__()

        self.n = n
        self.s = x.size(0)
        self.d = x.size(1)
        if init_ratio is None:
            self.ratio = nn.Parameter(torch.randn(1, self.s, 1), requires_grad=True)
        else:
            init_ratio = torch.from_numpy(init_ratio)
            self.ratio = nn.Parameter(init_ratio.squeeze().unsqueeze(1).unsqueeze(0), requires_grad=True)

        self.scale = scale
        self.rate = rate
        self.x = x
        self.dim = dim
        self.Phi = Phi
        x1 = x if len(x.shape) == 3 else x.reshape((1, self.s, self.d))
        self.phix = Phi(x1)
        self.phix_norm = torch.mean(self.phix, axis=(0)) ** 2
        self.device = x.device

        self.regular = None
        self.regular_i = None
        if regularization is not None:
            if type(regularization) is tuple:
                self.regular = regularization[0]
                self.regular_i = regularization[1]
            else:
                self.regular = regularization

        if self.regular is not None:
            self.regular = nn.Parameter(torch.tensor(self.regular).to(x), requires_grad=False)
        if self.regular_i is not None:
            self.regular_i = nn.Parameter(torch.tensor(self.regular_i).to(x), requires_grad=False)

        self.words = words
        if self.words is not None:
            assert self.s == len(words), 'the length of x should be of the same with the lengh of words'

    def forward(self):
        """ Calculate loss:

            $L(sigma) = (||Phi(embed + epsilon) - Phi(embed)||_2^2) // (regularization^2) - rate * log(sigma)$

        Returns:
            torch.FloatTensor: a scalar, the target loss.

        """
        ratios = torch.sigmoid(self.ratio)  # 1 * S * 1
        #print('self.x', self.x.shape)
        #print('ratios', ratios.shape)
        #print(self.n, self.s, self.d)
        x_tilde = self.x + ratios * torch.randn(self.n, self.s, self.d).to(self.device) * self.scale  # N * S * D
        s = self.phix  # N * D or N * S * D
        s_tilde = self.Phi(x_tilde)
        #print(x_tilde.shape, s_tilde.shape, s.shape, self.dim)
        if self.dim == None:
            loss = (s_tilde - s) ** 2
        else:
            loss = (s_tilde[:,self.dim,:] - s[self.dim,:]) ** 2
        if self.regular is not None:
            if self.dim == None and len(loss.shape) > 2:
                loss = torch.mean(loss / self.regular ** 2, axis=(1,2))
            elif len(loss.shape) > 1:
                loss = torch.mean(loss / self.regular ** 2, axis=(1))
            else:
                loss = loss / self.regular ** 2
        else:
            if self.dim == None and len(loss.shape) > 2:
                loss = torch.mean(loss, axis=(1,2))
            elif len(loss.shape) > 1:
                loss = torch.mean(loss, axis=(1))
            else:
                loss = loss
        if self.regular_i is not None:
            t = torch.mean(torch.log(ratios) / self.regular_i ** 2) * self.rate
        else:
            t = torch.mean(torch.log(ratios)) * self.rate
        # print('loss ', loss, t)
        return loss - t

    def optimize(self, iteration=500, lr=0.05, show_progress=False):
        """ Optimize the loss function

        Args:
            iteration (int): Total optimizing iteration
            lr (float): Learning rate
            show_progress (bool): Whether to show the learn progress

        """
        minLoss = None
        state_dict = None
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        func = (lambda x: x) if not show_progress else tqdm
        for _ in func(range(iteration)):
            optimizer.zero_grad()
            loss = self()
            avgloss = loss[0] / len(loss)
            for i in range(1, len(loss)):
                avgloss += loss[i] / len(loss)
            min_loss = (loss[loss.argmin()])
            avgloss.backward()
            optimizer.step()
            if minLoss is None or minLoss > min_loss:
                state_dict = {k:self.state_dict()[k] + 0. for k in self.state_dict().keys()}
                minLoss = min_loss
        #print(minLoss.item())
        self.eval()
        #self.load_state_dict(state_dict)

    def get_sigma(self):
        """ Calculate and return the sigma

        Returns:
            np.ndarray: of shape ``[seqLen]``, the ``sigma``.

        """
        ratios = torch.sigmoid(self.ratio[0])  # S * 1
        return ratios.detach().cpu().numpy()[:,0] * self.scale

    def get_ratio(self):
        return self.ratio.detach().cpu().numpy()

    def visualize(self):
        """ Visualize the information loss of every word.
        """
        sigma_ = self.get_sigma()
        _, ax = plt.subplots()
        ax.imshow([sigma_], cmap='GnBu_r')
        ax.set_xticks(range(self.s))
        #ax.set_xticklabels(self.words)
        ax.set_yticks([0])
        ax.set_yticklabels([''])
        plt.tight_layout()
        plt.show()
