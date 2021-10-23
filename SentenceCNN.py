# https://github.com/galsang/CNN-sentence-classification-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from Define import *


class SCNN(nn.Module):
    def __init__(self, vocabulary_size, word_dimensionality, word_vector_matrix=0, dropout_probability = 0.2):
        super(SCNN, self).__init__()

        self.CLASS_SIZE = Feature_Dimensions
        self.IN_CHANNEL = 1
        self.WV_MATRIX = word_vector_matrix
        self.VOCAB_SIZE = vocabulary_size
        self.MAX_SENT_LEN = vocabulary_size
        self.WORD_DIM = word_dimensionality
        self.DROPOUT_PROB = dropout_probability
        self.FILTERS = [3,4,5]
        self.FILTER_NUM = [100, 100, 100]

        print(self.WORD_DIM, self.VOCAB_SIZE)
        # one for UNK and one for zero padding
        self.embedding_unstatic = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        self.embedding_static = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if word_vector_matrix != 0:
            self.embedding_static.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            self.embedding_static.weight.requires_grad = False
            self.IN_CHANNEL = 2               

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM, stride=self.WORD_DIM)
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inp):
        x = self.embedding_unstatic(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)

        if self.IN_CHANNEL == 2:
            x2 = self.embedding_static(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)

        return x