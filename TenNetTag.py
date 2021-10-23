import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Define import *
from NUS_WIDE_Helper import * 
from Inception import *
from Utils import *


class TenNet_Tag(nn.Module): # input batchSize * 1 * tagNum * tagNum
    def __init__(self, vocabulary_list, dropout_probability=0.2):
        super().__init__()
        
        self.Feature_Dimensions = Feature_Dimensions
        self.VOCAB_SIZE = len(vocabulary_list)
        self.WORD_DIM = Word_Dimensionality 
        self.DROPOUT_PROB = dropout_probability
        self.IN_CHANNEL = 1
        self.WV_MATRIX = get_word_vector_matrix(vocabulary_list, self.WORD_DIM)

        # one for UNK and one for zero padding
        self.embedding_unstatic = nn.Embedding(self.VOCAB_SIZE, self.WORD_DIM)
        self.embedding_static = nn.Embedding(self.VOCAB_SIZE, self.WORD_DIM)
        if len(self.WV_MATRIX) == self.VOCAB_SIZE:
            self.embedding_static.weight.data.copy_(torch.from_numpy(np.asarray(self.WV_MATRIX)))
            self.embedding_static.weight.requires_grad = False
            self.IN_CHANNEL = 2               

 
        self.feature = nn.Sequential( 
            Inception(self.IN_CHANNEL,4,(4,8),(2,4),4), 
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            Inception(20,8,(8,16),(2,4),4), 
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            Inception(32,8,(8,16),(8,16),8), 
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            Inception(48,32,(8,32),(8,32),32), 
            GlobalAvgPool2d() 
        ) 

        self.fc = nn.Sequential( 
            nn.Linear(in_features=128, out_features=512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(in_features=512, out_features=1028, bias=True),
            nn.BatchNorm1d(1028),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(in_features=1028, out_features=self.Feature_Dimensions, bias=True)
        )

    def forward(self, tags):
        indexes = get_multy_tag_indexes(tags)
        x = torch.zeros((tags.shape[0],1,self.VOCAB_SIZE, self.WORD_DIM)).to(device)
        for i in range(len(indexes)):
            index_i =  torch.from_numpy(np.asarray(indexes[i]).astype(np.int32))
            for j in range(index_i.shape[0]):
                x[i][0][index_i[j]] = self.embedding_unstatic.weight[index_i[j]]

        if self.IN_CHANNEL == 2:
            x2 = torch.zeros((tags.shape[0],1,self.VOCAB_SIZE, self.WORD_DIM)).to(device)
            for i in range(len(indexes)):
                index_i =  torch.from_numpy(np.asarray(indexes[i]).astype(np.int32))
                for j in range(index_i.shape[0]):
                    x2[i][0][index_i[j]] = self.embedding_unstatic.weight[index_i[j]]
            #x2 = self.embedding_static(indexes)
            #x2 = extend(x2, self.Feature_Dimensions).view(-1, 1, self.VOCAB_SIZE, self.WORD_DIM)
            x = torch.cat((x, x2), 1)

        out = self.feature(x)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return minmaxscaler(out)