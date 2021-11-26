from enum import IntEnum
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Define import *
from .Utils import *

class EbeddingModel (IntEnum):
    # tag -> feature ->tag
    static = 0
    # image -> feature ->tag
    unstatic = 1
    multichannel = 2


class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.max_pool2d(x, kernel_size=x.size()[2:])

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class TenNet_Tag(nn.Module): # input batchSize * 1 * tagNum * tagNum
    def __init__(self, vocabulary_list, dropout_probability=0.1, filters=[3, 4, 5], filter_num=[100, 100, 100], in_channel=2, additional_matrix=None, max_length=8):
        super().__init__()
        
        self.Feature_Dimensions = Feature_Dimensions
        self.WORD_DIM = Word_Dimensions

        self.VOCAB_SIZE = len(vocabulary_list)
        self.DROPOUT_PROB = dropout_probability
        self.IN_CHANNEL = in_channel
        self.FILTERS = filters
        self.FILTER_NUM = filter_num
        self.max_length = max_length

        self.embedding_unstatic = nn.Embedding(self.VOCAB_SIZE +1, self.WORD_DIM, padding_idx=self.VOCAB_SIZE)
        
        if self.IN_CHANNEL == 2:
            self.embedding_static = nn.Embedding(self.VOCAB_SIZE +1, self.WORD_DIM, padding_idx=self.VOCAB_SIZE)
            self.WV_MATRIX = torch.cat((torch.FloatTensor(additional_matrix), torch.zeros((1,self.WORD_DIM))), 0)
            self.embedding_static.weight.data.copy_(torch.from_numpy(np.asarray(self.WV_MATRIX)))
            self.embedding_static.weight.requires_grad = False           

        self.branches = []
        for i in range(len(self.FILTERS)):
            self.branches.append(BasicConv2d(2, self.FILTER_NUM[i], kernel_size=(self.FILTERS[i], self.WORD_DIM)).to(device))

        self.pooling = GlobalMaxPool2d()
        
        self.fc = nn.Sequential( 
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(sum(self.FILTER_NUM), out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(in_features=2048, out_features=self.Feature_Dimensions, bias=True),
            )
        

    def forward(self, tagsets):
        index_list = get_multy_tag_indexes(tagsets)
        for i in range(len(index_list)):
            index_list[i] = random.sample(index_list[i], min(len(index_list[i]), self.max_length)) 
            index_list[i] = index_list[i] + [self.VOCAB_SIZE for i in range(self.max_length-len(index_list[i]))]

        index_list = torch.LongTensor(index_list).to(device)
        
        x = self.embedding_unstatic(index_list).view(-1, 1, self.max_length, self.WORD_DIM)

        if self.IN_CHANNEL == 2:
            x2 = self.embedding_static(index_list).view(-1, 1, self.max_length, self.WORD_DIM)
            x = torch.cat((x, x2), 1)

        out = [self.pooling(conv(x)).view(tagsets.shape[0], -1)  for conv in self.branches]
        out = torch.cat(out, 1)
        out = self.fc(out)
        return F.normalize(out)

'''
class TenNet_Tag_s1(nn.Module): # input batchSize * 1 * tagNum * tagNum
    def __init__(self, vocabulary_list, shuffle=True, dropout_probability=0.1, filters=[3, 4, 5], filter_num=[100, 100, 100], in_channel=2, additional_matrix=None):
        super().__init__()
        
        self.Feature_Dimensions = Feature_Dimensions
        self.WORD_DIM = Word_Dimensions

        self.SHUFFLE = shuffle
        self.VOCAB_SIZE = len(vocabulary_list)
        self.DROPOUT_PROB = dropout_probability
        self.IN_CHANNEL = in_channel
        self.FILTERS = filters
        self.FILTER_NUM = filter_num

        self.ORDER = random.sample(range(self.VOCAB_SIZE), self.VOCAB_SIZE)

        self.embedding_unstatic = nn.Embedding(self.VOCAB_SIZE +1, self.WORD_DIM, padding_idx=self.VOCAB_SIZE)
        
        if self.IN_CHANNEL == 2:
            self.embedding_static = nn.Embedding(self.VOCAB_SIZE +1, self.WORD_DIM, padding_idx=self.VOCAB_SIZE)
            self.WV_MATRIX = torch.cat((torch.FloatTensor(additional_matrix), torch.zeros((1,self.WORD_DIM))), 0)
            self.embedding_static.weight.data.copy_(torch.from_numpy(np.asarray(self.WV_MATRIX)))
            self.embedding_static.weight.requires_grad = False           

        self.branches = []
        for i in range(len(self.FILTERS)):
            self.branches.append(BasicConv2d(2, self.FILTER_NUM[i], kernel_size=(self.FILTERS[i], self.WORD_DIM)).to(device))

        self.pooling = GlobalMaxPool2d()
        
        self.fc = nn.Sequential( 
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(sum(self.FILTER_NUM), out_features=2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(in_features=2048, out_features=2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(in_features=2048, out_features=self.Feature_Dimensions, bias=True),
            )
        

    def forward(self, tagsets):

        if self.SHUFFLE:
            index_list = []
            for ts in tagsets:
                temp = []
                for i in self.ORDER:
                    if ts[i] != 0:
                        temp.append(i)
                temp = temp + [self.VOCAB_SIZE for i in range(self.VOCAB_SIZE-len(temp))]
                index_list.append(temp)
        else:
            index_list = []
            for ts in tagsets:
                temp = []
                for i in range(len(ts)):
                    if ts[i] != 0:
                        temp.append(i)
                temp = temp + [self.VOCAB_SIZE for i in range(self.VOCAB_SIZE-len(temp))]
                index_list.append(temp)

        #print(index_list)
        index_list = torch.LongTensor(index_list).to(device)
        
        x = self.embedding_unstatic(index_list).view(-1,1,self.VOCAB_SIZE,self.WORD_DIM)
        #print(self.embedding_unstatic(index_list).shape)
        #print(x.shape)
        #print(x)

        if self.IN_CHANNEL == 2:
            x2 = self.embedding_static(index_list).view(-1,1,self.VOCAB_SIZE,self.WORD_DIM)
            x = torch.cat((x, x2), 1)    
        out = [self.pooling(conv(x)).view(tagsets.shape[0], -1)  for conv in self.branches]
        out = torch.cat(out, 1)
        out = self.fc(out)
        return out

class TenNet_Tag_s2(nn.Module): # input batchSize * 1 * tagNum * tagNum
    def __init__(self, vocabulary_list, dropout_probability=0.5, filters=[3, 4, 5], filter_num=[100, 100, 100], in_channel=2, additional_matrix=None):
        super().__init__()
        
        self.Feature_Dimensions = Feature_Dimensions
        self.WORD_DIM = Word_Dimensions

        self.VOCAB_SIZE = len(vocabulary_list)
        self.DROPOUT_PROB = dropout_probability
        self.IN_CHANNEL = in_channel
        self.FILTERS = filters
        self.FILTER_NUM = filter_num

        self.ORDER = random.sample(range(self.VOCAB_SIZE), self.VOCAB_SIZE)

        self.embedding_unstatic = nn.Embedding(self.VOCAB_SIZE +1, self.WORD_DIM, padding_idx=self.VOCAB_SIZE)

        if self.IN_CHANNEL == 2:
            self.embedding_static = nn.Embedding(self.VOCAB_SIZE +1, self.WORD_DIM, padding_idx=self.VOCAB_SIZE)
            self.WV_MATRIX = additional_matrix
            self.WV_MATRIX.append([0 for i in range(self.WORD_DIM)])
            self.embedding_static.weight.data.copy_(torch.from_numpy(np.asarray(self.WV_MATRIX)))
            self.embedding_static.weight.requires_grad = False           

        self.branches = []
        for i in range(len(self.FILTERS)):
            self.branches.append(BasicConv2d(2, self.FILTER_NUM[i], kernel_size=(self.FILTERS[i], self.WORD_DIM)).to(device))

        self.pooling = GlobalMaxPool2d()
        
        self.fc = nn.Sequential( 
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(sum(self.FILTER_NUM), out_features=self.Feature_Dimensions, bias=True)
            )
        

    def forward(self, tagsets):

        index_list = get_multy_tag_indexes(tagsets)
        for i in range(len(index_list)):
            index_list[i] = random.sample(index_list[i], len(index_list[i])) 
            index_list[i] = index_list[i] + [self.VOCAB_SIZE for i in range(self.VOCAB_SIZE-len(index_list[i]))]

        index_list = torch.LongTensor(index_list).to(device)
        
        x = self.embedding_unstatic(index_list).view(-1,1,self.VOCAB_SIZE,self.WORD_DIM)

        if self.IN_CHANNEL == 2:
            x2 = self.embedding_static(index_list).view(-1,1,self.VOCAB_SIZE,self.WORD_DIM)
            x = torch.cat((x, x2), 1)

        out = [self.pooling(conv(x)).view(tagsets.shape[0], -1)  for conv in self.branches]
        out = torch.cat(out, 1)
        out = self.fc(out)
        return out

class TenNet_Tag_s3(nn.Module): # input batchSize * 1 * tagNum * tagNum
    def __init__(self, vocabulary_list, shuffle=True, dropout_probability=0.5, filters=[3, 4, 5], filter_num=[100, 100, 100], in_channel=2, additional_matrix=None):
        super().__init__()
        
        self.Feature_Dimensions = Feature_Dimensions
        self.WORD_DIM = Word_Dimensions

        self.SHUFFLE = shuffle
        self.VOCAB_SIZE = len(vocabulary_list)
        self.DROPOUT_PROB = dropout_probability
        self.IN_CHANNEL = in_channel
        self.FILTERS = filters
        self.FILTER_NUM = filter_num

        self.ORDER = random.sample(range(self.VOCAB_SIZE), self.VOCAB_SIZE)

        self.embedding_unstatic = nn.Embedding(self.VOCAB_SIZE +1, self.WORD_DIM, padding_idx=self.VOCAB_SIZE)

        if self.IN_CHANNEL == 2:
            self.embedding_static = nn.Embedding(self.VOCAB_SIZE +1, self.WORD_DIM, padding_idx=self.VOCAB_SIZE)
            self.WV_MATRIX = additional_matrix
            self.WV_MATRIX.append([0 for i in range(self.WORD_DIM)])
            self.embedding_static.weight.data.copy_(torch.from_numpy(np.asarray(self.WV_MATRIX)))
            self.embedding_static.weight.requires_grad = False           

        self.branches = []
        for i in range(len(self.FILTERS)):
            self.branches.append(BasicConv2d(2, self.FILTER_NUM[i], kernel_size=(self.FILTERS[i], self.WORD_DIM)).to(device))

        self.pooling = GlobalMaxPool2d()
        
        self.fc = nn.Sequential( 
            nn.Dropout(p=dropout_probability, inplace=False),
            nn.Linear(sum(self.FILTER_NUM), out_features=self.Feature_Dimensions, bias=True)
            )
        

    def forward(self, tagsets):

        if self.SHUFFLE:
            index_list = []
            for ts in tagsets:
                temp = []
                for i in self.ORDER:
                    if ts[i] == 0:
                        temp.append(self.VOCAB_SIZE)
                    else:
                        temp.append(i)
                index_list.append(temp)
        else:
            index_list = []
            for ts in tagsets:
                temp = []
                for i in range(len(ts)):
                    if ts[i] == 0:
                        temp.append(self.VOCAB_SIZE)
                    else:
                        temp.append(i)
                index_list.append(temp)

        index_list = torch.LongTensor(index_list).to(device)
        
        x = self.embedding_unstatic(index_list).view(-1,1,self.VOCAB_SIZE,self.WORD_DIM)

        if self.IN_CHANNEL == 2:
            x2 = self.embedding_static(index_list).view(-1,1,self.VOCAB_SIZE,self.WORD_DIM)
            x = torch.cat((x, x2), 1)
            
        out = [self.pooling(conv(x)).view(tagsets.shape[0], -1)  for conv in self.branches]
        out = torch.cat(out, 1)
        out = self.fc(out)
        return out


'''