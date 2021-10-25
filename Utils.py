from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
import random
import os
import json 

from Define import *

def log_print(string, log_path = './log.txt'):
    print(string)
    
    f = open(log_path, 'a+')
    f.write(string + '\n')
    f.close()

def read_txt(path):
    data = []
    for line in open(path):
        data.append(line.strip())
    return data
    
def one_hot(indexes, n):
    vector = np.zeros(n, dtype = np.float32)
    vector[indexes] = 1.
    return vector

def my_one_hot(tag):
    n = len(tag)
    vector = np.zeros((n, n), dtype = np.float32)
    for i in range(n):
        vector[i,i] = tag[i]
    return vector

# For each node in dataset, find a neighbor from dataset, such that the distance between them is less than dis. Maxmal check n*maxmal times
def get_one_neighbor(dataset, similarity_matrix, dis, maxmal=0.4):
    num = len(dataset)
    if num <= 1:
        print("at least 2 samples")
        return
    maxmal = min(max(2, ceil(maxmal * num)), num)
    indexes = []
    for i in range(num):
        candidate = -1
        min_dis = -1
        min_similarity = 3
        for j in range(maxmal):
            index = random.randint(1,num-1)
            if index == i:
                index = 0
            temp_dis = F.pairwise_distance(dataset[i].view(1,-1), dataset[index].view(1,-1))
            if  similarity_matrix[i][index] == 0 and temp_dis < dis[i]:
                candidate = index
                break
            if similarity_matrix[i][index] < min_similarity and (min_dis == -1 or min_dis >  temp_dis):
                min_similarity = similarity_matrix[i][index]
                candidate = index
                min_dis = temp_dis
        indexes.append(candidate)
    return indexes

def get_similarity_matrix(tag_list):
    m = torch.zeros((tag_list.shape[0],tag_list.shape[0])).to(device)
    for i in range(tag_list.shape[0]):
        for j in range(i,tag_list.shape[0]):
            m[i][j] = similarity_tags(tag_list[i], tag_list[j])
            m[j][i] = m[i][j]
    return m

# For each node in dataset, find a pos and a neg samples from dataset. Maxmal check n*maxmal times
def get_pos_neg(tag_list, similarity_matrix, maxmal=0.4):
    num = len(tag_list)
    if num <= 2:
        print("at least 3 samples")
        return
    maxmal = min(max(3, ceil(maxmal * num)), num)
    pos_indexes = []
    neg_indexes = []
    for i in range(num):
        max_index = -1
        max_similarity = -1 
        min_index = -1
        min_similarity = -1
        for j in range(maxmal):
            index = random.randint(1,num-1)
            if index <= i:
                index = i-1
            similarity = similarity_matrix[i][index]
            if min_similarity == -1 or min_similarity > similarity:
                min_similarity = similarity
                min_index = index
            if max_similarity == -1 or max_similarity <  similarity:
                max_similarity = similarity
                max_index = index
        pos_indexes.append(max_index)
        neg_indexes.append(min_index)
    return pos_indexes, neg_indexes

#compare two tag sets 
def Precision_Recall_Tags(pred_tags, gt_tags):
    if len(pred_tags) == 0 and len(gt_tags) == 0:
        return 100, 100, 100
    elif len(pred_tags) == 0 or len(gt_tags) == 0:
        return 0, 0, 0
    
    pred_tags = np.asarray(pred_tags)
    gt_tags = np.asarray(gt_tags)

    precision = pred_tags[:, np.newaxis] == gt_tags[np.newaxis, :]
    recall = gt_tags[:, np.newaxis] == pred_tags[np.newaxis, :]
    
    precision = np.sum(precision) / len(precision) * 100
    recall = np.sum(recall) / len(recall) * 100
    mAP = (precision + recall) / 2

    return precision, recall, mAP

def similarity_tags(tag1, tag2):
    return len(torch.nonzero(tag1 * tag2))

def get_tag_from_prediction(predictions, threshold=0.5):
    tags = []
    for i in range(predictions.shape[0]):
        temp_tag = []
        for j in range(predictions.shape[1]):
            if predictions[i][j] > threshold:
                temp_tag.append(1)
            else:
                temp_tag.append(0)
        tags.append(temp_tag)
    return torch.tensor(tags).to(device)

# word_vectors: batch * tag_num of the image * word dimensionalities
# return: batch * tag_size * word dim
def extend(word_vectors, indexes, num):
    matrix = np.zeros((word_vectors.shape[0], num, word_vectors.shape[1]))
    for i in range(indexes.shape[0]):
        for j in range(word_vectors.shape[1]):
            matrix[i][j] = word_vectors[i][j]
    return matrix

def get_tag_indexes(tag):
    indexes = [i for i in range(len(tag)) if tag[i] == 1]
    return np.asarray(indexes).astype(np.int32)

def get_multy_tag_indexes(tags):
    indexes = []
    for i in range(tags.shape[0]):
        indexes.append([j for j in range(tags.shape[1]) if tags[i][j] == 1])
    return indexes

def get_word_vector_matrix(vocabulary_list, dimensions):
    
    matrix_path = Processed_Word_Matrix_Path + str(dimensions) + ".txt"

    if os.path.isfile(matrix_path):
        matrix = [line.strip().split() for line in open(matrix_path).readlines()]
        matrix = [list( map(float,i) ) for i in matrix]

    else:
        full_matrix_path = Word_Vector_Path + str(dimensions) + "d.txt"
        full_matrix = [line.strip().split() for line in open(full_matrix_path,'rb').readlines()]
        m = [full_matrix[i] for i in range(len(full_matrix)) if full_matrix[i][0].decode("utf-8") in vocabulary_list]

        matrix = [[] for i in range(len(vocabulary_list))]
        for i in range(len(vocabulary_list)):
            pos = vocabulary_list.index(m[i][0].decode("utf-8"))
            matrix[pos] = [float(_.decode("utf-8")) for _ in m[i][1:]]

        with open(matrix_path, 'w') as f:
            for i in range(len(matrix)):
                for value in matrix[i]:
                    f.write(str(value) + " ")
                f.write("\n")

    return matrix

def minmaxscaler(data_list):
    for i in data_list.shape[0]:
        dmin = torch.min(data_list[i])
        dmax = torch.max(data_list[i])
        data_list[i] = (data_list[i] - dmin)/(dmax-dmin)
    return data_list
