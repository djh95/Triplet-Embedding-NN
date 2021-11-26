from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
import random
import os

from Define import *

def get_tag_vector(indexes, n):
    vector = [0] * n
    for i in indexes:
        vector[i] = 1
    return vector

def get_tag_vectors(indexes_list, n):
    res = []
    for i in range(len(indexes_list)):
        temp = get_tag_vector(indexes_list[i], n)
        res.append(temp)
    return np.asarray(res)

def get_similarity_matrix(tag_list):
    t = tag_list.float()
    return t.mm(t.t())

def similarity_tags(tag1, tag2):
    return (tag1 * tag2).sum(axis=1)

# For each node in dataset, find a neighbor from dataset, such that the distance between them is less than dis. Maxmal check n*maxmal times
def get_neg_neighbor(image_features, tag_features, similarity_matrix, IT_dist, Margindis):
    num = len(image_features)
    if num <= 1:
        print("at least 2 samples")
        return
    indexes = []
    for i in range(num):
        n_set = [index for index in range(len(similarity_matrix[i])) if similarity_matrix[i][index] == 0]
        candidate = n_set[0]
        min_dis = -1
        n_set = random.sample(n_set, len(n_set))
        for j in range(len(n_set)):
            index = n_set[j]
            temp_dis = F.pairwise_distance(image_features[i].view(1,-1), tag_features[index].view(1,-1))
            if torch.pow(temp_dis, 2) < Margindis[i]:
                if temp_dis > IT_dist[i]:
                    candidate = index
                    break
                elif min_dis == -1 or min_dis <  temp_dis:
                    candidate = index
                    min_dis = temp_dis
        indexes.append(candidate)
    return indexes

# For each node in dataset, find a pos and a neg samples from dataset. Maxmal check n*maxmal times
def get_pos_neg(tag_list, similarity_matrix):
    num = len(tag_list)
    if num <= 2:
        print("at least 3 samples")
        return
    pos_indexes = []
    neg_indexes = []
    for i in range(num):
        p_set = [index for index in range(len(similarity_matrix[i])) if similarity_matrix[i][index] != 0 and index != i]
        if len(p_set) == 0:
            max_index = i
        else:
            max_index = random.sample(p_set, 1)[0]

        n_set = [index for index in range(len(similarity_matrix[i])) if similarity_matrix[i][index] == 0]
        min_index = random.sample(n_set, 1)[0]

        pos_indexes.append(max_index)
        neg_indexes.append(min_index)
    return pos_indexes, neg_indexes

def get_tag_indexes(tag):
    indexes = [i for i in range(len(tag)) if tag[i] == 1]
    return np.asarray(indexes).astype(np.int32)

def get_multy_tag_indexes(tags):
    indexes = []
    for i in range(tags.shape[0]):
        indexes.append([j for j in range(tags.shape[1]) if tags[i][j] == 1])
    return indexes

def get_word_vector_matrix_glove(vocabulary_list, dimensions):
    
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

def compute_column_maximum(m):
    res = [i for i in m[0][0]]
    for i in range(len(m)):
        for j in range(len(m[0][0])):
            if m[i][0][j] > res[j]:
                res[j] = m[i][0][j]
            if m[i][1][j] > res[j]:
                res[j] = m[i][1][j]
    return res

def compute_loss(x_images, y_tags, image_model, tag_model, triplet_loss, Lambda, Margin_Dis):

    image_features = image_model(x_images)
    tag_features = tag_model(y_tags)

    # first triplet loss, an image, cor tag, and a neg image
    anchor_image = image_features
    positive_tag = tag_features

    similarity_matrix = get_similarity_matrix(y_tags)
    IT_dist =  F.pairwise_distance(image_features, tag_features)
    
    z_tag_indexes = get_neg_neighbor(image_features, tag_features, similarity_matrix, IT_dist, torch.pow(IT_dist, 2) + Margin_Dis)
    negative_tag = torch.cat([tag_features[i].view(1,-1) for i in z_tag_indexes])

    z_images_pos, z_images_neg = get_pos_neg(y_tags, similarity_matrix)
    positive_image = torch.cat([image_features[i].view(1,-1) for i in z_images_pos])
    negative_image = torch.cat([image_features[i].view(1,-1) for i in z_images_neg])

    lossIT, dist_image_tag_pos, dist_image_tag_neg = triplet_loss(anchor_image, positive_tag, negative_tag)

    # second triplet loss, an image, a pos image, a neg image
    lossII, dist_image_image_pos, dist_image_image_neg =triplet_loss(anchor_image, positive_image, negative_image)
    loss = lossIT +  Lambda * lossII

    return loss, dist_image_tag_pos, dist_image_image_pos, dist_image_tag_neg, dist_image_image_neg

def single_epoch_computation(image_model, tag_model, loader, triplet_loss, Lambda, Margin_Dis, optim=None, updata=False):
    loss = 0
    IT_positive_dis = 0
    II_positive_dis = 0
    IT_negative_dis = 0
    II_negative_dis = 0

    for (x_images,y_tags) in loader:
        
        x_images, y_tags = x_images.to(device, non_blocking=True), y_tags.to(device, non_blocking=True)    
        res = compute_loss(x_images, y_tags, image_model, tag_model, triplet_loss, Lambda, Margin_Dis)

        if updata:
            optim.zero_grad()
            res[0].backward()
            optim.step()

        loss += res[0].item()
        IT_positive_dis += res[1].item()
        II_positive_dis += res[2].item()
        IT_negative_dis += res[3].item()
        II_negative_dis += res[4].item()

    loss /= len(loader)
    IT_positive_dis /= len(loader)
    II_positive_dis /= len(loader)
    IT_negative_dis /= len(loader)
    II_negative_dis /= len(loader)

    return loss, IT_positive_dis, II_positive_dis, IT_negative_dis, II_negative_dis

def select_k_tags(loader, image_model, tag_model, k, rows=None):
    res = []

    def firstV(d):
        return d[0]

    tag_features = compute_single_tag_feature(tag_model, len(loader.dataset.tag_list))

    for (x_images,y_tags) in loader:  
        x_images = x_images.to(device, non_blocking=True)   
        image_features = image_model(x_images)

        for feature in image_features:
            dis = (torch.square(tag_features - feature)).sum(dim=1)
            res_i = torch.topk(dis, k, largest=False)[1]
            res.append(res_i)
    return res

def compute_single_tag_feature(tag_model, n):
    y_tags = torch.zeros((n,n)) 

    for i in range(n):
        y_tags[i][i] = 1
    
    y_tags = y_tags.to(device)   
    tag_features = tag_model(y_tags)

    return tag_features

def evalue_single(ground_truth_tag_v, tp, sum_p):
    fp = sum_p - tp
    sum_g = sum(ground_truth_tag_v)
    fn = sum_g - tp
    tn = len(ground_truth_tag_v) - sum_p - fn

    precision = tp / sum_p
    recall = tp / sum_g
    if precision== 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    c = (tp+tn) / (tp+tn+fp+fn)
    return precision, recall, f1, c, tp, sum_p, sum_g
