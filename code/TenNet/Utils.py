from json import load
from math import ceil
import numpy as np
import torch
import torch.nn.functional as F
import random
import os

from Define import *
from .TenNetUtils import *

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

def get_similarity_matrix(tag_matrix):
    t = tag_matrix.float()
    return t.mm(t.t())

def get_similarity_vector(tag_v, tag_matrix):
    t1 = tag_v.view(len(tag_v),1).float()
    t2 = tag_matrix.float()
    return t2.mm(t1).squeeze()

def similarity_tags(tag1, tag2):
    return (tag1 * tag2).sum(axis=1)

# For each node in dataset, find a neighbor from dataset, such that the distance between them is less than dis. Maxmal check n*maxmal times
def get_semi_neg_sample(image_features, tag_features, similarity_matrix, IT_dist, Margindis):
    num = len(image_features)
    if num <= 1:
        print("at least 2 samples")
        return
    indexes = []
    for i in range(num):
        min_simil = 0
        n_set = []
        while len(n_set) == 0:
            n_set = [index for index in range(len(similarity_matrix[i])) if similarity_matrix[i][index] == min_simil]
            min_simil = min_simil + 1

        n_set = random.sample(n_set, len(n_set))
        candidate = -1
        min_dis = -1
        unsat_set = []
        for j in range(len(n_set)):
            index = n_set[j]
            temp_dis = F.pairwise_distance(image_features[i].view(1,-1), tag_features[index].view(1,-1))
            if torch.pow(temp_dis, 2) < Margindis[i]:
                unsat_set.append(n_set[j])
                if temp_dis > IT_dist[i]:
                    candidate = index
                    break
                elif min_dis == -1 or min_dis <  temp_dis:
                    candidate = index
                    min_dis = temp_dis
        if candidate == -1:
            if len(unsat_set) != 0:
                candidate = random.choice(unsat_set)
            else:
                candidate = random.choice(n_set)
        indexes.append(candidate)
    return indexes

def get_random_neg_sample(image_features, tag_features, similarity_matrix, IT_dist, Margindis):
    num = len(image_features)
    if num <= 1:
        print("at least 2 samples")
        return
    indexes = []
    for i in range(num):
        min_simil = 0
        n_set = []
        while len(n_set) == 0:
            n_set = [index for index in range(len(similarity_matrix[i])) if similarity_matrix[i][index] == min_simil]
            min_simil = min_simil + 1

        n_set = random.sample(n_set, len(n_set))
        candidate = -1
        for j in range(len(n_set)):
            index = n_set[j]
            temp_dis = F.pairwise_distance(image_features[i].view(1,-1), tag_features[index].view(1,-1))
            if torch.pow(temp_dis, 2) < Margindis[i]:
                candidate = index
                break
        if candidate == -1:
            candidate = random.choice(n_set)
        indexes.append(candidate)
    return indexes

# For each node in dataset, find a pos and a neg samples from dataset. Maxmal check n*maxmal times
def get_pos_indexes(data, tag_matrix):
    condition = [True] * 2 + [False]
    pos_image_indexes = []
    for tag_v in tag_matrix:
        tag_index = -1
        for j in range(len(tag_v)):
            if tag_v[j]:
                if tag_index == -1:
                    tag_index = j
                #if  random.choice(condition):
                if random.random() < max(data.tag_weight[j].item(), 0.4):
                    tag_index = j
                    break
        image_ids = data.image_ids_for_tag[tag_index]
        sampled_ids = random.sample(image_ids, min(len(image_ids),10))
        sampled_image_indexes = [data.image_ids_dic[id] for id in sampled_ids]
        sampled_image_tag_matrix = data.get_tag_matrix(sampled_image_indexes)
        sampled_image_tag_matrix = sampled_image_tag_matrix.to(device)

        similarity_vector = get_similarity_vector(tag_v, sampled_image_tag_matrix)
        image_index = sampled_image_indexes[torch.topk(similarity_vector,2)[1][1]]
        pos_image_indexes.append(image_index)

    return pos_image_indexes

def get_pos(tag_list, similarity_matrix):
    num = len(tag_list)
    if num <= 2:
        print("at least 3 samples")
        return
    pos_indexes = []
    for i in range(num):
        #p_set = torch.topk(similarity_matrix[i],5)[1]
        #p_set = [a.item() for a in p_set]
        p_set = [index for index in range(len(similarity_matrix[i])) if similarity_matrix[i][index] > 0 and index != i]
        if len(p_set) == 0:
            index = i
        else:
            index = random.sample(p_set, 1)[0]
        pos_indexes.append(index)
    return pos_indexes

def get_neg(tag_list, similarity_matrix):
    num = len(tag_list)
    if num <= 2:
        print("at least 3 samples")
        return
    neg_indexes = []
    for i in range(num):
        min_simil = 0
        n_set = []
        while len(n_set) == 0:
            n_set = [index for index in range(len(similarity_matrix[i])) if similarity_matrix[i][index] == min_simil]
            min_simil = min_simil + 1

        index = random.sample(n_set, 1)[0]
        neg_indexes.append(index)
    return neg_indexes

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

def compute_loss(data, x_images, y_tags, image_model, tag_model, triplet_loss, Lambda, Margin_Dis):

    image_features = image_model(x_images)
    tag_features = tag_model(y_tags)

    # first triplet loss, an image, cor tag, and a neg image
    anchor_image = image_features
    positive_tag = tag_features

    similarity_matrix = get_similarity_matrix(y_tags)
    IT_dist =  F.pairwise_distance(image_features, tag_features)
    
    z_tag_indexes = get_semi_neg_sample(image_features, tag_features, similarity_matrix, IT_dist, torch.pow(IT_dist, 2) + Margin_Dis)
    #z_tag_indexes = get_random_neg_sample(image_features, tag_features, similarity_matrix, IT_dist, torch.pow(IT_dist, 2) + Margin_Dis)
    
    negative_tag = torch.cat([tag_features[i].view(1,-1) for i in z_tag_indexes])

    if global_sample[0]:
        z_image_indexes_pos = get_pos_indexes(data, y_tags)
        z_images = data.get_images(z_image_indexes_pos).to(device)
        positive_image =  image_model(z_images)
    else:
        z_images_pos = get_pos(y_tags, similarity_matrix)
        positive_image = torch.cat([image_features[i].view(1,-1) for i in z_images_pos])

    z_images_neg = get_neg(y_tags, similarity_matrix)
    negative_image = torch.cat([image_features[i].view(1,-1) for i in z_images_neg])

    lossIT, dist_image_tag_pos, dist_image_tag_neg, lossIT_part1 = triplet_loss(anchor_image, positive_tag, negative_tag)

    # second triplet loss, an image, a pos image, a neg image
    lossII, dist_image_image_pos, dist_image_image_neg, lossII_part1 = triplet_loss(anchor_image, positive_image, negative_image)
    
    if tag_weight[0]:
        loss = torch.mean(lossIT +  Lambda * lossII * data.get_weight(y_tags))
    else:
        loss = torch.mean(lossIT +  Lambda * lossII)

    return loss, dist_image_tag_pos, dist_image_image_pos, dist_image_tag_neg, dist_image_image_neg, lossIT_part1, lossII_part1

def single_epoch_computation(image_model, tag_model, loader, triplet_loss, Lambda, Margin_Dis, optim=None, updata=False):
    loss = 0
    IT_positive_dis = 0
    II_positive_dis = 0
    IT_negative_dis = 0
    II_negative_dis = 0
    i = 0
    tag_num = loader.dataset.tag_num
    not_0_loss_tag_number = [0] * tag_num
    number_0_IT_loss = 0
    number_0_II_loss = 0
    for (x_images,y_tags) in loader:
        
        x_images, y_tags = x_images.to(device, non_blocking=True), y_tags.to(device, non_blocking=True)    
        res = compute_loss(loader.dataset, x_images, y_tags, image_model, tag_model, triplet_loss, Lambda, Margin_Dis)

        if updata:
            optim.zero_grad()
            res[0].backward()
            optim.step()

        #print("res[0].item()",res[0].item())
        loss += res[0].item()
        #print("loss", loss)
        IT_positive_dis += res[1].item()
        II_positive_dis += res[2].item()
        IT_negative_dis += res[3].item()
        II_negative_dis += res[4].item()
        for i, loss_i in enumerate(res[5]):
            if loss_i != 0:
                indexes = [ j for j in range(tag_num) if y_tags[i][j]==1 ]
                for index in indexes:
                    not_0_loss_tag_number[index] = not_0_loss_tag_number[index] + 1
            else:
                number_0_IT_loss = number_0_IT_loss + 1

        for i, loss_i in enumerate(res[6]):
            if loss_i == 0:
                number_0_II_loss = number_0_II_loss + 1

    loss /= len(loader)
    IT_positive_dis /= len(loader)
    II_positive_dis /= len(loader)
    IT_negative_dis /= len(loader)
    II_negative_dis /= len(loader)

    print("The number of label occurrences in non-zero loss:")
    print(not_0_loss_tag_number)
    print(loader.dataset.tag_list)
    print(loader.dataset.image_number_for_tag)

    return loss, IT_positive_dis, II_positive_dis, IT_negative_dis, II_negative_dis, number_0_IT_loss, number_0_II_loss

def select_k_tags(loader, image_model, tag_model, k):
    res = []
    tag_features = compute_single_tag_feature(tag_model, len(loader.dataset.tag_list))

    for (x_images,y_tags) in loader:  
        x_images = x_images.to(device, non_blocking=True)   
        image_features = image_model(x_images)

        for feature in image_features:
            dis = (torch.square(tag_features - feature)).sum(dim=1)
            res_i = torch.topk(dis, k, largest=False)[1]
            res.append(res_i)
    return res

def select_k_tags_one_by_one(data, image_model, tag_model, k, rows=None):
    res = []

    def firstV(d):
        return d[0]

    tag_features = compute_single_tag_feature(tag_model, len(data.tag_list))

    if rows == None:
        rows = range(data.image_number)

    for i in rows:
        temp = []
        image_i = data.get_image(i)
        image_i = image_i.unsqueeze(0)
        image_i = image_i.to(device)
        image_features_i = image_model(image_i)
        
        dis = (torch.square(tag_features - image_features_i)).sum(dim=1)
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
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    c = (tp+tn) / (tp+tn+fp+fn)
    return precision, recall, f1, c, tp, sum_p, sum_g

def write_loss_log(writer, res, name, index):
    #loss, IT_positive_dis, II_positive_dis, IT_negative_dis, II_negative_dis
    writer.add_scalar('Loss/' + name, res[0], index)
    writer.add_scalar('0_IT_Loss/' + name, res[5], index)
    writer.add_scalar('0_II_Loss/' + name, res[6], index)

def write_evalue_log(writer, res, name, index):
    # precision, recall, F1, accuracy, tp, sum_p, sum_g
    writer.add_scalar('precision/' + name, res[0], index)
    writer.add_scalar('recall/'  + name, res[1], index)
    writer.add_scalar('F1/' + name, res[2], index)
    writer.add_scalar('accuracy/' + name, res[3], index)
    

