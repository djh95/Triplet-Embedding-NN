import cv2
import numpy as np
import torch
import random

from enum import IntEnum


from .Define import *
from DataAugmentation import *

#'All_All_tags' No complete data set
class DataSetType (IntEnum):
    Train_81 = 0
    Test_81 = 1
    All_81 = 2
    Train_1k = 3 
    Test_1k = 4 
    All_1k = 5

class NUS_WIDE_Helper (torch.utils.data.Dataset):

    tag_dic = {}
    tag_list = []
    image_tags = []
    image_list = []

    
    def __init__(self, data_set_type, number=-1, min_tag_num=2):
        super().__init__() 

        if (data_set_type % 3) == 0:
            self.image_list_path = Image_List_Path_Train
            #self.image_number = Number_Of_Images_Train
        elif (data_set_type % 3) == 1:
            self.image_list_path = Image_List_Path_Test
            #self.image_number = Number_Of_Images_Test
        else:
            self.image_list_path = Image_List_Path_All
            #self.image_number = Number_Of_Images_All

        if data_set_type < 3:
            self.tag_list_path = Tag_List_Path_81
        else:
            self.tag_list_path = Tag_List_Path_1k

        if data_set_type == DataSetType.Train_81:
            self.image_tags_path = Image_Tags_Path_Train_81
        elif data_set_type == DataSetType.Test_81:
            self.image_tags_path = Image_Tags_Path_Test_81
        elif data_set_type == DataSetType.All_81:
            self.image_tags_path = Image_Tags_Path_All_81
        elif data_set_type == DataSetType.Train_1k:
            self.image_tags_path = Image_Tags_Path_Train_1k
        elif data_set_type == DataSetType.Test_1k:
            self.image_tags_path = Image_Tags_Path_Test_1k
        else:
            self.image_tags_path = Image_Tags_Path_All_1k

        self.image_path = Image_Path
        self.image_urls_path = Image_URLs_Path

        self.tag_list = [line.strip() for line in open(self.tag_list_path).readlines()]
        #self.tag_list = np.asarray(self.tag_list)
        
        self.tag_dic = {class_name : i for i, class_name in enumerate(self.tag_list)}

        # read data
        self.image_tags = np.loadtxt(self.image_tags_path, dtype=bool)
        self.image_tags = np.multiply(self.image_tags, 1)
        self.image_list = np.loadtxt(self.image_list_path, dtype=str)
        self.image_number = len(self.image_tags)

        self.tag_num_statistic = [0 for i in range(len(self.tag_list))]

        # remain the data with at least min_tag_num tags
        rows = []
        for i in range(self.image_number):
            tn = int(np.sum(self.image_tags[i]))
            self.tag_num_statistic[tn] = self.tag_num_statistic[tn] + 1
            if tn >= min_tag_num:
                try:
                    cv2.imread(self.get_image_path(self.image_list[i])).shape
                except:
                    continue
                rows.append(i)
        self.filter_data(rows)
        
        # remain number data
        if number == -1 or number > self.image_number or number < 3:
            number = self.image_number
        rows = random.sample(range(self.image_number), number)
        self.filter_data(rows)

        #self.image_states = np.zeros(self.image_number)
    def filter_data(self, rows):
        self.image_tags = [self.image_tags[i] for i in rows]
        self.image_list = [self.image_list[i] for i in rows]
        self.image_number = len(self.image_tags)

    def get_image_path(self, image_name):
        return (self.image_path + image_name.replace("\\","/"))

    def get_tags(self, index):
        return torch.from_numpy(self.image_tags[index])

    def get_image_name(self, index):
        return self.image_list[index]
    def get_image_names(self, indexes):
        return [self.image_list[i] for i in indexes]

    def get_tag_list(self):
        return self.tag_list

    def get_tag_num(self):
        return len(self.tag_list)

    def get_image_by_name(self, image_name):      
        image = cv2.imread(self.get_image_path(image_name))
        image = DataAugmentation(image)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = image.transpose(2,1,0)
        image = image.astype(np.float32)
        return torch.from_numpy(image)

    def get_image(self, index):
        return self.get_image_by_name(self.get_image_name(index)) / 255.0

    def get_image_path_by_index(self, index):
        return self.get_image_path(self.get_image_name(index))

    def __getitem__(self, index):
        return self.get_image(index), self.get_tags(index)

    def __len__(self):
        return len(self.image_list)

    def get_tag_num_statistic(self):
        return self.tag_num_statistic