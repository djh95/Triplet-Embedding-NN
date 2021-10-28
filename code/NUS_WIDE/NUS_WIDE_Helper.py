import cv2
import numpy as np
import torch
import random

from enum import IntEnum


from Define import *
from .DataAugmentation import *

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
    
    def __init__(self, data_set_type, number = -1, min_tag_num = 3):
        super().__init__() 

        if (data_set_type % 3) == 0:
            self.image_list_path = Image_List_Path_Train
            #self.images_number = Number_Of_Images_Train
        elif (data_set_type % 3) == 1:
            self.image_list_path = Image_List_Path_Test
            #self.images_number = Number_Of_Images_Test
        else:
            self.image_list_path = Image_List_Path_All
            #self.images_number = Number_Of_Images_All

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
        self.image_list = np.loadtxt(self.image_list_path, dtype=str)
        self.images_number = len(self.image_tags)

        # remain the data with at least min_tag_num tags
        rows = []
        for i in range(self.images_number):
            if np.sum(self.image_tags[i]) >= min_tag_num:
                try:
                    cv2.imread(self.get_image_path(self.image_list[i])).shape
                except:
                    continue
                rows.append(i)
        self.filter_data(rows)
        
        # remain number data
        if number == -1 or number > self.images_number or number < 3:
            number = self.images_number
        rows = random.sample(range(self.images_number), int(number/BATCH_SIZE) * BATCH_SIZE)
        self.filter_data(rows)

        #self.image_states = np.zeros(self.images_number)
    def filter_data(self, rows):
        self.image_tags = [self.image_tags[i] for i in rows]
        self.image_list = [self.image_list[i] for i in rows]
        self.images_number = len(self.image_tags)
    
    def get_image(self, image_name):
        
        image = cv2.imread(self.get_image_path(image_name))
        image = DataAugmentation(image)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = image.reshape(3,IMAGE_WIDTH, IMAGE_HEIGHT)
        image = image.astype(np.float32)
        return image

    def get_image_path(self, image_name):
        return (self.image_path + image_name.replace("\\","/"))

    def get_tags(self, indexes):
        return [self.image_tags[i] for i in indexes]

    def get_image_names(self, indexes):
        return [self.image_list[i] for i in indexes]

    def get_tag(self, index):
        tag = self.image_tags[index]
        #print(tag.shape)
        return np.asarray(tag).astype(np.int32)

    def get_tag_list(self):
        return self.tag_list

    def get_tag_num(self):
        return len(self.tag_list)

    def get_image_name(self, index):
        return self.image_list[index]

    def __getitem__(self, index):
        return torch.from_numpy(self.get_image(self.get_image_name(index))) / 255.0, torch.from_numpy(self.get_tag(index))

    def __len__(self):
        return len(self.image_list)