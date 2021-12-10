from numpy import sqrt
from pycocotools.coco import COCO
import json
import torch
from torch._C import device
from .Define import *
from DataAugmentation import *
from enum import IntEnum

class DataSetType (IntEnum):
    Train17 = 0
    Test17 = 1
    Valid17 = 2

class coco_Helper (torch.utils.data.Dataset):

    def __init__(self, data_set_type, min_tag_num=1):
        super().__init__() 

        if data_set_type == DataSetType.Train17:
            self.dataType='train2017'
        elif data_set_type == DataSetType.Valid17:
            self.dataType='val2017'
        elif data_set_type == DataSetType.Test17:
            self.dataType='test2017'
        else:
            print("coco type is wrong")

        annFile = annotation_Path + 'instances_{}.json'.format(self.dataType)
        self.coco=COCO(annFile)

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.tag_list = [cat['name'] for cat in cats]
        self.tag_id_list = [cat['id'] for cat in cats]
        self.tag_num = len(self.tag_list)

        self.image_ids_list = self.coco.getImgIds()
        self.image_ids_list.sort()
        self.image_number = len(self.image_ids_list)
        self.image_ids_dic = dict(zip(self.image_ids_list, range(0,self.image_number)))

        #Sort the tags by frequency, from low to high
        '''
        self.image_number_for_tag = []
        for i, catId in enumerate(self.tag_id_list):
            imgIds = self.coco.getImgIds(catIds=catId)
            self.image_number_for_tag.append(len(imgIds))
        tag_order = torch.topk(torch.from_numpy(np.asarray(self.image_number_for_tag)) , self.tag_num, largest=False)[1]
        self.tag_list = self.reorder(self.tag_list, tag_order)
        self.tag_id_list = self.reorder(self.tag_id_list, tag_order)
        '''

        # compute tag set for each image
        self.image_tags = [[0] * self.tag_num for i in range(self.image_number)]
        self.image_number_for_tag = []
        self.image_ids_for_tag = []
        self.is_tail = [0] * self.tag_num
        for i, catId in enumerate(self.tag_id_list):
            imgIds = self.coco.getImgIds(catIds=catId)
            self.image_number_for_tag.append(len(imgIds))
            self.image_ids_for_tag.append(imgIds)
            if self.image_number_for_tag[i] < self.image_number / 100 * 2:
                self.is_tail[i] = 1
            for id in imgIds:
                self.image_tags[self.image_ids_dic[id]][i] = 1

        self.tag_weight = 1 - np.asarray(self.image_number_for_tag) / self.image_number * 30
        self.tag_weight = [max(w, 0.1) for w in self.tag_weight]
        self.tag_weight = torch.from_numpy(np.asarray(self.tag_weight))
        self.tag_weight = self.tag_weight.to(device)

        tag_num = [sum(self.image_tags[i]) for i in range(len(self.image_tags))]
        rows = [i for i in range(len(tag_num)) if tag_num[i] >= min_tag_num]
        self.filter_data(rows)
        self.image_tags = np.asarray(self.image_tags)

        self.image_path = []

        for index in range(self.image_number):
            img = self.coco.loadImgs(self.image_ids_list[index])[0]
            self.image_path.append(Image_Path + '%s/%s'%(self.dataType,img['file_name']))

    def reorder(self, data, order):
        return [data[i] for i in order]

    def filter_data(self, rows):
        self.image_tags = [self.image_tags[i] for i in rows]
        self.image_ids_list = [self.image_ids_list[i] for i in rows]
        self.image_number = len(self.image_tags)
        self.image_ids_dic = dict(zip(self.image_ids_list, range(0,self.image_number)))

    def get_weight(self, tag_matrix):
        res = []
        for tag_v in tag_matrix:
            res.append(sum(tag_v * self.tag_weight).item())
        return torch.from_numpy(np.asarray(res)).to(device)

    def get_tag_list(self):
        return self.tag_list

    def get_image(self, index):
        return self.get_image_by_path(self.image_path[index])

    def get_image_by_path(self, image_path):
        image = cv2.imread(image_path)
        image = DataAugmentation(image)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = image.transpose(2,1,0)
        image = image.astype(np.float32)
        return torch.from_numpy(image) / 255.0

    def get_images(self, indexes):
        res = None
        for index in indexes:
            temp = self.get_image(index)
            temp = temp.view((1,) + temp.shape)
            if res == None:
                res = temp
            else:
                res = torch.cat((res, temp), 0)
        return res

    def get_tag_matrix(self, indexes):
        res = None
        for index in indexes:
            temp = self.get_tags(index)
            temp = temp.view((1,) + temp.shape)
            if res == None:
                res = temp
            else:
                res = torch.cat((res, temp), 0)
        return res

    def refresh_images(self):
        for index in range(self.image_number):
            img = self.coco.loadImgs(self.image_ids_list[index])[0]
            img_path = Image_Path + '%s/%s'%(self.dataType,img['file_name'])
            img = cv2.imread(img_path)
            scale = min(img.shape[0]/IMAGE_HEIGHT, img.shape[1]/IMAGE_WIDTH) -0.1
            print(img.shape)
            if scale < 1.05:
                continue
            img = cv2.resize(img, (int(img.shape[1] / scale), int(img.shape[0] / scale)))
            cv2.imwrite(img_path, img)
            print(img.shape)
            print(index, scale)

    def get_tags(self, index):
        return torch.from_numpy(self.image_tags[index])

    def __getitem__(self, index):
        return self.get_image(index), self.get_tags(index)

    def __len__(self):
        return self.image_number

    def get_image_path_by_index(self, index):
        return self.image_path[index]

    def get_image_path_by_name(self, name):
        return Image_Path + '%s/%s'%(self.dataType, name)

