import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

    
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNEL = 3

Tag_List_Path = '../coco/tag_list/'

Image_Path = '../coco/images/'
annotation_Path = '../coco/annotations/'

Number_Of_Images_All = 269648
Number_Of_Images_Train = 161789
Number_Of_Images_Test = 107859
Number_Of_Images_Valid = 10000




