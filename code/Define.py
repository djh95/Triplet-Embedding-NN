import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNEL = 3

WEIGHT_DECAY = 5e-5

global_sample = [False]
tag_weight = [False]


Feature_Dimensions = 1000
#25 50 100 200
Word_Dimensions = 256

N_Epochs_Decoder = 20

Threshold = 0.5

Word_Vector_Path = '../glove.twitter.27B/glove.twitter.27B.'
Word2Vec_Model_Path = '../SavedModelState/Word2Vec_'
Processed_Word_Matrix_Path = '../glove.twitter.27B/WordMatrix'




