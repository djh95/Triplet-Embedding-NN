import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNEL = 3

Image_List_Path_Train = '../NUS_WID/ImageList/TrainImageList.txt'
Image_List_Path_Test  = '../NUS_WID/ImageList/TestImageList.txt'
Image_List_Path_All   = '../NUS_WID/ImageList/ImageList.txt'

Image_Tags_Path_Train_1k = '../NUS_WID/NUS_WID_Tags/TrainTags1k.txt'
Image_Tags_Path_Test_1k  = '../NUS_WID/NUS_WID_Tags/TestTags1k.txt'
Image_Tags_Path_All_1k   = '../NUS_WID/NUS_WID_Tags/AllTags1k.txt'

Image_Tags_Path_Train_81 = '../NUS_WID/NUS_WID_Tags/TrainTags81.txt'
Image_Tags_Path_Test_81  = '../NUS_WID/NUS_WID_Tags/TestTags81.txt'
Image_Tags_Path_All_81   = '../NUS_WID/NUS_WID_Tags/AllTags81.txt'

Tag_List_Path_81 = '../NUS_WID/ConceptsList/TagList81.txt'
Tag_List_Path_1k = '../NUS_WID/ConceptsList/TagList1k.txt'
Tag_List_Path_All = '../NUS_WID/ConceptsList/TagListAll.txt'

Image_Path = '../NUS_WID/Flickr/'
Image_URLs_Path = '../NUS_WID/NUS-WIDE-urls.txt'

Number_Of_Images_All = 269648
Number_Of_Images_Train = 161789
Number_Of_Images_Test = 107859
Number_Of_Images_Valid = 10000

Word_Vector_Path = '../glove.twitter.27B/glove.twitter.27B.'
Word2Vec_Model_Path = '../SavedModelState/Word2Vec_'
Processed_Word_Matrix_Path = '../glove.twitter.27B/WordMatrix'




