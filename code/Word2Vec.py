from gensim.models import Word2Vec
import os
from Define import *

def compute_sentence_list(tags_list):
    sl = []
    for i in range(len(tags_list)):
        s = []
        for j in range(len(tags_list[0])):
            if tags_list[i][j]:
                s.append(str(j))
        sl.append(s)
    return sl

def compute_word_matrix(tags_list):
    if os.path.isfile(Word2Vec_Model_Path):
        model = Word2Vec.load(Word2Vec_Model_Path)
    else:
        sentence_list = compute_sentence_list(tags_list)
        model = Word2Vec(sentences=sentence_list, vector_size=Word_Dimensions, window=5, min_count=1, workers=4)
        model.save(Word2Vec_Model_Path)
    res = []
    for i in range(len(model.wv)):
        res.append(model.wv[str(i)].tolist())

    return res