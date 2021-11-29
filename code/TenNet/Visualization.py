from tqdm.notebook import tqdm
from jupyterplot import ProgressPlot
from IPython.display import *

from .Utils import *



def output_loss_dis(s, loss_dis):
    print(  s + "\n" +
            f"loss: {loss_dis[0]:.6f},  " +
            f"IT_pos_dis: {loss_dis[1]:.4f},  " +
            f"IT_neg_dis: {loss_dis[3]:.4f},  " + 
            f"II_pos_dis: {loss_dis[2]:.4f},  " + 
            f"II_neg_dis: {loss_dis[4]:.4f}." )

def printLossLog(res):

    pbar = tqdm(range(len(res)))
    for e in pbar:
    
        dis = res[e][0]
        output_loss_dis(f"epoch:{e}: 1-train dataset with train model", dis)
        
        loss_dis_valid = res[e][1]   
        output_loss_dis(f"epoch:{e}: 2-valid dataset with evalue model", loss_dis_valid)

def printLossProgressPlot(res, n_epochs):

    max_v = compute_column_maximum(res)[0]

    pp = ProgressPlot(plot_names=["loss"],
                    line_names=["train", "valid"],
                    x_lim=[0, n_epochs-1], 
                    y_lim=[0, max_v])

    pbar = tqdm(range(min(len(res), n_epochs)))
    for e in pbar:
    
        train_loss = res[e][0]
        valid_loss = res[e][1]
    
        pp.update([[train_loss[0], valid_loss[0]]])

    pp.finalize()

def printDistanceProgressPlot(res, n_epochs, train=True):

    max_v = max(compute_column_maximum(res)[1:])

    if train:
        names = "train distance"
    else:
        names = "valid distance"

    pp = ProgressPlot(plot_names=[names],
                  line_names=["pos_IT", "neg_IT", "pos_II", "neg_II"],
                  x_lim=[0, n_epochs-1], 
                  y_lim=[0, max_v])

    pbar = tqdm(range(min(len(res), n_epochs)))
    for e in pbar:
        
        if train:
            dis = res[e][0]
        else:
            dis = res[e][1]
    
        pp.update([[dis[1], dis[3], dis[2], dis[4]]])

    pp.finalize()

def print_tags(data, tag_v):
    tags = [data.tag_list[i] for i in range(len(tag_v)) if tag_v[i] ]
    print(tags)

def predict(loader, image_model, tag_model, k=3, number=10):
    data = loader.dataset
    image_model.eval()
    tag_model.eval()

    rows = random.sample(range(data.image_number), number)
    k_tags = select_k_tags(loader, image_model, tag_model, k, rows)
    tag_matrix = get_tag_vectors(k_tags, len(data.tag_list))

    for index, r in enumerate(rows):

        print("Prediction:")
        print_tags(data, tag_matrix[index])
        print("Ground Truth:")
        print_tags(data, data.image_tags[r])

        tp = similarity_tags(tag_matrix[index], data.image_tags[r])
        print("True Positive:")
        print(tp)

        #display(Image(data.get_image_path(data.get_image_name(r))))
        display(Image(data.get_image_path_by_index(r)))

def show_data(data, number=10):

    rows = random.sample(range(data.image_number), number)

    for r in rows:
        print("Ground Truth:")
        print_tags(data, data.image_tags[r])
        display(Image(data.get_image_path_by_index(r)))

def show_data_with_indexes(data, rows):

    for r in rows:
        print("Ground Truth:")
        print_tags(data, data.image_tags[r])
        display(Image(data.get_image_path_by_index(r)))



