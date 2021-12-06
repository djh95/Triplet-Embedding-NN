import torch
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
from jupyterplot import ProgressPlot
from IPython.display import *
import IPython.display

from Define import *
from .Visualization import *
from .Utils import *


def save_TenNet(image_model, tag_model, optim, epoch, loss, name):
    torch.save({
            'epoch': epoch,
            'image_model_state_dict': image_model.state_dict(),
            'tag_model_state_dict': tag_model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'loss': loss,
            }, name)

def getTenModel(tag_model, image_model, name):
    try:
        checkpoint = torch.load(name)
        image_model.load_state_dict(checkpoint['image_model_state_dict'])   
        tag_model.load_state_dict(checkpoint['tag_model_state_dict'])        # 从字典中依次读取
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("loss: ", loss)
        print("epoch: ",epoch)
        print("Load last checkpoint data")
    except FileNotFoundError:
        print("Can\'t found " + name)

def run(writer, image_model, tag_model, train_loader, valid_loader, train_loader_for_evalue, test_loader, triplet_loss, lr, gamma, Margin_Dis, n_epochs, Lambda, print_log=True):
    print("global_sample:", global_sample[0], "tag_weight:", tag_weight[0])
    name = "../SavedModelState/IT_model_" + str(Margin_Dis) +".ckpt"
    ten_res = []
    pbar = tqdm(range(n_epochs))
    lr = lr
    Lambda = Lambda
    min_valid_loss = -1
    optimizer = torch.optim.RMSprop([{'params' : image_model.parameters()}, {'params' : tag_model.parameters()}], lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma, last_epoch=-1)
    for e in pbar:
        loss_dis_train = train(image_model, tag_model, train_loader, triplet_loss, Lambda, optimizer, Margin_Dis)
        loss_dis_valid = validate(image_model, tag_model, valid_loader, triplet_loss, Lambda, optimizer, Margin_Dis, e, min_valid_loss, True, name=name) 
        scheduler.step()

        write_loss_log(writer, loss_dis_train, "train", e)
        write_loss_log(writer, loss_dis_valid, "valid", e)

        if print_log:
            print(f"epoch:{e}:")
            output_loss_dis(f" 1-train dataset train model", loss_dis_train) 
            output_loss_dis(f" 2-valid dataset evalue model", loss_dis_valid)
        min_valid_loss = loss_dis_valid[5]

        #write_evalue_log(writer, evalue(test_loader, image_model, tag_model, k=3),  "test", e)
        evalu_both(train_loader_for_evalue, test_loader, image_model, tag_model, writer, e)
        print("")

        ten_res.append([list(loss_dis_train),list(loss_dis_valid)])

    return ten_res

def train(image_model, tag_model, loader, triplet_loss, Lambda, optim, Margin_Dis, updata=True):

    image_model.train()
    tag_model.train()

    res = single_epoch_computation(image_model, tag_model, loader, triplet_loss, Lambda, Margin_Dis, optim, updata=True)

    return res

def validate(image_model, tag_model, loader, triplet_loss, Lambda, optim, Margin_Dis, epoch, min_loss, save_best=True, name="../SavedModelState/IT_model_.ckpt"):
    
    image_model.eval()
    tag_model.eval()

    with torch.no_grad():
        res = single_epoch_computation(image_model, tag_model, loader, triplet_loss, Lambda, Margin_Dis, optim, updata=False)

    if save_best and (min_loss == -1 or min_loss > res[0]):
        min_loss = res[0]
        save_TenNet(image_model, tag_model, optim, epoch, loss=min_loss, name=name)
    return res + (min_loss,)

def test(image_model, tag_model, loader, triplet_loss, Lambda, Margin_Dis):
    image_model.eval()
    tag_model.eval()

    with torch.no_grad():
        res = single_epoch_computation(image_model, tag_model, loader, triplet_loss, Lambda, Margin_Dis, updata=False)
        output_loss_dis("test result:", res)
    return res

def evalue(loader, image_model, tag_model, k=3):
    data = loader.dataset
    image_model.eval()
    tag_model.eval()

    precision = 0
    recall = 0
    F1 = 0
    accuracy = 0
    tp = 0
    sum_p = k
    sum_g = 0

    k_tags = select_k_tags(loader, image_model, tag_model, k)
    tag_matrix = get_tag_vectors(k_tags, len(data.tag_list))

    tp_list = similarity_tags(tag_matrix, data.image_tags)
    tp = tp_list.mean()

    for i in range(data.image_number):

        res = evalue_single(data.image_tags[i], tp_list[i], k)

        precision = precision + res[0]
        recall = recall + res[1]
        F1 = F1 + res[2]
        accuracy = accuracy + res[3]
        sum_g = sum_g + res[6]

    num = data.image_number
    precision = precision / num
    recall = recall / num
    F1 = F1 / num
    accuracy = accuracy / num
    sum_g = sum_g / num
    print( "Evaluate result:")
    print(  f"Precision: {precision:.4f},  " +
            f"Recall: {recall:.4f},  " +
            f"F1: {F1:.4f},  " + 
            f"Accuracy: {accuracy:.4f}." )
    print( "Average number of tags")
    print(  f"True positive: {tp:.4f},  " +
            f"Pos. tags prediction: {sum_p:.4f},  " +
            f"Pos. tags ground truth (<=3): {sum_g:.4f}." )
    return precision, recall, F1, accuracy, tp, sum_p, sum_g

def evalu_both(train_loader, test_loader, image_model, tag_model, writer, index):
    write_evalue_log(writer, evalue(train_loader, image_model, tag_model, k=3), "train", index)
    write_evalue_log(writer, evalue(test_loader, image_model, tag_model, k=3),  "test", index)
