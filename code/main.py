import torch
import torch.nn.functional as F
import torchvision.models as models
from jupyterplot import ProgressPlot
from tqdm.notebook import tqdm
import numpy as np
import GPUtil
import cv2
import random
from torch.utils.tensorboard import SummaryWriter
import argparse

import NUS_WIDE as nus
import TenNet as TenNet
import PredictionModel as P
from TripletLossFunc import TripletLossFunc
from Word2Vec import *
import COCO as coco

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

Feature_Dimensions = 1000
Word_Dimensions = 256

Word_Vector_Path = '../glove.twitter.27B/glove.twitter.27B.'
Word2Vec_Model_Path = '../SavedModelState/Word2Vec_'
Processed_Word_Matrix_Path = '../glove.twitter.27B/WordMatrix'


def main():
    parser = argparse.ArgumentParser(description="-----[TenNet]-----")
    parser.add_argument("--job_num", type=str, required=True, help="the job number on HPC")
    #parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    #parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="COCO2017", choices=["NUS81", "NUS1k", "COCO2017"], help="available datasets: NUS81, NUS1k, COCO2017")
    #parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    #parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=50, type=int, help="number of max epoch")
    parser.add_argument("--margin_distance", default=0.7, type=float, help="margin distance between positive and negative samples")
    parser.add_argument("--learning_rate", default=0.0006, type=float, help="learning rate")
    parser.add_argument("--decay_model", default="STEPLR", choices=["STEPLR", "NONE"], help="learning rate decay model")
    parser.add_argument("--step_size", default=10, type=int, help="step size of step decay")
    parser.add_argument("--decay_rate", default=0.65, type=float, help="decay rate of step decay")
    parser.add_argument("--log_directory", default="runs/", help="log directory of tensorboard")
    parser.add_argument("--loss_lambda", default=0.1, type=float, help="weight of II triplet loss")
    parser.add_argument("--global_sample", default=False, action='store_true', help="whether select a positive sample globally")
    parser.add_argument("--depth_of_tag_model", default=1, type=int, help="the number of convolutional layers in tag model")
    parser.add_argument("--tag_max_length", default=8, type=int, help="the max length of tags")
    
    # tag_weight
    #parser.add_argument("--gpu", default=-1, type=int, help="the number of gpu to be used")

    options = parser.parse_args()

    # get name
    model_name = "_lr_" + str(options.learning_rate)
    model_name = model_name + "_dm_" + str(options.decay_model)
    model_name = model_name + "_dcy_" + str(options.decay_rate)
    model_name = model_name + "_dep_" + str(options.depth_of_tag_model)
    if options.global_sample:
        model_name = model_name + "_gs"
    model_name = model_name + "_ml_" + str(options.tag_max_length)
    model_name = model_name + "_md_" + str(options.margin_distance)

    print("model_name:", model_name)

    with open('./job_info.txt', 'a') as f:
        f.write(options.job_num + ":" + model_name + "\n")

    if device == torch.device('cuda'):
        Gpus = GPUtil.getGPUs()
        for gpu in Gpus:
            if gpu.memoryTotal > 30000:
                batch_size = 256
                break
            if gpu.memoryTotal > 15000:
                batch_size = 128
                break
            if gpu.memoryTotal > 12000:
                batch_size = 64
                break
            if gpu.memoryTotal > 11000:
                batch_size = 56
                break
            if gpu.memoryTotal > 7000:
                batch_size = 32
                break
    print("batch_size:", batch_size)
# dataset
    if options.dataset == "NUS81": 
        train_data = nus.NUS_WIDE_Helper(nus.DataSetType.Train_81, min_tag_num=1)
        valid_data = nus.NUS_WIDE_Helper(nus.DataSetType.Test_81, min_tag_num=1)
    elif options.dataset == "NUS1k":
        train_data = nus.NUS_WIDE_Helper(nus.DataSetType.Train_1k, min_tag_num=1)
        valid_data = nus.NUS_WIDE_Helper(nus.DataSetType.Test_1k,  min_tag_num=1)
    elif options.dataset == "COCO2017":
        train_data = coco.coco_Helper(coco.DataSetType.Train17, min_tag_num=1)
        valid_data = coco.coco_Helper(coco.DataSetType.Valid17, min_tag_num=1)

    num_workers = 16
    #train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, drop_last=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    #valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, drop_last=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    #for_evalue
    #train_loader_E = torch.utils.data.DataLoader(train_data, batch_size=int(batch_size/2), num_workers=num_workers, pin_memory=True)
    #valid_loader_E = torch.utils.data.DataLoader(valid_data, batch_size=int(batch_size/2), num_workers=num_workers, pin_memory=True) 

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    #for_evalue
    train_loader_E = torch.utils.data.DataLoader(train_data, batch_size=int(batch_size/2), num_workers=num_workers, pin_memory=True)
    valid_loader_E = torch.utils.data.DataLoader(valid_data, batch_size=int(batch_size/2), num_workers=num_workers, pin_memory=True)


# log       
    log_path = options.log_directory + options.dataset + "/" + model_name + "/"
    writer = SummaryWriter(log_dir=log_path)

# model
    image_model = TenNet.VGG16_Normalize().to(device)

    static_word_matrix = compute_word_matrix(train_data.image_tags, Word_Dimensions)
    tag_list = valid_data.get_tag_list()
    filters=[3, 4, 5]
    filter_num=[100, 100, 100]
    # tag_list, feature_dims, word_dims, dropout_prob, filters, filter_num, depth, in_channel=2, word_matrix=None, max_length=8
    tag_model = TenNet.TenNet_Tag( tag_list, Feature_Dimensions, Word_Dimensions, 0.5, filters, 
                                   filter_num, options.depth_of_tag_model, word_matrix=static_word_matrix,
                                   max_length=options.tag_max_length).to(device)

    triplet_loss = TripletLossFunc(options.margin_distance)

    optimizer = torch.optim.RMSprop([{'params' : image_model.parameters()}, 
                                    {'params' : tag_model.parameters()}], 
                                    lr=options.learning_rate)

    
    if options.decay_model == "STEPLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=options.step_size, 
                                                    gamma=options.decay_rate)
    elif options.decay_model == "NONE":
        scheduler = None
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma, last_epoch=-1)

# run
    TenNet.run(writer, image_model, tag_model, train_loader, valid_loader, train_loader_E, 
                valid_loader_E, triplet_loss, optimizer, scheduler, 
                options.margin_distance, options.global_sample,
                options.epoch, options.loss_lambda, print_log=True)

    params = {
        "DEVICE": device,
        "WRITER": writer,
        "EPOCH": options.epoch,
        "BATCH_SIZE": batch_size,

        "Image_Model": image_model,

        "Margin_Distance": options.margin_distance,
        "DROPOUT_PROB": 0.5,
        "LEARNING_RATE": options.learning_rate
    }






if __name__ == "__main__":
    main()