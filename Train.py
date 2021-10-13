
from TripletLossFunc import TripletLossFunc
import random

import torch
import torch.nn.functional as F

import torchvision.models as models

from Utils import *
from Define import *
from NUS_WIDE_Helper import *

from jupyterplot import ProgressPlot
from tqdm.notebook import tqdm
from TenNetImage import *
from TenNetTag import *


#NUS_WIDE
train_data = NUS_WIDE_Helper(DataSetType.Train_81, Number_Of_Images_Valid)
valid_data = NUS_WIDE_Helper(DataSetType.Test_81, Number_Of_Images_Valid)
#test_data = NUS_WIDE_Helper(DataSetType.Test_81)

batch_size = BATCH_SIZE
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=batch_size)
#test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
#image_model = TenNet_Image().to(device)
image_model = models.vgg16(pretrained=True).to(device)
tag_model = TenNet_Tag(train_data.get_tag_num()).to(device)
optim = torch.optim.Adam([{'params' : image_model.parameters()}, {'params' : tag_model.parameters()}], lr=0.001)


n_epochs = N_Epochs
pp = ProgressPlot(plot_names=["loss", "positive dis", "negative dis"], line_names=["train", "val"],
                  x_lim=[0, n_epochs-1], y_lim=[[0,100], [0,50], [0,100]])

min_valid_loss = -1
pbar = tqdm(range(n_epochs))

for e in pbar:
    train_loss = 0
    train_positive_dis = 0
    train_negative_dis = 0

    image_model.train()
    tag_model.train()
    lossImageTag = TripletLossFunc(Margin_Distance)
    lossImageImage = TripletLossFunc(Margin_Distance)

    num = 0
    for (x_images,y_tags) in train_loader:
        
        if random.random() > 0.5:
            continue
        num = num + x_images.shape[0]

        x_images, y_tags = x_images.to(device), y_tags.to(device)
        images_feature = image_model(x_images)
        tags_feature = tag_model(y_tags)
        
        # in feature space
        dist_image_tag_pos =  F.pairwise_distance(images_feature, tags_feature)

        # first triplet loss, an image, cor tag, and a neg image
        anchor = images_feature
        positive = tags_feature
        z_images_neg = get_one_neighbor(images_feature, dist_image_tag_pos + Margin_Distance)
        negative = torch.cat([images_feature[i].view(1,-1) for i in z_images_neg])

        lossIT, dist_image_tag_pos, dist_image_image_neg = lossImageTag(anchor, positive, negative)

        # second triplet loss, an image, a pos image, a neg image
        anchor = images_feature
        z_images_pos, z_images_neg = get_pos_neg(y_tags)
        positive = torch.cat([images_feature[i].view(1,-1) for i in z_images_pos])
        negative = torch.cat([images_feature[i].view(1,-1) for i in z_images_neg])

        lossII, dist_image_image_pos, dist_image_image_neg =lossImageImage(anchor, positive, negative)

        Lambda = 0.3
        loss = lossIT +  Lambda * lossII

        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.item()
        train_positive_dis += dist_image_tag_pos.float().sum().item()
        train_negative_dis += dist_image_image_neg.float().sum().item()

    train_loss /= num
    train_positive_dis /= num
    train_negative_dis /= num
    
    image_model.eval()
    tag_model.eval()
    valid_loss = 0
    valid_positive_dis = 0
    valid_negative_dis = 0
    with torch.no_grad():
        for (x_images,y_tags) in valid_loader:
            x_images, y_tags = x_images.to(device), y_tags.to(device)
            images_feature = image_model(x_images)
            tags_feature = tag_model(y_tags)
            
            # in feature space
            dist_image_tag_pos =  F.pairwise_distance(images_feature, tags_feature)

            anchor = images_feature
            positive = tags_feature
            z_images_neg = get_one_neighbor(images_feature, dist_image_tag_pos + Margin_Distance)
            negative = torch.cat([images_feature[i].view(1,-1) for i in z_images_neg])

            lossIT, dist_image_tag_pos, dist_image_image_neg = lossImageTag(anchor, positive, negative)

            anchor = images_feature
            z_images_pos, z_images_neg = get_pos_neg(y_tags)
            positive = torch.cat([images_feature[i].view(1,-1) for i in z_images_pos])
            negative = torch.cat([images_feature[i].view(1,-1) for i in z_images_neg])

            Lambda = 0.3
            lossII, dist_image_image_pos, dist_image_image_neg =lossImageImage(anchor, positive, negative)

            loss = lossIT +  Lambda * lossII
       
            valid_loss += loss.item()
            valid_positive_dis += dist_image_tag_pos.float().sum().item()
            valid_negative_dis += dist_image_image_neg.float().sum().item()

        valid_loss /= len(valid_loader)
        valid_positive_dis /= len(valid_loader)
        valid_negative_dis /= len(valid_loader)

        if min_valid_loss == -1 or min_valid_loss > valid_loss:
            min_valid_loss = valid_loss
            torch.save({
            'epoch': e,
            'image_model_state_dict': image_model.state_dict(),
            'tag_model_state_dict': tag_model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'valid_loss': valid_loss,
            }, "best_val.ckpt")
    
    pp.update([[train_loss, valid_loss], [train_positive_dis, valid_positive_dis], [train_negative_dis, valid_negative_dis]])
    pbar.set_description(f"train loss: {train_loss:.4f}, pos distance: {train_positive_dis:.4f}, neg distance: {train_negative_dis:.4f}")
pp.finalize()
