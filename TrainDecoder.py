from TripletLossFunc import TripletLossFunc
import torch
import torch.nn.functional as F

from Utils import *
from Define import *
from NUS_WIDE_Helper import *

from jupyterplot import ProgressPlot
from tqdm.notebook import tqdm
from TenNetImage import *
from TenNetTag import *
from TagDecoder import *

'''
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=False, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
image_model = TenNet_Image().to(device)
tag_model = TenNet_Tag(train_data.get_tag_num()).to(device)
optim = torch.optim.Adam([{'params' : image_model.parameters()}, {'params' : tag_model.parameters()}], lr=0.001)
'''

def train_decoder(tag_model, image_model, train_data, train_loader, valid_loader):
    n_epochs = N_Epochs_Decoder

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = TagDecoder(train_data.get_tag_num()).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    pp = ProgressPlot(plot_names=["loss", "mean num of correct pre tags", "mean num of pre tags"], line_names=["train", "val"],
                  x_lim=[0, n_epochs-1], y_lim=[[0,5], [0,10], [0,20]])
    best_val_acc = -1
    pbar = tqdm(range(n_epochs))
    lossfunc = F.pairwise_distance

    tag_model.eval()
    image_model.eval()
    for e in pbar:
        train_loss = 0
        train_tag_acc = 0 # the number of correctly predicted tags
        train_tag_num = 0 # the number of predicted tags

        model.train()
        # the decoder takes the tag features as input
        for (x,y_tags) in train_loader:
            optim.zero_grad()   # 清空上一步的残余更新参数值
            y_tags =  y_tags.to(device)
            tag_features = tag_model(y_tags)
            output = model(tag_features)    # get prediction
            #print(output)
            loss = lossfunc(output,y_tags).sum()  # 计算两者的误差
            loss.backward()         # 误差反向传播, 计算参数更新值
            optim.step()        # 将参数更新值施加到 net 的 parameters 上

            train_loss += loss.item()
            for i in range(output.shape[0]):
                predicted_tag = get_tag_from_prediction(output, threshold=Threshold)
                train_tag_acc = similarity_tags(y_tags[i],predicted_tag) + train_tag_acc
                train_tag_num = predicted_tag.float().sum().item() + train_tag_num                

        train_loss = train_loss / len(train_loader.dataset)
        train_tag_acc = train_tag_acc / len(train_loader.dataset)
        train_tag_num = train_tag_num / len(train_loader.dataset)


        valid_loss = 0
        valid_tag_acc = 0
        valid_tag_num = 0

        model.eval()
        # the decoder takes image features as input
        with torch.no_grad():
            for (x_images,y_tags) in valid_loader:
                x_images = x_images.to(device)
                image_features = image_model(x_images)
                output = model(image_features)
                loss = lossfunc(output, y_tags).sum()

                valid_loss += loss.item()
                for i in range(output.shape[0]):
                    predicted_tag = get_tag_from_prediction(output, threshold=Threshold)
                    valid_tag_acc = similarity_tags(y_tags[i],predicted_tag) + valid_tag_acc
                    valid_tag_num = predicted_tag.float().sum().item() + valid_tag_num

        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_tag_num = valid_tag_num / len(valid_loader.dataset)
        valid_tag_acc = valid_tag_acc / len(valid_loader.dataset) / valid_tag_num

        pp.update([[train_loss, valid_loss], [train_tag_acc, valid_tag_acc], [train_tag_num, valid_tag_num]])
        pbar.set_description(f"train loss: {train_loss:.4f}, train acc.: {train_tag_acc:.4f}, train num.: {train_tag_num:.4f}")
    
    
        if valid_tag_acc > best_val_acc:
            valid_tag_acc = best_val_acc
            torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'val_acc': valid_tag_acc,
            }, "best_val.ckpt")
  
    pp.finalize()