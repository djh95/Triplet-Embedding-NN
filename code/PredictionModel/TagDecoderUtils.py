from TripletLossFunc import TripletLossFunc
import torch
import torch.nn.functional as F

from Utils import *
from Define import *

from jupyterplot import ProgressPlot
from tqdm.notebook import tqdm

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

def compute_loss(data, expect_res, encoder, decoder, loss_funk):

    tag_features = encoder(data)
    output = decoder(tag_features)    # get prediction
    loss = torch.mean(loss_funk(output,expect_res))  # 计算两者的误差

    return loss, output



def single_epoch_computation(decoder, image_model, tag_model, loader, loss_funk, optim, threshold, updata=True, predict=False):
    loss = 0
    tag_corr_num = 0 # the number of correctly predicted tags
    tag_total_num = 0 # the number of predicted tags
    
    for (x_images,y_tags) in loader:

        y_tags = y_tags.to(device)
        if predict == True:
            x_images = x_images.to(device)
            batch_loss, output = compute_loss(x_images, y_tags, image_model, decoder, loss_funk)
        else:
            batch_loss, output = compute_loss(y_tags, y_tags, tag_model, decoder, loss_funk)

        if updata:
            optim.zero_grad()   # 清空上一步的残余更新参数值
            batch_loss.backward()         # 误差反向传播, 计算参数更新值
            optim.step()        # 将参数更新值施加到 net 的 parameters 上

        loss += batch_loss.item()
        for i in range(output.shape[0]):
            predicted_tag = get_tag_from_prediction(output, threshold=threshold)
            tag_corr_num = similarity_tags(y_tags[i],predicted_tag) + tag_corr_num
            tag_total_num = predicted_tag.float().sum().item() + tag_total_num 

    loss = loss / len(loader)
    tag_corr_num = tag_corr_num / len(loader.dataset)
    tag_total_num = tag_total_num / len(loader.dataset)

    if tag_total_num == 0:
        tag_accuracy = 0
    else:
        tag_accuracy = tag_corr_num / tag_total_num

    return loss, tag_corr_num, tag_total_num, tag_accuracy

def train(decoder, tag_model, image_model,  loader, loss_funk, optim, threshold=0.5):
    
    tag_model.eval()
    image_model.eval()
    decoder.train()
    
    res = single_epoch_computation(decoder, image_model, tag_model, loader, loss_funk, optim, threshold, updata=True, predict=False)

    return res

def predict(decoder, tag_model, image_model,  loader, loss_funk, optim, threshold=0.5, save_best=True, max_accuracy=-1, e=0 ):
    
    tag_model.eval()
    image_model.eval()
    decoder.eval()

    with torch.no_grad():
        res = single_epoch_computation(decoder, image_model, tag_model, loader, loss_funk, optim, threshold, updata=False, predict=True)

    if save_best and res[3] > max_accuracy:
        max_accuracy = res[3]
        torch.save({
            'epoch': e,
            'model_state_dict': decoder.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'loss': res[0],
            }, "decoder_best_val.ckpt")
            
    return res + (max_accuracy,)
    

def output_loss_num(s, loss_num):
    print(  s + "\n" +
            f"loss: {loss_num[0]:.2f},  " +
            f"tag_corr_num: {loss_num[1]:.2f},  " +
            f"tag_total_num: {loss_num[2]:.2f},  " + 
            f"tag_accuracy: {loss_num[4]:.2f}\n " )
    return

def getDecoderModel(decoder, name = "decoder_best_val.ckpt"):
    try:
        checkpoint = torch.load(name)
        decoder.load_state_dict(checkpoint["model_state_dict"]) 
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print("loss: ", loss)
        print("epoch: ",epoch)
        print("Load last checkpoint data")
    except FileNotFoundError:
        print("Can\'t found " + name)
    return

def updataProgressPlot(pp, loss_num_train, loss_num_valid):
    pp.update([[loss_num_train[0], loss_num_valid[0]], 
               [loss_num_train[1], loss_num_train[2]], 
               [loss_num_train[3], loss_num_valid[3]]])



def printResult(res, n_epochs):

    pbar = tqdm(range(min(len(res), n_epochs)))
    pp = ProgressPlot(plot_names=["loss", "mean num of tags", "accuracy"], 
                  line_names=["train/correct", "valid/total"],
                  x_lim=[0, n_epochs-1], 
                  y_lim=[[0,50], [0,20], [0,1]])

    for e in pbar:

        loss_num_train =  res[e][0]
        output_loss_num(f"epoch:{e}: 1-train dataset with train model", loss_num_train)
        
        loss_num_valid = res[e][1]   
        output_loss_num(f"epoch:{e}: 2-valid dataset with evalue model", loss_num_valid)
    
        updataProgressPlot(pp, loss_num_train, loss_num_valid)
        
    pp.finalize()