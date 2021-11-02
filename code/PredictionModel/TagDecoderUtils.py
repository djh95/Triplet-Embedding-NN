import torch
from jupyterplot import ProgressPlot
from tqdm.notebook import tqdm
from enum import IntEnum

from Define import *
from .Utils import *



def compute_loss(data, expect_res, encoder, decoder, loss_funk):
    tag_features = encoder(data)
    output = decoder(tag_features)    # get prediction
    loss = loss_funk(output,expect_res.float())  # 计算两者的误差
    loss = torch.mean(loss)

    return loss, output

class LossModel (IntEnum):
    # tag -> feature ->tag
    DecoderModel = 0
    # image -> feature ->tag
    PredictModel = 1
    HybridModel = 2
    HybridModel2 = 3

def single_epoch_computation(decoder, image_model, tag_model, loader, loss_funk, optim, threshold, updata=True, mod=LossModel.PredictModel):
    loss = 0
    p_tag_corr_num = 0 # the number of correctly predicted tags
    p_tag_total_num = 0 # the number of predicted tags
    tag_total_num = 0
    tag_accuracy = 0
    
    for (x_images,y_tags) in loader:

        y_tags = y_tags.to(device)
        if mod == LossModel.PredictModel:
            x_images = x_images.to(device)
            batch_loss, output = compute_loss(x_images, y_tags, image_model, decoder, loss_funk)
            print(batch_loss, output)
        elif mod == LossModel.DecoderModel:
            batch_loss, output = compute_loss(y_tags, y_tags, tag_model, decoder, loss_funk)
        else:
            x_images = x_images.to(device)
            batch_loss_i, output_i = compute_loss(x_images, y_tags, image_model, decoder, loss_funk)
            batch_loss_t, output_t = compute_loss(y_tags, y_tags, tag_model, decoder, loss_funk)
            batch_loss = (batch_loss_i + batch_loss_t) / 2
            output = (output_i + output_t) / 2

        if updata:
            optim.zero_grad()   # 清空上一步的残余更新参数值
            batch_loss.backward()         # 误差反向传播, 计算参数更新值
            optim.step()        # 将参数更新值施加到 net 的 parameters 上

        loss += batch_loss.item()
        tag_total_num += sum(sum(y_tags)).item()
        for i in range(output.shape[0]):
            predicted_tag = get_tag_from_prediction(output[i], threshold=threshold)
            s = similarity_tags(y_tags[i],predicted_tag)
            p_tag_corr_num =  s + p_tag_corr_num
            p_tag_total_num = predicted_tag.float().sum().item() + p_tag_total_num 
            tag_accuracy = s / y_tags[i].float().sum().item() + tag_accuracy

    loss = loss / len(loader)
    p_tag_corr_num = p_tag_corr_num / len(loader.dataset)
    p_tag_total_num = p_tag_total_num / len(loader.dataset)
    tag_total_num = tag_total_num / len(loader.dataset)
    tag_accuracy = tag_accuracy / len(loader.dataset)

    return loss, p_tag_corr_num, p_tag_total_num, tag_accuracy, tag_total_num

def train(decoder, tag_model, image_model,  loader, loss_funk, optim, threshold=0.5, mod=LossModel.DecoderModel):
    
    tag_model.eval()
    image_model.eval()
    decoder.train()

    res = single_epoch_computation(decoder, image_model, tag_model, loader, loss_funk, optim, threshold, updata=True, mod=mod)

    return res

def predict(decoder, tag_model, image_model,  loader, loss_funk, optim, threshold=0.5, save_best=True, max_accuracy=-1, e=0 ):
    
    tag_model.eval()
    image_model.eval()
    decoder.eval()

    with torch.no_grad():
        res = single_epoch_computation(decoder, image_model, tag_model, loader, loss_funk, optim, threshold, updata=False, mod=LossModel.PredictModel)

    if save_best and res[3] > max_accuracy:
        max_accuracy = res[3]
        torch.save({
            'epoch': e,
            'model_state_dict': decoder.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'loss': res[0],
            }, "../SavedModelState/decoder_model_" + str(Margin_Distance) +".ckpt")
            
    return res + (max_accuracy,)
    

def output_loss_num(s, loss_num):
    print(  s + "\n" +
            f"loss: {loss_num[0]:.2f},  " +
            f"p_tag_corr_num: {loss_num[1]:.2f},  " +
            f"p_tag_total_num: {loss_num[2]:.2f},  " + 
            f"P_tag_accuracy: {loss_num[3]:.2f},  " +  
            f"tag_total_num: {loss_num[4]:.2f}")
    return

def getDecoderModel(decoder, name="../SavedModelState/decoder_model.ckpt"):
    try:
        print(name)
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


def printLossLog(res, n_epochs):

    pbar = tqdm(range(min(len(res), n_epochs)))
    for e in pbar:
        
        loss_num_train =  res[e][0]
        output_loss_num(f"epoch:{e}: 1-train dataset with train model", loss_num_train)
        
        loss_num_valid = res[e][1]   
        output_loss_num(f"epoch:{e}: 2-valid dataset with evalue model", loss_num_valid)

def printLossProgressPlot(res, n_epochs):

    max_v = compute_column_maximum(res)[0]
    if max_v > 1:
        max_v = np.ceil(max_v)

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

def printTagNumProgressPlot(res, n_epochs):

    max_v = max(compute_column_maximum(res)[1:])

    pp = ProgressPlot(plot_names=["Tag num"],
                  line_names=["p_correct_t", "p_total_t", "p_correct_v", "p_total_v", "total"],
                  x_lim=[0, n_epochs-1], 
                  y_lim=[0, max_v])

    pbar = tqdm(range(min(len(res), n_epochs)))
    
    for e in pbar:
        
        loss_num = res[e]
        pp.update([[loss_num[0][1], loss_num[0][2], loss_num[1][1], loss_num[1][2], loss_num[0][4]]])

    pp.finalize()

def printAccuracyProgressPlot(res, n_epochs):

    pp = ProgressPlot(plot_names=["Accuracy"],
                  line_names=["train", "valid"],
                  x_lim=[0, n_epochs-1], 
                  y_lim=[0, 1])

    pbar = tqdm(range(min(len(res), n_epochs)))
    
    for e in pbar:
        
        loss_num = res[e]
        pp.update([[loss_num[0][3],loss_num[1][3]]])

    pp.finalize()

def run(model, mod, tag_model, image_model, train_loader, valid_loader, loss_funk, threshold, n_epochs, test, getDecoderFromFile=False, name="../SavedModelState/decoder_model.ckpt"):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    max_accuracy = -1
    res = []
    if getDecoderFromFile:
        getDecoderModel(model, name)
    else:
        pbar = tqdm(range(n_epochs))

    for e in pbar:

        
        if mod == LossModel.HybridModel2:
            loss_num_train = train(model, tag_model, image_model, train_loader, loss_funk, optim, threshold, mod=LossModel.DecoderModel)
            output_loss_num(f"epoch:{e}: 1-train dataset with train model", loss_num_train)
            loss_num_train = train(model, tag_model, image_model, train_loader, loss_funk, optim, threshold, mod=LossModel.PredictModel)
            output_loss_num(f"epoch:{e}: 1-train dataset with train model", loss_num_train)
        else:
            loss_num_train = train(model, tag_model, image_model, train_loader, loss_funk, optim, threshold, mod=mod)
            output_loss_num(f"epoch:{e}: 1-train dataset with train model", loss_num_train)
        
        loss_num_valid = predict(model, tag_model, image_model, valid_loader, loss_funk, optim, threshold, True, max_accuracy, e)   
        output_loss_num(f"epoch:{e}: 2-valid dataset with evalue model", loss_num_valid)
        max_accuracy = loss_num_valid[5]
    
        res.append([loss_num_train,loss_num_valid])

        if (test or device == torch.device('cpu')) and e == 0:
            break

    return res