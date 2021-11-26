import torch
from jupyterplot import ProgressPlot
from tqdm.notebook import tqdm

from Define import *
from .Utils import *



def compute_loss(data, expect_res, encoder, decoder, loss_funk):
    tag_features = encoder(data)
    output = decoder(tag_features)    # get prediction
    loss = loss_funk(output,expect_res.float())  # 计算两者的误差
    loss = loss.sum()

    return loss, output

def single_epoch_computation(decoder, image_model, tag_model, loader, loss_funk, optim, threshold, updata=True, mod=LossModel.PredictModel):
    loss = 0
    precision = 0
    recall = 0
    F1 = 0
    accuracy = 0
    tp = 0
    sum_p = 0
    sum_g = 0
    
    for (x_images,y_tags) in loader:

        y_tags = y_tags.to(device)

        if mod == LossModel.PredictModel:
            x_images = x_images.to(device)
            batch_loss, output = compute_loss(x_images, y_tags, image_model, decoder, loss_funk)
        elif mod == LossModel.DecoderModel:
            batch_loss, output = compute_loss(y_tags, y_tags, tag_model, decoder, loss_funk)
        else:
            x_images = x_images.to(device)
            batch_loss_i, output_i = compute_loss(x_images, y_tags, image_model, decoder, loss_funk)
            batch_loss_t, output_t = compute_loss(y_tags, y_tags, tag_model, decoder, loss_funk)
            batch_loss = (batch_loss_i + batch_loss_t) / 2
            output = output_i

        if updata:
            optim.zero_grad()   # 清空上一步的残余更新参数值
            batch_loss.backward()         # 误差反向传播, 计算参数更新值
            optim.step()        # 将参数更新值施加到 net 的 parameters 上

        loss += batch_loss.item()
        for i in range(output.shape[0]):
            prediction_tag_v = get_tag_from_prediction(output[i], threshold=threshold)
            res = compute_evaluation_terms(prediction_tag_v, y_tags[i])

            precision = precision + res[0]
            recall = recall + res[1]
            F1 = F1 + res[2]
            accuracy = accuracy + res[3]
            tp = tp + res[4]
            sum_p = sum_p + res[5]
            sum_g = res[6]
            
    
    num = loader.dataset.image_number
    loss = loss / num
    precision = precision / num
    recall = recall / num
    F1 = F1 / num
    accuracy = accuracy / num
    tp = tp / num
    sum_p = sum_p / num
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

    return loss, precision

def train(decoder, tag_model, image_model,  loader, loss_funk, optim, threshold=0.5, mod=LossModel.DecoderModel):
    
    tag_model.eval()
    image_model.eval()
    decoder.train()

    res = single_epoch_computation(decoder, image_model, tag_model, loader, loss_funk, optim, threshold, updata=True, mod=mod)

    return res

def predict(decoder, tag_model, image_model,  loader, loss_funk, optim, threshold=0.5, save_best=True, max_precision=-1, e=0 ):
    
    tag_model.eval()
    image_model.eval()
    decoder.eval()

    with torch.no_grad():
        res = single_epoch_computation(decoder, image_model, tag_model, loader, loss_funk, optim, threshold, updata=False, mod=LossModel.PredictModel)

    if save_best and res[1] > max_precision:
        max_precision = res[1]
        torch.save({
            'epoch': e,
            'model_state_dict': decoder.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'loss': res[0],
            }, "../SavedModelState/decoder_model_.ckpt")
            
    return res + (max_precision,)

def compute_evaluation_terms(prediction_tag_v, ground_truth_tag_v):
    tp = similarity_tags(prediction_tag_v, ground_truth_tag_v)
    sum_p = sum(prediction_tag_v)
    fp = sum_p - tp
    sum_g = sum(ground_truth_tag_v)
    fn = sum_g - tp
    tn = len(ground_truth_tag_v) - sum_p - fn

    precision = tp / sum_p
    recall = tp / sum_g
    if precision== 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    c = (tp+tn) / (tp+tn+fp+fn)
    return precision, recall, f1, c, tp, sum_p, sum_g

def output_loss(s, loss):
    print(  s + "\n" + f"loss: {loss:.4f}" )
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
        output_loss(f"epoch:{e}: 1-train dataset with train model", loss_num_train)
        
        loss_num_valid = res[e][1]   
        output_loss(f"epoch:{e}: 2-valid dataset with evalue model", loss_num_valid)

def printLossProgressPlot(res, n_epochs):

    pp = ProgressPlot(plot_names=["loss"],
                    line_names=["train", "valid"],
                    x_lim=[0, n_epochs-1], 
                    y_lim=[0, res[0][0]])

    pbar = tqdm(range(min(len(res), n_epochs)))
    for e in pbar:
    
        train_loss = res[e][0]
        valid_loss = res[e][1]
    
        pp.update([[train_loss[0], valid_loss[0]]])

    pp.finalize()

def run(model, mod, tag_model, image_model, train_loader, valid_loader, loss_funk, n_epochs, test, threshold=0.5, getDecoderFromFile=False, name=""):
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    max_presicion = -1
    res = []
    if getDecoderFromFile:
        getDecoderModel(model, name)
    else:
        pbar = tqdm(range(n_epochs))

    for e in pbar:

        
        if mod == LossModel.HybridModel2:
            loss_train = train(model, tag_model, image_model, train_loader, loss_funk, optim, threshold, mod=LossModel.DecoderModel)
            output_loss(f"epoch:{e}: 1-train dataset with train model", loss_train)
            loss_train = train(model, tag_model, image_model, train_loader, loss_funk, optim, threshold, mod=LossModel.PredictModel)
            output_loss(f"epoch:{e}: 1-train dataset with train model", loss_train)
        else:
            loss_train = train(model, tag_model, image_model, train_loader, loss_funk, optim, threshold, mod=mod)
            output_loss(f"epoch:{e}: 1-train dataset with train model", loss_train)
        
        loss_valid = predict(model, tag_model, image_model, valid_loader, loss_funk, optim, threshold, True, max_presicion, e)   
        output_loss(f"epoch:{e}: 2-valid dataset with evalue model", loss_valid)
        max_presicion = loss_valid[1]
    
        res.append([loss_train,loss_valid])

        if (test or device == torch.device('cpu')) and e == 0:
            break

    return res