import torch 
import torch.nn as nn 
import torch.nn.functional as F

from Define import *



class TripletLossFunc(nn.Module): 
    def __init__(self, alpha = Margin_Distance, t1 = 0, t2 = 0, beta = 0.05): 
        super(TripletLossFunc, self).__init__() 
        self.alpha = alpha
        self.t1 = t1
        self.t2 = t2
        self.beta = beta

    def forward(self, anchor, positive, negative): 
        pos_dis = F.pairwise_distance(anchor, positive)
        matched = torch.pow(pos_dis, 2)
        neg_dis = F.pairwise_distance(anchor, negative)
        mismatched = torch.pow(neg_dis,2)
        part_1 = torch.clamp(matched - mismatched + self.alpha, min=self.t1) 
        part_2 = torch.clamp(matched, min=self.t2) 
        loss = part_1 #+ self.beta * part_2 
        return torch.mean(loss), torch.mean(pos_dis), torch.mean(neg_dis)