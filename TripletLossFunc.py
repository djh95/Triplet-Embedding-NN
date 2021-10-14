import torch 
import torch.nn as nn 
import torch.nn.functional as F



class TripletLossFunc(nn.Module): 
    def __init__(self, alpha, t1 = 0, t2 = 0, beta = 0.05): 
        super(TripletLossFunc, self).__init__() 
        self.alpha = alpha
        self.t1 = t1
        self.t2 = t2
        self.beta = beta
        return 

    def forward(self, anchor, positive, negative): 
        matched = torch.pow(F.pairwise_distance(anchor, positive), 2) 
        mismatched = torch.pow(F.pairwise_distance(anchor, negative), 2) 
        part_1 = torch.clamp(matched - mismatched + self.alpha, min=self.t1) 
        part_2 = torch.clamp(matched, min=self.t2) 
        dist_hinge = part_1 + self.beta * part_2 
        loss = torch.mean(dist_hinge) 
        return loss, matched, mismatched