import math
import torch
import numpy as np

from src.utils.utils import l2_normalize
import torch


class SimCLRObjective(torch.nn.Module):

    def __init__(self, outputs1, outputs2, t, push_only=False):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.t = t
        self.push_only = push_only
    
    def get_loss(self):
        batch_size = self.outputs1.size(0)  # batch_size x out_dim
        witness_score = torch.sum(self.outputs1 * self.outputs2, dim=-1)
        outputs12 = torch.cat([self.outputs1, self.outputs2], dim=0)
        # [num_examples, 2 * num_examples]
        witness_norm_raw = self.outputs1 @ outputs12.T
        witness_norm = torch.logsumexp(
            witness_norm_raw / self.t, dim=1) - math.log(2 * batch_size)
        loss = -torch.mean(witness_score / self.t - witness_norm)
        return loss
    
    
class SimCLRTripletObjective(torch.nn.Module):

    def __init__(self, outputs1, outputs2, outputs3, t, push_only=False, margin=0.5, l2=0.0):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        self.outputs3 = l2_normalize(outputs3, dim=1)
        self.t = t
        self.margin = margin
        self.l2 = l2
        self.push_only = push_only
    
    def get_loss(self):
        batch_size = self.outputs1.size(0)  # batch_size x out_dim
        loss_neighbour = torch.sqrt(((self.outputs1 - self.outputs2) ** 2).sum(dim=1))
        loss_distant = - torch.sqrt(((self.outputs1 - self.outputs3) ** 2).sum(dim=1))
        
        loss = loss_neighbour + loss_distant + self.margin
        loss = torch.mean(torch.nn.functional.relu(loss))
        if self.l2 != 0:
            loss += self.l2 * (torch.norm(self.outputs1) + torch.norm(self.outputs2) + torch.norm(self.outputs3))
        
        return loss

