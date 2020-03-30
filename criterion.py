import torch
import torch.nn as nn
import numpy as np

class GANLoss(nn.Module):

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, prediction, target_is_real):

        if target_is_real:
            target = torch.LongTensor([1]) # if target is "Being True"
        else:
            target = torch.LongTensor([0]) # if target is "Being False"

        target = target.repeat(prediction.size(0)).to(self.device)
        loss = self.loss(prediction, target)
        return loss

class NLLLoss(nn.Module):

    def __init__(self,  pad_idx=0):
        super(NLLLoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum', ignore_index=pad_idx)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, logit, target, length):

        logit = logit[:, :torch.max(length)]
        target = target[:, :torch.max(length)]

        flat_logit = logit.reshape(-1, logit.size(2))
        target = target.reshape(-1)

        target = target.to(self.device)

        loss = self.loss(flat_logit, target)


        return loss

def KL(mean, logv, \
       step, k, x0, anneal_f='logistic'):
    kl_loss = -0.5 * torch.sum(-mean.pow(2) -logv.exp() + logv + 1)
    
    if anneal_f == 'logistic':
        kl_weight= float(1/(1+np.exp(-k*(step - x0))))
    elif anneal_f == 'linear':
        kl_weight =  min(1, step/x0)

    return kl_loss, kl_weight


# NLL: reduction='sum': is worked.
# Gan: reduction='sum': Not been used.
