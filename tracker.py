import numpy as np
import os
import json
import torch

class Tracker():
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_score = -np.Inf
        self.es_flag = False
        self.val_loss_min = np.inf
        # initialize all the records
        self._purge()

    def _elbo(self, train_loss):
        loss = torch.cuda.FloatTensor([train_loss]) if torch.cuda.is_available() else torch.Tensor([train_loss]) 
        self.elbo = torch.cat((self.elbo, loss))


    def record_A(self, truth, predict, i2w, pad_idx, eos_idx):
        '''note the predicted and groundtruth of Domain A'''
        gt, hat = [], []
        useless = [torch.tensor(pad_idx), torch.tensor(eos_idx)]
        for i, (s1, s2) in enumerate(zip(truth.long(), predict)):
            gt.append(" ".join([i2w[str(int(idx))] for idx in s1 if idx not in useless])+"\n")
            hat.append(" ".join([i2w[str(int(idx))] for idx in s2 if idx not in useless])+"\n")

        self.gt_A += gt
        self.hat_A += hat
        
    def record_B(self, truth, predict, i2w, pad_idx, eos_idx):
        '''note the predicted and groundtruth'''
        gt, hat = [], []
        useless = [torch.tensor(pad_idx), torch.tensor(eos_idx)]
        for i, (s1, s2) in enumerate(zip(truth.long(), predict)):
            gt.append(" ".join([i2w[str(int(idx))] for idx in s1 if idx not in useless])+"\n")
            hat.append(" ".join([i2w[str(int(idx))] for idx in s2 if idx not in useless])+"\n")

        self.gt += gt
        self.hat += hat
        
    def dumps(self, epoch, file, domain='B'):
        if not os.path.exists(file):
            os.makedirs(file)
        with open(os.path.join(file+'/Target_E%i.txt'%epoch), 'w') as dump_tgt:
            dump_tgt.writelines(self.gt)
        with open(os.path.join(file+'/Predict_E%i.txt'%epoch), 'w') as dump_predict:
            dump_predict.writelines(self.hat)
        # If necessary, dump the passage prediction
        if domain == 'A':
            with open(os.path.join(file+'_A/Target_E%i.txt'%epoch), 'w') as dump_src:
                dump_src.writelines(self.gt_A)
            with open(os.path.join(file+'_A/Predict_E%i.txt'%epoch), 'w') as dump_predict:
                dump_predict.writelines(self.hat_A)

            
    def _save_checkpoint(self, epoch, file, model):
        if not os.path.exists(file):
            os.makedirs(file)
        torch.save(model, os.path.join(file+'/Model_E%i.txt'%epoch))
        print('Model_E%i saved'%epoch)
            
    def _purge(self):
        '''
        elbo: (list) of loss on each pahse.
        gt: (list) of groundtruth on validation phase.
        hat: (list) of predicted text output
        z: (tensor) stacked output from latent space.
        '''
        self.gt, self.hat = [], []
        self.gt_A, self.hat_A = [], []
        self.elbo = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.Tensor()
