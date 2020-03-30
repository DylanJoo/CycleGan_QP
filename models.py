import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import itertools
from torch.nn import LayerNorm, MultiheadAttention, Dropout
from criterion import GANLoss, NLLLoss
from copy import deepcopy


class CycleGAN(nn.Module):

    def __init__(self, G_A, G_B, D_A, D_B, args, isTrain=True):
        super().__init__()
        #==Test the GPU availability
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #==Main architectures==
        self.netG_A = G_A
        self.netG_B = G_B
        self.netD_A = D_A
        self.netD_B = D_B        
        #==Sequence==
        self.max_A_len = args.max_A_len
        self.max_B_len = args.max_B_len
        self.embed_dim = args.embedding_dimension

        #==Meta data==
        self.lambda_idt = args.lambdaidt
        self.lambda_A = args.lambdaA
        self.lambda_B = args.lambdaB
        
        #==Preparations==
        self.optimizers = []
        self.isTrain = isTrain

        #==Losses
        self.gan_loss = GANLoss() #mean
        self.nll_loss = NLLLoss(pad_idx=G_A.pad) #sum
        self.L1 = nn.L1Loss() #mean

        #==Optimizers==initialization
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_B.parameters(), self.netG_A.parameters()),
            lr= args.learning_rate)
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            lr= args.learning_rate)
        #self.optimizers.append(self.optimizer_G)
        #self.optimizers.append(self.optimizer_D)

    def load_prt(self, weight):
        self.netG_A.load_prt(weight)
        self.netG_B.load_prt(weight)
        self.netD_A.load_prt(weight)
        self.netD_B.load_prt(weight)
        # Load pretrained embedding for all sub-models

    def forward(self, data, step, split='train'):
        self.set_input(data)

        full_A = torch.LongTensor([self.max_A_len] * self.real_A.size(0))
        full_B = torch.LongTensor([self.max_B_len] * self.real_B.size(0))

        self.fake_B, dist_B, self.logit_B = self.netG_B(self.real_A, self.real_A_len, self.real_B, self.real_B_len, split)  # A*--> B^
        self.rec_A, _, _  = self.netG_A(self.fake_B, full_B,  self.real_A, self.real_A_len, split)   # B^ --> A'
        self.fake_A, dist_A, self.logit_A  = self.netG_A(self.real_B, self.real_B_len, self.real_A, self.real_A_len, split)  # B*--> A^
        self.rec_B, _, _  = self.netG_B(self.fake_A, full_A,  self.real_B, self.real_B_len, split)  # A^ --> B'
        
        #dist: (mu, logv, z)
        self.loss_kl_B = self.netG_B.kl(dist_B, step)
        self.loss_kl_A = self.netG_A.kl(dist_A, step)

        #log the z(continuous space)
        z_A = dist_B[2] # form G_B generate from real A to fake B
        z_B = dist_A[2]
        self.loss_Z = 0

        #text to image process(embeddings)
        self.real_B_embed = self.netG_A.embed(self.real_A)
        self.rec_B_embed = self.netG_A.embed(self.rec_A)
        self.real_A_embed = self.netG_B.embed(self.real_B)
        self.rec_A_embed = self.netG_B.embed(self.rec_B)

    def set_input(self, input):
        self.real_A = input['src'].to(self.device) #Passage
        self.real_B = input['tgt'].to(self.device) #Query
        self.real_A_len = input['srclen'].to(self.device)
        self.real_B_len = input['tgtlen'].to(self.device)

    def set_grad(self, net, flag=False):
        for p in net:
            p.requires_grad = flag

    def optimize(self):
        #G_A & G_B
        self.set_grad([self.netD_B, self.netD_A], False)
        self.optimizer_G.zero_grad()
        self.loss_G()
        self.GLOSS.backward()
        self.optimizer_G.step()

        #D_A & D_B
        self.set_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.loss_D()
        self.loss_D_A.backward()
        self.loss_D_B.backward()
        self.optimizer_D.step()
        
    def loss_G(self):
        '''Loss at the generating side.'''
        #Idt loss of (difference between
        self.loss_idt_A, self.loss_idt_B = 0, 0
            
        #Gan loss of for G_A(D_B on fake_B)...
        self.loss_G_B_gan = self.gan_loss(self.netD_B(self.fake_B), True)
        self.loss_G_A_gan = self.gan_loss(self.netD_A(self.fake_A), True)

        # Gen loss
        rate = self.real_B_len/ self.real_A_len 
        self.loss_G_B_nl = self.nll_loss(self.logit_B, self.real_B, self.real_B_len)
        self.loss_G_A_nl = self.nll_loss(self.logit_A, self.real_A, self.real_A_len)

        # CyC loss
        n_batch = self.real_A.size(0)
        self.loss_cyc_A = self.L1(self.rec_A_embed, self.real_A_embed)
        self.loss_cyc_B = self.L1(self.rec_B_embed, self.real_B_embed)

        #Loss in total
        self.GLOSS = (self.loss_idt_A + self.loss_cyc_A) * self.lambda_A + \
                     (self.loss_idt_B + self.loss_cyc_B) * self.lambda_B + \
                     (self.loss_G_A_nl + self.loss_kl_A) * self.lambda_A + \
                     (self.loss_G_B_nl + self.loss_kl_B) * self.lambda_B + \
                     (self.loss_G_A_gan + self.loss_G_B_gan) + \
                     (self.loss_Z)

    def loss_D(self):
        '''Loss at the generating side.'''
        #If images, pooling is needed. In text..?, currently ignored.
        #Gan loss of (Discriminator classifying the truth and fasle)
        truth, pred = self.netD_A(self.real_A), self.netD_A(self.fake_A)
        self.loss_D_A = 0.5 * (self.gan_loss(truth, True) + self.gan_loss(pred, False))
        
        truth, pred = self.netD_B(self.real_B), self.netD_B(self.fake_B)
        self.loss_D_B = 0.5 * (self.gan_loss(truth, True) + self.gan_loss(pred, False))
        
