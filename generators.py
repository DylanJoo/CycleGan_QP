import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from modules import Encoder, Decoder, RNNLayer
from criterion import KL
import utils
import random

class SVAE(nn.Module):

    def __init__(self,
            embed_dim=300, hidden_dim=256, latent_dim=16, 
                 teacher_forcing=False, dropout=0, n_direction=1, n_parallel=1, 
                 max_src_len=100, max_tgt_len=20,
                 vocab_size=5000, sos_idx=2, eos_idx=3, pad_idx=0, unk_idx=1,
                 attention=False,
                 args=False):
        
        super().__init__()
        #===Argument parser activated
        if args :
            vocab_size = args.vocab_size
            embed_dim, hidden_dim, latent_dim = args.embedding_dimension, args.hidden_dimension, args.latent_dimension
            teacher_forcing, dropout, n_direction, n_parallel = args.teacher_forcing, args.dropout, args.n_direction, args.n_parallel
            max_src_len, max_tgt_len = args.max_src_length, args.max_tgt_length
            sos_idx, eos_idx, pad_idx, unk_idx = args.sos_idx, args.eos_idx, args.pad_idx, args.unk_idx
            k ,x0, af = args.k, args.x0, args.af
            attn = args.attention
            
        #===Test the GPU availability
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        #===Parameters
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.hidden_n = n_direction * n_parallel #bidirectional or parallel layers
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.dropout = nn.Dropout(p=dropout)
        self.teacher_forcing = teacher_forcing
        self.attn = attn

        #==Variational==
        self.k = k
        self.x0 = x0
        self.af = af
        
        #===Tokens Indices
        self.sos, self.eos, self.pad, self.unk = sos_idx, eos_idx, pad_idx, unk_idx

        #===Embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.to(self.device)

        #===Base layers in en/de
        gru_layer_en = RNNLayer(embed_dim, hidden_dim, n_parallel)
        gru_layer_de = RNNLayer(embed_dim, hidden_dim, n_parallel)
        
        #===Main Archetecture(enc, dec)
        self.encoder = Encoder(gru_layer_en, 1)
        self.decoder = Decoder(gru_layer_de, 1)
        
        #===VAE( latent z space then to hidden context)
        self.hidden2mean = nn.Linear(hidden_dim * self.hidden_n, latent_dim)
        self.hidden2logv = nn.Linear(hidden_dim * self.hidden_n, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim * self.hidden_n)

        #===Output for generating
        self.outputs2vocab = nn.Linear(hidden_dim * n_direction, vocab_size)

        #===Loss function
        self.NLL = nn.NLLLoss(reduction='sum', ignore_index=self.pad)

    def load_prt(self, weight):
        self.embed = nn.Embedding.from_pretrained(weight)
        self.embed.to(self.device)

    def kl(self, dist, step):
           #mu, logv, step
        mu, logv, _ = dist
        kl_loss, kl_weight = KL(mu, logv, step, self.k, self.x0, self.af)
        return (kl_loss * kl_weight)
    
    def sampling(self, generation, context):
        '''Finish the reamining generations.'''
        input = self.embed(generation)
        n_batch, t = generation.size(0), generation.size(1)-1       
        samples = torch.LongTensor(n_batch, 0).to(self.device)
        
        # Simulating from <token_0> to <token_t>,
        # to obatin the <token_t+1> & <hidden_t>
        hidden = context
        output, hidden = self.deocder(input, hidden)
        logit_t = F.softmax(self.outputs2vocab(output[:, -1, :]), dim=-1)
        input = logit_t.multinomial(1)

        # Drop the <sos> & Concat <token_t+1> 
        generation = torch.cat((generations[:, 1:t], input), dim=1)
        t += 1
        
        # Generating for reamining from <token_t+1> to <token_len-1>,
        # to obtain <token_t+2> to <token_len>
        for i in range(t+1, self.max_tgt_len+1):
            output, hidden = self.decoder(input, hidden)
            logit = F.softmax(self.outputs2vocab(output).squeeze(), dim=-1)
            input = logit.multinomial(1)
            generations = torch.cat((generations, input), dim=1)
            
        return generations

        
    def forward(self, src, src_len, tgt, tgt_len, split):

        src = src.to(self.device)
        n_batch = tgt.size(0)
        src = self.embed(src)
        src = pack(src, src_len, batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(src) 
        hidden = hidden.squeeze()

        z_ = torch.rand([n_batch, self.latent_dim]).to(self.device)
        mu = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv) #? why exponential
        z = z_ * std + mu

        context = self.latent2hidden(z)  
        context = context.unsqueeze(0)

        if (self.teacher_forcing > random.uniform(0,1)):
            teacher = True
        else:
            teacher = False

        if split == 'valid':
            teacher = False

        t=0
        if teacher:
            tgt = tgt[:, :-1]
            tgt = torch.cat((torch.LongTensor([[self.sos]]*n_batch).to(self.device), tgt), dim=1)
            tgt = self.embed(tgt)
            tgt = pack(tgt, tgt_len, batch_first=True, enforce_sorted=False)
            pad_output, _ = self.decoder(tgt, context)
            output, output_len = unpack(pad_output, batch_first=True, total_length=self.max_tgt_len)
            logits = F.log_softmax(self.outputs2vocab(output), dim=-1)
            generations = torch.argmax(logits, dim=-1)
            
        else:
            input = torch.LongTensor([[self.sos]]*n_batch).to(self.device)
            generations = torch.LongTensor(n_batch, 0).to(self.device)
            logits = []

            while(t<self.max_tgt_len):
                input = self.embed(input)
                output, context = self.decoder(input, context)

                logit = self.outputs2vocab(output).squeeze()
                # Multinomial
                prob = F.softmax(logit, dim=-1)
                input = prob.multinomial(1)
                logit = torch.log(prob)
                # MLE
                #logit = F.log_softmax(logit, dim=-1)
                #input = torch.argmax(logit, dim=-1).unsqueeze(1)
                logits.append(logit)  
                generations = torch.cat((generations, input), dim=1) 
                t=t+1
            logits = torch.stack(logits, dim=1)
                
        return generations, (mu, logv, z), logits

    def inference(self, n=4, z=None):
        '''Infernece from 'assigned sentence' or 'random sentence'
        '''
        if z is None:
            n_batch = n
            z = torch.randn([n_batch, self.latent_dim]).to(self.device)
        else:
            z = z.to(self.device)
            n_batch = z.size(0)

        context = self.latent2hidden(z)
        
        if self.hidden_n != 1:            # unflatten the hidden
            context = context.view(self.hidden_n, n_batch, self.hidden_dim)
        else:
            context = context.unsqueeze(0)
            
        #context for generating from start domain

        #dynamic stopping of setnence generation
        seq_idx = torch.arange(0, n_batch, out=torch.LongTensor())
        seq_torun = torch.arange(0, n_batch, out=torch.LongTensor())
        seq_mask = torch.tensor([True]*n_batch)
        
        #record the "Unfinished" sequence sentence
        run = seq_idx.clone() 
        mask = seq_mask.clone()
        
        generations = torch.LongTensor(n_batch, self.max_tgt_len).fill_(self.pad)

        t = 0
        while(t<self.max_tgt_len and len(seq_torun)>0):
            
            if t==0:
                src_t = torch.LongTensor([self.sos]*n_batch).to(self.device)
            src_t = src_t.unsqueeze(1) 
            src_t_embed = self.embed(src_t)
            output, context = self.decoder(src_t_embed, context)

            logit = self.outputs2vocab(output)
            src_t = torch.argmax(logit, dim=-1).flatten()

            generations = self._update_sample(generations, src_t, seq_torun, t)

            mask =  (src_t != self.eos)
            if sum(mask) > 0:
                src_t = src_t[mask]
                context = context[:, mask, :] 
            seq_torun = seq_torun[mask]
            t += 1
            
        return generations, z
    
    def test(self, src, src_len):
        '''for the real scenario'''
        n_batch = src.size(0)
        src = src.to(self.device)
        #src==>embed==>rnn-packed
        src = self.embed(src)
        src = pack(src, src_len, batch_first=True, enforce_sorted=False)
        #===Encoding===
        _, hidden = self.encoder(src)
        if self.hidden_n != 1:             # flatten the hidden
            hidden = hidden.view(n_batch, self.hidden_dim*self.hidden_n)
        else:
            hidden = hidden.squeeze()

        #z_(before) reparameterize with mu and sigma (prior)
        z_ = torch.rand([n_batch, self.latent_dim]).to(self.device)
        mu = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv) #? why exponential
        z = z_ * std + mu

        return z
