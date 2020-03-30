import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

NGRAM=[1,2,3,4,5]
KERNELS=[100,100,150,200,200]

NEURONS=[100,150,200,256,300]


class CNN(nn.Module):

    def __init__(self, classes=2, vocab_size=5000, 
                 embed_dim=100, ngram=NGRAM, kernels=KERNELS,
                 dropout=0, 
                 args=False):
        super().__init__()

        #==Argument parser activated
        if args:
            vocab_size = args.vocab_size
            embed_dim = args.embedding_dimension
            ngram, kernels = ngram[:args.convolution_size], kernels[:args.convolution_size]
            dropout = args.dropout
            
        #==Test the GPU availability
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        #==Parameters
        self.embed_dim = embed_dim

        #==Embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed.to(self.device)

        #==Feature Convolutions
        #self.conv2 = nn.Conv2d(1, kernel, (2, embed_dim), padding=(1, 0))
        #self.conv3 = nn.Conv2d(1, kernel, (3, embed_dim), padding=(2, 0))
        #self.conv4 = nn.Conv2d(1, kernel, (4, embed_dim), padding=(2, 0))

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, kernel, (n, embed_dim)) for (kernel, n) in zip(kernels, ngram)])

        #==Binary classification
        self.final = nn.Linear(sum(kernels), 2)

    def load_prt(self, weight):
        self.embed = nn.Embedding.from_pretrained(weight)
        self.embed.to(self.device)
        
    def forward(self, src, src_len=False):
        '''Maybe sequence length is useful..
        '''
        src = src.to(self.device)
        src = self.embed(src).unsqueeze(1)

        features = [F.relu(conv(src)).squeeze(3) for conv in self.convs]
        features = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in features]

        outputs = torch.cat((features), 1)
        logits = self.final(outputs)

        return logits 
