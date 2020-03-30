from generators import SVAE
from discriminators import CNN
from models import CycleGAN
from utils import *
from tracker import Tracker
from dataset import seq_data
import argparse
import numpy as np
from multiprocessing import cpu_count
from collections import OrderedDict
from torch.utils.data import DataLoader
from options import OPT

def main(args):
    print(args)
    splits = ['train', 'valid'] +  (['dev'] if args.test else [])
    # Load Options from opt module(too many arguments)
    opt = OPT()
    args_data = opt('data')
    # Load dataset
    datasets = OrderedDict()
    for split in splits:
        datasets[split]=seq_data(
            data_dir=args_data.data_dir,
            split=split,
            mt=args_data.mt,
            max_len_src=args_data.max_src_length,
            max_len_tgt=args_data.max_tgt_length,
            min_occ=args_data.min_occ
            )
    #-- Retreive params from datasets
    opt = OPT(
        vs=datasets['train'].vocab_size,
        sos=datasets['train'].sos_idx,
        eos=datasets['train'].eos_idx,
        pad=datasets['train'].pad_idx,
        unk=datasets['train'].unk_idx
        )
    print('Data OK')

    # Create sub-models
    GeneratorA = SVAE(args=opt('generator', 'P'))
    GeneratorB = SVAE(args=opt('generator', 'Q'))
    DiscriminatorA = CNN(args=opt('discriminator', 'P'))
    DiscriminatorB = CNN(args=opt('discriminator', 'Q'))

    print('Model OK')
    print(opt('generator', 'P'))
    print(opt('generator', 'Q'))
    
    # Build model
    model = CycleGAN(GeneratorA, GeneratorB, DiscriminatorA, DiscriminatorB, args)
    if args.fasttext:
        prt = torch.load(args.data_dir+'/prt_fasttext.model')
        model.load_prt(prt)
    print(model)
    
    if torch.cuda.is_available():
        model = model.cuda()
    device = model.device
    
    tracker = Tracker(patience=10, verbose=True) #record training history & es function
    step = 0
    
    for epoch in range(args.epochs):
        
        for split in splits:
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.n_batch,
                shuffle=(split=='train'),
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
                )
            if split == 'train':
                model.train()
            else:
                model.eval()
                
            #Executing
            for i, data in enumerate(data_loader):
                #Get batchsize
                n_batch = data['src'].size(0)

                #BP & Optimize with losses
                if split == 'train':
                    #Setup & FP, step for svae usage
                    model(data, step, split)
                    model.optimize()
                    step += 1
                    
                else:
                    with torch.no_grad():
                        model(data, step, split)
                        model.loss_G()
                        model.loss_D()
                        step += 1

                #LOSS
                LOSS_GAN_A, LOSS_GAN_B = model.loss_G_A_gan, model.loss_G_B_gan
                LOSS_NLL_A, LOSS_NLL_B = model.loss_G_A_nl, model.loss_G_B_nl
                LOSS_CYC = (model.loss_cyc_A + model.loss_cyc_B)
                LOSS_KL = (model.loss_kl_A + model.loss_kl_B)
                LOSS_D = (model.loss_D_A + model.loss_D_B)
                LOSS_Z = model.loss_Z
                
                loss = model.GLOSS + LOSS_D
                    
                #RECORD & RESULT(batch)
                if i % 100 == 0 or i+1 == len(data_loader):
                    print("{} Phase - Batch {}/{}, Loss: {}, \nGAN_B: {}, NLL_B: {}, Cyc: {}, KL: {}, D: {}, Z: {} ".format(
                        split.upper(), i, len(data_loader)-1, \
                        loss/n_batch, LOSS_GAN_B, LOSS_NLL_B, LOSS_CYC, LOSS_KL, LOSS_D, LOSS_Z))
                tracker._elbo(torch.Tensor([loss]))
                if split == 'valid':
                    tracker.record_B(model.real_B, model.fake_B, datasets['train'].i2w,
                                   datasets['train'].pad_idx, datasets['train'].eos_idx)                    
                    tracker.record_A(model.real_A, model.fake_A, datasets['train'].i2w,
                                   datasets['train'].pad_idx, datasets['train'].eos_idx)
        
           #SAVING & RESULT(epoch)
            if split == 'valid':
                tracker.dumps(epoch, args.dump_file, 'A') #dump the predicted text. and alse the sources.
            else:
                tracker._save_checkpoint(epoch, args.model_file, model.state_dict()) #save the checkpooint
            print("{} Phase - Epoch {} , Mean ELBO: {}".format(split.upper(), epoch, torch.mean(tracker.elbo)))

            tracker._purge()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='msmarco_data')
    parser.add_argument('--fasttext', action='store_true', default=False)
    parser.add_argument('-ep', '--epochs', type=int, default=8)
    parser.add_argument('-nb', '--n_batch', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-te', '--test', action='store_true')
    parser.add_argument('-eb', '--embedding_dimension', type=int, default=300)
    parser.add_argument('-ldi', '--lambdaidt', type=int, default=0)
    parser.add_argument('-lda', '--lambdaA', type=int, default=2)
    parser.add_argument('-ldb', '--lambdaB', type=int, default=10)
    parser.add_argument('-ma', '--max_A_len', type=int, default=100)
    parser.add_argument('-mb', '--max_B_len', type=int, default=20) 
    parser.add_argument('-df', '--dump_file', type=str, default='dumps')
    parser.add_argument('-mf', '--model_file', type=str, default='checkpoint')

    args = parser.parse_args()

    main(args)
