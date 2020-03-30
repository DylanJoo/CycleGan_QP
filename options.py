import argparse
from copy import deepcopy

class OPT():

    def __init__(self, vs=10000, pad=0, sos=1, eos=2, unk=3):
        # To be enter real args.
        self.vs = vs
        self.pad = pad
        self.sos = sos
        self.eos = eos
        self.unk = unk
        
    def __call__(self, mode='default', domain='default'):
        parser = argparse.ArgumentParser(prog='model.control')
        parser.add_argument('--data_dir', type=str, default='msmarco_data')
        parser.add_argument('-bi', '--n_direction', type=int, default=1)
        if domain in ['Q', 'P']:
            parser = self.add_opt(parser)
            parser = self.get_opt(parser, mode, domain)
        else:
            parser = self.get_opt(parser, mode, domain)
            
        return parser.parse_args()

    def add_opt(self, parser):
        parser.add_argument('--vocab_size', type=int, default=self.vs)
        parser.add_argument('--sos_idx', type=int, default=self.sos)
        parser.add_argument('--eos_idx', type=int, default=self.eos)
        parser.add_argument('--unk_idx', type=int, default=self.unk)
        parser.add_argument('--pad_idx', type=int, default=self.pad)
        return parser

    def get_opt(self, parser, mode, domain):
        if mode == 'data':
            parser.add_argument('--mt', action='store_true', default=True)
            parser.add_argument('--create_data', action='store_true', default=True)
            parser.add_argument('--min_occ', type=int, default=3)
            parser.add_argument('--max_src_length', type=int, default=100)
            parser.add_argument('--max_tgt_length', type=int, default=20)
            parser.add_argument('-nb', '--n_batch', type=int, default=32)
            
        elif mode == 'generator':
            #Model param
            parser.add_argument('-ed', '--embedding_dimension', type=int, default=300)
            parser.add_argument('-hd', '--hidden_dimension', type=int, default=256)
            parser.add_argument('-ld', '--latent_dimension', type=int, default=16)
            parser.add_argument('-dp', '--dropout', type=float, default=0.5)
            parser.add_argument('-np', '--n_parallel', type=int, default=1)
            parser.add_argument('-nl', '--n_layer', type=int, default=1)
            parser.add_argument('-at', '--attention', action='store_true', default=False)
            
            if domain == 'Q':
                parser.add_argument('-tf', '--teacher_forcing', type=float, default=0.5)
                parser.add_argument('--max_src_length', type=int, default=100)
                parser.add_argument('--max_tgt_length', type=int, default=20)
                
            elif domain == 'P':
                parser.add_argument('-tf', '--teacher_forcing', type=float, default=1)
                parser.add_argument('--max_src_length', type=int, default=20)
                parser.add_argument('--max_tgt_length', type=int, default=100)

            #--Variational config--
            parser.add_argument('-af', '--af', type=str, default='logistic')
            parser.add_argument('-k', '--k', type=float, default=0.0025)
            parser.add_argument('-x0', '--x0', type=int, default=2500)
        
        elif mode == 'discriminator':
            parser.add_argument('-ed', '--embedding_dimension', type=int, default=300)
            parser.add_argument('-cs', '--convolution_size', type=int, default=5)
            parser.add_argument('-hd', '--hidden_dimension', type=int, default=256)
            parser.add_argument('-dp', '--dropout', type=float, default=0.5)
            parser.add_argument('-nn', '--n_neurons', type=int, default=5)
            
        else:
            assert mode == 'default', 'Unsupported mode.'
            
        return parser
