import torch
import argparse
import torch.utils.data as Data
import train
import preprocess
import module
import test


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=3)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--n_step', type=int, default=0)
parser.add_argument('--n_class', type=int, default=0)
parser.add_argument('--n_hidden', type=int, default=0)
args = parser.parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

letter = [c for c in 'SE?abcdefghijklmnopqrstuvwxyz']
letter2idx = {n:i for i, n in enumerate(letter)}
seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
args.n_step = max([max(len(i), len(j)) for i, j in seq_data])
args.n_hidden = 128
args.n_class = len(letter)
args.batch_size = 3

enc_input_all, dec_input_all, dec_output_all = preprocess.make_data(args, seq_data, letter2idx)
loader = Data.DataLoader(preprocess.TranslateDataset(enc_input_all, dec_input_all, dec_output_all), args.batch_size, True)

model = module.Seq2Seq(args).to(args.device)
train.train(loader, model, args)
print('test')
print('man ->', test.translate('man', model, args, letter, letter2idx))
print('mans ->', test.translate('mans', model, args, letter, letter2idx))
print('king ->', test.translate('king', model, args, letter, letter2idx))
print('black ->', test.translate('black', model, args, letter, letter2idx))
print('up ->', test.translate('up', model, args, letter, letter2idx))








