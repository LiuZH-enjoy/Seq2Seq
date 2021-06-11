import torch
import numpy as np
import torch.utils.data as Data


def make_data(args, seq_data, letter2idx):
    enc_input_all, dec_input_all, dec_output_all = [], [], []
    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + '?'*(args.n_step - len(seq[i]))
        enc_input = [letter2idx[n] for n in (seq[0]+'E')]
        dec_input = [letter2idx[n] for n in ('S'+seq[1])]
        dec_output = [letter2idx[n] for n in (seq[1]+'E')]

        enc_input_all.append(np.eye(args.n_class)[enc_input])
        dec_input_all.append(np.eye(args.n_class)[dec_input])
        dec_output_all.append(dec_output)
    return torch.Tensor(enc_input_all), torch.Tensor(dec_input_all), torch.LongTensor(dec_output_all)

'''
enc_input_all: [6, n_step+1 (because of 'E'), n_class]
dec_input_all: [6, n_step+1 (because of 'S'), n_class]
dec_output_all: [6, n_step+1 (because of 'E')]
'''


class TranslateDataset(Data.Dataset):
    def __init__(self, enc_input_all, dec_input_all, dec_output_all):
        self.enc_input_all = enc_input_all
        self.dec_input_all = dec_input_all
        self.dec_output_all = dec_output_all

    def __len__(self):
        return len(self.enc_input_all)

    def __getitem__(self, idx):
        return self.enc_input_all[idx], self.dec_input_all[idx], self.dec_output_all[idx]
