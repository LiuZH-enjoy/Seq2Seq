import torch.nn as nn
import torch


class Seq2Seq(nn.Module):
    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.RNN(input_size=args.n_class, hidden_size=args.n_hidden, dropout=0.5)
        self.decoder = nn.RNN(input_size=args.n_class, hidden_size=args.n_hidden, dropout=0.5)
        self.fc = nn.Linear(args.n_hidden, args.n_class)
    
    def forward(self, enc_input, enc_hidden, dec_input):
        # enc_input(=input_batch): [batch_size, n_step+1, n_class]
        # dec_input(=output_batch): [batch_size, n_step+1, n_class]
        enc_input = enc_input.transpose(0, 1) # enc_input: [n_step+1, batch_size, n_class]
        dec_input = dec_input.transpose(0, 1) # dec_input: [n_step+1, batch_size, n_class]
        # h_t : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, h_t = self.encoder(enc_input, enc_hidden)
        # outputs : [n_step+1, batch_size, num_directions(=1) * n_hidden(=128)]
        outputs, _ = self.decoder(dec_input, h_t)
        model = self.fc(outputs) # model : [n_step+1, batch_size, n_class]
        return model



