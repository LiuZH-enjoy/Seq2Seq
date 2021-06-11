import torch
import preprocess


def translate(word, model, args, letter, letter2idx):
    enc_input, dec_input, _ = preprocess.make_data(args, [[word, '?' * args.n_step]], letter2idx)
    enc_input, dec_input = enc_input.to(args.device), dec_input.to(args.device)
    # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
    hidden = torch.zeros(1, 1, args.n_hidden).to(args.device)
    output = model(enc_input, hidden, dec_input)
    # output : [n_step+1, batch_size, n_class]

    predict = output.data.max(2, keepdim=True)[1] # select n_class dimension
    decoded = [letter[i] for i in predict]
    translated = ''.join(decoded[:decoded.index('E')])

    return translated.replace('?', '')