import torch
import torch.nn as nn
import torch.optim as optim


def train(loader, model, args):
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(args.epochs):
        for enc_input_batch, dec_input_batch, dec_output_batch in loader:
            # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
            h_0 = torch.zeros(1, args.batch_size, args.n_hidden).to(args.device)

            (enc_input_batch, dec_input_batch, dec_output_batch) = (enc_input_batch.to(args.device), dec_input_batch.to(args.device), dec_output_batch.to(args.device))
            # enc_input_batch : [batch_size, n_step+1, n_class]
            # dec_input_batch : [batch_size, n_step+1, n_class]
            # dec_output_batch : [batch_size, n_step+1], not one-hot
            pred = model(enc_input_batch, h_0, dec_input_batch)
            # pred : [n_step+1, batch_size, n_class]
            pred = pred.transpose(0, 1)  # [batch_size, n_step+1(=6), n_class]
            loss = 0
            for i in range(len(dec_output_batch)):
                # pred[i] : [n_step+1, n_class]
                # dec_output_batch[i] : [n_step+1]
                loss += criterion(pred[i], dec_output_batch[i])
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
