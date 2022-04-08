# coding: utf-8
import argparse
import math
import torch
import torch.nn as nn
import torch.onnx
import os.path as OsPath

from data import Corpus, get_batch
from models.rnn import RNNModel

parser = argparse.ArgumentParser(description='CE7455 Final Project')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='location of the corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

corpus = Corpus(
    OsPath.join(args.data_dir, 'adolescent_train.txt'),
    OsPath.join(args.data_dir, 'adolescent_valid.txt'),
    OsPath.join(args.data_dir, 'adult_test.txt'),
    log_fn=print
)
train_data, val_data, test_data = corpus.batchify_all(args.batch_size, device)

def evaluate(data_source, model, criterion, eval_batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    numerator = 0
    denominator = 0
    model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            inputs, targets = get_batch(data_source, i, args.bptt)
            # inputs size: (35, 10), targets size (350,). 35 is bptt, 10 is bsz
            outputs = model(inputs)
            # outputs size(350, 37974)
            total_loss += len(inputs) * criterion(outputs, targets).item()
            preds = torch.argmax(outputs, dim=1)
            numerator += torch.eq(targets, preds).sum()
            denominator += len(targets)
    loss = total_loss / (len(data_source) - 1)
    accu = numerator / denominator
    return loss, accu, numerator, denominator

def train(epoch, model, lr, criterion):
    # Turn on training mode which enables dropout.
    model.train()
    loss_total = 0.
    loss_cnt = 0
    b_cnt = len(train_data) // args.bptt  # batch count
    model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        b_num = batch + 1 # batch number
        inputs, targets = get_batch(train_data, i, args.bptt)
        model.zero_grad()
        output = model(inputs)
        # output size: [700, 33278]
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        loss_total += loss.item()
        loss_cnt += 1

        if b_num % args.log_interval == 0 or b_num == b_cnt:
            cur_loss = loss_total / loss_cnt
            msg = f"E{epoch:03d} | {b_num:5d}/{b_cnt} batches | lr {lr:02.4f} | " \
                  f"loss {cur_loss:6.3f} | ppl {math.exp(cur_loss):9.2f}"
            print(msg)
            loss_total = 0.
            loss_cnt = 0

def main():
    ntokens = corpus.ntokens
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
    model = model.to(device)
    criterion = nn.NLLLoss()
    lr = args.lr
    best_val_loss = None

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, lr, criterion)
        l, a, n, d = evaluate(val_data, model, criterion)
        print(f"End E{epoch:03d} | valid loss {l:6.3f}; ppl {math.exp(l):9.2f} | accu {a:6.4f} = {n}/{d}")
        val_loss = l
        l, a, n, d = evaluate(test_data, model, criterion)
        print(f"End E{epoch:03d} |  test loss {l:6.3f}; ppl {math.exp(l):9.2f} | accu {a:6.4f} = {n}/{d}")
        print('=' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
    # for epoch

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # Run on test data.
    l, a, n, d  = evaluate(test_data, model, criterion)
    print('=' * 89)
    print(f"End. test loss {l:5.2f}; ppl {math.exp(l):8.2f} | accu {a:6.4f} = {n}/{d}")
    print('=' * 89)

if __name__ == '__main__':
    main()
