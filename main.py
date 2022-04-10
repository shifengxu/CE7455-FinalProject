# coding: utf-8
import argparse
import math
import torch
import torch.nn as nn
import torch.onnx
import os.path as OsPath
from collections import OrderedDict

from data import Corpus, get_batch
from models.model_rnn import RNNModel
import utils
from utils import log_info

log_fn = log_info  # define the log function

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
parser.add_argument('--subword_vs_list', nargs='+', type=int, default=[0, 5000],
                    help='subword vocabulary size list. 0 means no subword')
parser.add_argument('--char_mode_list', nargs='+', type=str, default=['None', 'CNN'],
                    help='char_mode list: None|CNN|LSTM')
parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0],
                    help='GPU ID list')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
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
parser.add_argument('--train_file_list', nargs='+', type=str, default=['adolescent_train.txt'],
                    help='train file list')
parser.add_argument('--valid_file_list', nargs='+', type=str, default=['adolescent_valid.txt'],
                    help='valid file list')
parser.add_argument('--test_file_list', nargs='+', type=str, default=['adult_test.txt'],
                    help='test file list')
args = parser.parse_args()
device = f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() and args.gpu_ids else "cpu"
device = torch.device(device)
log_fn(f"args: {args}")
log_fn(f"device: {device}")

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

def evaluate(data_source, model, criterion, corpus: Corpus):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    numerator = 0
    denominator = 0
    model.init_hidden(args.batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            inputs, targets = get_batch(data_source, i, args.bptt)
            # inputs size: (35, 10), targets size (350,). 35 is bptt, 10 is bsz
            chars3 = corpus.word_to_char(inputs)
            outputs = model(inputs, chars3)
            # outputs size(350, 37974)
            total_loss += len(inputs) * criterion(outputs, targets).item()
            preds = torch.argmax(outputs, dim=1)
            numerator += torch.eq(targets, preds).sum()
            denominator += len(targets)
    loss = total_loss / (len(data_source) - 1)
    accu = numerator / denominator
    return loss, accu, numerator, denominator

def train(epoch, train_data, model, lr, criterion, corpus: Corpus):
    # Turn on training mode which enables dropout.
    model.train()
    loss_total = 0.
    loss_cnt = 0
    b_cnt = len(train_data) // args.bptt  # batch count
    model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        b_num = batch + 1 # batch number
        inputs, targets = get_batch(train_data, i, args.bptt)
        chars3 = corpus.word_to_char(inputs)
        model.zero_grad()
        output = model(inputs, chars3)
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
            log_fn(msg)
            loss_total = 0.
            loss_cnt = 0

def run(subword_vs, char_mode):
    train_file_list = [OsPath.join(args.data_dir, f) for f in args.train_file_list]
    valid_file_list = [OsPath.join(args.data_dir, f) for f in args.valid_file_list]
    test_file_list  = [OsPath.join(args.data_dir, f) for f in args.test_file_list]
    corpus = Corpus(
        train_file_list,
        valid_file_list,
        test_file_list,
        subword_vocab_size=subword_vs,
        log_fn=log_fn
    )
    train_data = corpus.batchify(train_file_list, args.batch_size)
    train_data = train_data.to(device)
    val_data   = corpus.batchify(valid_file_list, args.batch_size)
    val_data   = val_data.to(device)
    test_data_list = []
    for file in test_file_list:
        batched = corpus.batchify([file], args.batch_size)
        batched = batched.to(device)
        test_data_list.append((batched, file))

    ntokens = corpus.ntokens
    if char_mode is None or char_mode.lower() == 'none':
        char_mode = ''
    char_cnt = corpus.char_count + 1  # char id starts from 1. So need to plus 1.
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers,
                     args.dropout, args.tied, log_fn=log_fn, device=device,
                     char_mode=char_mode, char_cnt=char_cnt)
    model = model.to(device)
    criterion = nn.NLLLoss()
    lr = args.lr
    best_val_loss = None

    stat_dict = OrderedDict()  # statistic data dict
    stat_dict['valid_loss'] = []
    for f in test_file_list: stat_dict[f"test_loss of {f}"] = []

    for epoch in range(1, args.epochs + 1):
        log_fn(f"Epoch {epoch:03d}/{args.epochs} ==================================================")
        train(epoch, train_data, model, lr, criterion, corpus)
        l, a, n, d = evaluate(val_data, model, criterion, corpus)
        log_fn(f"End E{epoch:03d} | valid loss {l:6.3f}; ppl {math.exp(l):9.2f} | accu {a:6.4f} = {n}/{d}")
        stat_dict['valid_loss'].append(l)
        val_loss = l
        for test_data, file in test_data_list:
            l, a, n, d = evaluate(test_data, model, criterion, corpus)
            log_fn(f"End E{epoch:03d} |  test loss {l:6.3f}; ppl {math.exp(l):9.2f} | accu {a:6.4f} = {n}/{d} {file}")
            stat_dict[f"test_loss of {file}"].append(l)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            log_fn(f"Save: {args.save}... ")
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
            log_fn(f"Save: {args.save}...Done. best_val_loss: {best_val_loss :6.4f}")
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 2.0
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
    for test_data, file in test_data_list:
        l, a, n, d  = evaluate(test_data, model, criterion, corpus)
        log_fn(f"End. test loss {l:5.2f}; ppl {math.exp(l):8.2f} | accu {a:6.4f} = {n}/{d} {file}")
    log_fn('=' * 89)
    for k, ls_list in stat_dict.items():
        log_fn(k)
        str_list = [f"{ls:5.2f}" for ls in ls_list]
        log_fn(','.join(str_list))
        log_fn(f"{k} ppl")
        str_list = [f"{math.exp(ls):8.2f}" for ls in ls_list]
        log_fn(','.join(str_list))
# run()

def main():
    for vs in args.subword_vs_list:
        for char_mode in args.char_mode_list:
            log_file = f"./output_sws_{vs:05d}_chm_{char_mode}.log"
            print(f"Log file: {log_file} open...")
            utils.log_info_file = open(log_file, 'w')
            run(vs, char_mode)
            utils.log_info_file.close()
            utils.log_info_file = None
            print(f"Log file: {log_file} closed.")
    # for

if __name__ == '__main__':
    main()
