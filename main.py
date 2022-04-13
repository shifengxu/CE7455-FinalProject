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
from subwords.subword_adapter import SubwordAdapter
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
parser.add_argument('--subword_vocab_size', nargs='+', type=int, default=0,
                    help='subword vocabulary size list. 0 means no subword')
parser.add_argument('--word_split_mode_list', nargs='+', type=str, default=['Char', 'Subword.1000'],
                    help='word split modes: Char|Subword.1000|Subword.5000')
parser.add_argument('--fragment_aggregate_mode_list', nargs='+', type=str, default=['None', 'CNN', 'LSTM'],
                    help='fragment (char or subword) aggregation modes: None|CNN|LSTM')
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

def evaluate(data_source, model, criterion, corpus: Corpus, subword_adapter: SubwordAdapter):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    numerator = 0
    denominator = 0
    model.init_hidden(args.batch_size)
    word_cnt = 0
    frag_cnt = 0
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            inputs, targets = get_batch(data_source, i, args.bptt)
            # inputs size: (35, 10), targets size (350,). 35 is bptt, 10 is bsz
            if subword_adapter:
                fragment3D, f_cnt = subword_adapter.inputs_to_ids(inputs, corpus.dictionary.idx2word)
            else:
                fragment3D, f_cnt = corpus.word_to_char(inputs)
            word_cnt += inputs.numel()
            frag_cnt += f_cnt
            outputs = model(inputs, fragment3D)
            # outputs size(350, 37974)
            total_loss += len(inputs) * criterion(outputs, targets).item()
            preds = torch.argmax(outputs, dim=1)
            numerator += torch.eq(targets, preds).sum()
            denominator += len(targets)
    loss = total_loss / (len(data_source) - 1)
    accu = numerator / denominator
    w_len = frag_cnt / word_cnt
    return loss, accu, numerator, denominator, w_len, frag_cnt, word_cnt

def train(epoch, train_data, model, lr, criterion, corpus: Corpus,
          subword_adapter: SubwordAdapter):
    # Turn on training mode which enables dropout.
    model.train()
    loss_total = 0.
    loss_cnt = 0
    word_cnt = 0
    frag_cnt = 0  # fragment count
    b_cnt = len(train_data) // args.bptt  # batch count
    model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        b_num = batch + 1 # batch number
        inputs, targets = get_batch(train_data, i, args.bptt)
        if subword_adapter:
            fragment3D, f_cnt = subword_adapter.inputs_to_ids(inputs, corpus.dictionary.idx2word)
        else:
            fragment3D, f_cnt = corpus.word_to_char(inputs)
        word_cnt += inputs.numel()
        frag_cnt += f_cnt
        model.zero_grad()
        output = model(inputs, fragment3D)
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
            w_len = frag_cnt / word_cnt
            msg = f"E{epoch:03d} | {b_num:5d}/{b_cnt} batches | lr {lr:02.4f} | " \
                  f"loss {cur_loss:6.3f} | ppl {math.exp(cur_loss):9.2f} " \
                  f"frag_cnt/word_cnt={w_len:4.3f}={frag_cnt}/{word_cnt}"
            log_fn(msg)
            loss_total = 0.
            loss_cnt = 0
            word_cnt = 0
            frag_cnt = 0

def run(word_split_mode, fragment_aggregate_mode):
    train_file_list = [OsPath.join(args.data_dir, f) for f in args.train_file_list]
    valid_file_list = [OsPath.join(args.data_dir, f) for f in args.valid_file_list]
    test_file_list  = [OsPath.join(args.data_dir, f) for f in args.test_file_list]
    corpus = Corpus(
        train_file_list,
        valid_file_list,
        test_file_list,
        subword_vocab_size=args.subword_vocab_size,
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
    if fragment_aggregate_mode is None or fragment_aggregate_mode.lower() == 'none':
        fragment_aggregate_mode = ''
    if word_split_mode.lower() == 'char':
        subword_adapter = None
        log_fn(f"subword_adapter = None due to word_split_mode='{word_split_mode}'")
        fragment_cnt = corpus.char_count + 1  # char id starts from 1. So need to plus 1.
    else: # Subword.1000
        log_fn(f"subword_adapter will be generated due to word_split_mode='{word_split_mode}'")
        v_str = word_split_mode.split('.')[1]
        v_size = int(v_str)
        subword_adapter = SubwordAdapter(train_file_list + valid_file_list, v_size, log_fn=log_fn)
        fragment_cnt = v_size + 2
        # Why plus 2:
        # 1) subword id starts from 1.
        # 2) if subword encoder returns empty (for example when encode "/"), return special id

    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers,
                     args.dropout, args.tied, log_fn=log_fn, device=device,
                     fragment_aggregate_mode=fragment_aggregate_mode,
                     fragment_cnt=fragment_cnt)
    model = model.to(device)
    criterion = nn.NLLLoss()
    lr = args.lr
    best_val_loss = None

    stat_dict = OrderedDict()  # statistic data dict
    stat_dict['valid_loss'] = []
    for f in test_file_list: stat_dict[f"test_loss of {f}"] = []

    for epoch in range(1, args.epochs + 1):
        log_fn(f"Epoch {epoch:03d}/{args.epochs} ==================================================")
        train(epoch, train_data, model, lr, criterion, corpus, subword_adapter)
        l, a, n, d, wl, fc, wc = evaluate(val_data, model, criterion, corpus, subword_adapter)
        log_fn(f"End E{epoch:03d} | valid loss {l:6.3f}; ppl {math.exp(l):9.2f} | "
               f"accu {a:6.4f} = {n}/{d} | w_len {wl:6.4f} = {fc}/{wc}")
        stat_dict['valid_loss'].append(l)
        val_loss = l
        for test_data, file in test_data_list:
            l, a, n, d, wl, fc, wc = evaluate(test_data, model, criterion, corpus, subword_adapter)
            log_fn(f"End E{epoch:03d} |  test loss {l:6.3f}; ppl {math.exp(l):9.2f} | "
                   f"accu {a:6.4f} = {n}/{d} | w_len {wl:6.4f} = {fc}/{wc} {file}")
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
        l, a, n, d, wl, fc, wc = evaluate(test_data, model, criterion, corpus, subword_adapter)
        log_fn(f"End. test loss {l:5.2f}; ppl {math.exp(l):8.2f} | "
               f"accu {a:6.4f} = {n}/{d} | w_len {wl:6.4f} = {fc}/{wc} {file}")
    log_fn('=' * 89)
    for k, ls_list in stat_dict.items():
        log_fn(k)
        str_list = [f"{ls:5.2f}" for ls in ls_list]
        log_fn(','.join(str_list))
        log_fn(f"{k} ppl")
        str_list = [f"{math.exp(ls):8.2f}" for ls in ls_list]
        log_fn(','.join(str_list))
    log_fn('')  # 3 empty lines by the end of the running
    log_fn('')
    log_fn('')
# run()

def main():
    for word_split_mode in args.word_split_mode_list:
        for fragment_aggregate_mode in args.fragment_aggregate_mode_list:
            log_file = f"./output_wsm_{word_split_mode}_fam_{fragment_aggregate_mode}.log"
            print(f"Log file: {log_file} open...")
            utils.log_info_file = open(log_file, 'w')
            run(word_split_mode, fragment_aggregate_mode)
            utils.log_info_file.close()
            utils.log_info_file = None
            print(f"Log file: {log_file} closed.")
    # for

if __name__ == '__main__':
    main()
