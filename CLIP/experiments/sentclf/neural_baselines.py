import argparse
from collections import Counter, defaultdict
from datetime import date
import glob
import json
import os
import pickle
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import multilabel_eval
sys.path.append('../tagger')
from train import get_embeddings

from constants import *

class CNN(nn.Module):

    def __init__(self, pretrained_embs, embed_size, vocab_size, num_filter_maps=100, filter_size=4):
        super(CNN, self).__init__()
        if pretrained_embs:
            embs = torch.Tensor(pretrained_embs)
            self.embed = nn.Embedding.from_pretrained(embs, freeze=False)
        else:
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.conv = nn.Conv1d(embed_size, num_filter_maps, kernel_size=filter_size, padding=round(filter_size/2))
        nn.init.xavier_uniform_(self.conv.weight)

        self.fc = nn.Linear(num_filter_maps, len(LABEL_TYPES))
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, target):
        #embed
        x = self.embed(x)
        x = x.transpose(1,2)
        # conv/max-pooling
        try:
            x = self.conv(x)
        except:
            import pdb; pdb.set_trace()
        x = F.max_pool1d(torch.tanh(x), kernel_size=x.size()[2])
        x = x.squeeze(dim=2)
        #linear output
        x = self.fc(x)
        #sigmoid to get predictions
        loss = F.binary_cross_entropy_with_logits(x, target)
        yhat = torch.sigmoid(x)
        return yhat, loss

class SentDataset(Dataset):
    def __init__(self, fname, word2ix=None):
        self.sents = pd.read_csv(fname)
        self.tok_cnt = Counter()
        if word2ix is not None:
            self.word2ix = word2ix
            self.ix2word = {ix:word for word,ix in self.word2ix.items()}
        else:
            self.word2ix = {'<PAD>': 0}
            self.ix2word = {}
            self.min_cnt = 0

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        row = self.sents.iloc[idx]
        sent = [x.lower() for x in eval(row.sentence)]
        labels = eval(row.labels)
        doc_id = row.doc_id
        return sent, labels, doc_id

    def build_vocab(self):
        print("building vocab...")
        for ix in tqdm(range(len(self))):
            sent, labels, doc_id = self[ix]
            self.tok_cnt.update(sent)
        # add 1 for pad token
        self.word2ix.update({word: ix+1 for ix,(word,count) in enumerate(sorted(self.tok_cnt.items(), key=lambda x: x[0])) if count > self.min_cnt})
        # add UNK to the end
        self.word2ix[UNK] = len(self.word2ix)
        self.ix2word = {ix:word for word,ix in self.word2ix.items()}

    def set_vocab(self, vocab_file):
        # add 1 for pad token
        self.word2ix.update({row.strip(): ix+1 for ix,row in enumerate(open(vocab_file))})
        # add UNK to the end
        self.word2ix[UNK] = len(self.word2ix)
        self.ix2word = {ix:word for word,ix in self.word2ix.items()}

def collate(batch, word2ix, task):
    sents, labels, doc_ids = [], [], []
    # sort by decreasing length
    batch = sorted(batch, key=lambda x: -len(x[0]))
    max_length = len(batch[0][0])
    for sent, label, doc_id in batch:
        sent = [word2ix.get(w, word2ix[UNK]) for w in sent]
        sent.extend([0 for ix in range(len(sent), max_length)])
        sents.append(sent)
        if task == 'finegrained':
            label_ixs = [LABEL_TYPES.index(l) for l in label]
            label = np.zeros(len(LABEL_TYPES))
            label[label_ixs] = 1
            labels.append(label)
        else:
            import pdb; pdb.set_trace()
            labels.append(label)
        doc_ids.append(doc_id)
    return torch.LongTensor(sents), torch.Tensor(labels), doc_ids

def check_best_model_and_save(model, metrics_hist, criterion, out_dir):
    is_best = False
    if criterion == 'loss':
        if np.nanargmin(metrics_hist[criterion]) == len(metrics_hist[criterion]) - 1:
            # save model
            sd = model.state_dict()
            torch.save(sd, f'{out_dir}/model_best_{criterion}.pth')
            is_best = True
    else:
        if np.nanargmax(metrics_hist[criterion]) == len(metrics_hist[criterion]) - 1:
            # save model
            sd = model.state_dict()
            torch.save(sd, f'{out_dir}/model_best_{criterion}.pth')
            is_best = True
    return is_best

def save_metrics(metrics_hist, out_dir):
    # save predictions
    if out_dir is not None and not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(f'{out_dir}/metrics.json', 'w') as of:
        json.dump(metrics_hist, of, indent=1)
    # make and save plot
    for metric in metrics_hist:
        plt.figure()
        plt.plot(metrics_hist[metric])
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.title(f"dev {metric} vs. epochs")
        plt.savefig(f'{out_dir}/dev_{metric}_plot.png')
        plt.close()

def early_stop(metrics_hist, criterion, patience):
    if len(metrics_hist[criterion]) >= patience:
        if criterion == 'loss':
            return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
        else:
            return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        return False

def main(args):
    dev_fname = args.train_fname.replace('train', 'val')
    test_fname = args.train_fname.replace('train', 'test')
    tr_data = SentDataset(args.train_fname)
    if not args.vocab_file:
        tr_data.build_vocab()
        date_str = date.today().strftime('%Y%m%d')
        with open(f'vocab_{date_str}.txt', 'w') as of:
            for word, _ in sorted(tr_data.word2ix.items(), key=lambda x: x[1]):
                of.write(word + "\n")
    else:
        tr_data.set_vocab(args.vocab_file)
    tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate(batch, tr_data.word2ix, args.task))
    dv_data = SentDataset(dev_fname, tr_data.word2ix)
    dv_loader = DataLoader(dv_data, batch_size=1, shuffle=False, collate_fn=lambda batch: collate(batch, tr_data.word2ix, args.task))

    # load pre-trained embeddings
    if args.embed_file is None or len(glob.glob(args.embed_file)) == 0:
        word_list = [word for ix,word in sorted(tr_data.ix2word.items(), key=lambda x: x[0])]
        pretrained_embs = get_embeddings("BioWord", f"{PWD}/CLIP/experiments/tagger/embeddings/", word_list)
        import pdb; pdb.set_trace()
        emb_out_fname = 'embs.pkl'
        pickle.dump(pretrained_embs, open(emb_out_fname, 'wb'))
    else:
        pretrained_embs = pickle.load(open(args.embed_file, 'rb'))


    # add one for UNK
    model = CNN(pretrained_embs, args.embed_size, len(tr_data.word2ix)+1)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    timestamp = time.strftime('%b_%d_%H:%M:%S', time.localtime())
    out_dir = f'results/{args.model}_{timestamp}'

    losses = []
    metrics_hist = defaultdict(list)
    best_epoch = 0
    model.train()
    stop_training = False
    for epoch in range(args.max_epochs):
        for batch_ix, batch in tqdm(enumerate(tr_loader)):
            sents, labels, doc_ids = batch
            optimizer.zero_grad()
            yhat, loss = model(sents.to(DEVICE), labels.to(DEVICE))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if batch_ix % 500 == 0 and batch_ix > 0:
                print(f"loss: {np.mean(losses[-10:])}")

        # eval
        with torch.no_grad():
            model.eval()
            yhat_raw = []
            yhat = []
            y = []
            for ix, x in tqdm(enumerate(dv_loader)):
                sent, label, doc_id = x
                pred, _ = model(sent.to(DEVICE), label.to(DEVICE))
                pred = pred.cpu().numpy()[0]
                yhat_raw.append(pred)
                yhat.append(np.round(pred))
                y.append(label.cpu().numpy()[0])
            yhat = np.array(yhat)
            yhat_raw = np.array(yhat_raw)
            y = np.array(y)
            metrics = multilabel_eval.all_metrics(yhat, y, k=3, yhat_raw=yhat_raw, calc_auc=True, label_order=LABEL_TYPES)

            for name, metric in metrics.items():
                metrics_hist[name].append(metric)

            # save best model, creating results dir if needed
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            save_metrics(metrics_hist, out_dir)
            is_best = check_best_model_and_save(model, metrics_hist, args.criterion, out_dir)
            if is_best:
                best_epoch = epoch

            if early_stop(metrics_hist, args.criterion, args.patience):
                print(f"{args.criterion} hasn't improved in {args.patience} epochs, early stopping...")
                stop_training = True
                break
            multilabel_eval.print_metrics(metrics, True)
 
        if stop_training:
            break

    # save args
    with open(f'{out_dir}/args.json', 'w') as of:
        of.write(json.dumps(args.__dict__, indent=2) + "\n")

    if args.max_epochs > 0:
        # save the model at the end
        sd = model.state_dict()
        torch.save(sd, out_dir + "/model.pth")

        # reload the best model
        print(f"\nReloading and evaluating model with best {args.criterion} (epoch {best_epoch})")
        sd = torch.load(f'{out_dir}/model_best_{args.criterion}.pth')
        model.load_state_dict(sd)

    # eval on dev at end
    with torch.no_grad():
        model.eval()
        yhat_raw = []
        yhat = []
        y = []
        for ix, x in tqdm(enumerate(dv_loader)):
            sent, label, doc_id = x
            pred, _ = model(sent.to(DEVICE), label.to(DEVICE))
            pred = pred.cpu().numpy()[0]
            yhat_raw.append(pred)
            yhat.append(np.round(pred))
            y.append(label.cpu().numpy()[0])
        yhat = np.array(yhat)
        yhat_raw = np.array(yhat_raw)
        y = np.array(y)
        metrics = multilabel_eval.all_metrics(yhat, y, k=3, yhat_raw=yhat_raw, calc_auc=True, label_order=LABEL_TYPES)
        print(args)
        multilabel_eval.print_metrics(metrics, True)

        thresh, best_thresh_metrics = multilabel_eval.metrics_v_thresholds(yhat_raw, y)
        print(f"best threshold metrics (threshold = {thresh})")
        multilabel_eval.print_metrics(best_thresh_metrics)

        label_type_metrics = multilabel_eval.f1_per_label_type(yhat_raw, y, LABEL_TYPES, thresh)
        multilabel_eval.print_per_label_metrics(label_type_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_fname", type=str)
    parser.add_argument("model", choices=['cnn', 'lstm'])
    parser.add_argument("--embed_file", type=str, help="path to a file holding pre-trained token embeddings")
    parser.add_argument("--task", choices=['binary', 'finegrained'], default='finegrained')
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--criterion", type=str, default="f1_micro", required=False, help="metric to use for early stopping")
    parser.add_argument("--patience", type=int, default=5, required=False, help="epochs to wait for improved criterion before early stopping (default 5)")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_size", type=int, default=200)
    parser.add_argument("--num_filter_maps", type=int, default=100)
    parser.add_argument("--filter_size", type=int, default=4)
    parser.add_argument("--vocab_file", type=str, help="path to precomputed vocab")
    parser.add_argument("--seed", type=int, default=11, help="random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
