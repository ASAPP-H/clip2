import argparse
from collections import defaultdict
import csv
import os
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from constants import *
import multilabel_eval

def build_data_and_label_matrices(fname, cvec, tvec, lvec=None, fit=False):
    corpus = []
    labels = []
    with open(fname) as f:
        r = csv.DictReader(f)
        for row in r:
            corpus.append(' '.join(eval(row['sentence'])))
            if lvec:
                labels.append(';'.join(eval(row['labels'])))
            else:
                labels.append(1 if eval(row['labels']) else 0)

    if fit:
        cX = cvec.fit_transform(corpus)
        tX = tvec.fit_transform(corpus)
        # create multi-label matrix
        if lvec:
            yy = lvec.fit_transform(labels).todense()
    else:
        cX = cvec.transform(corpus)
        tX = tvec.transform(corpus)
        if lvec:
            yy = lvec.transform(labels).todense()
    if lvec is None:
        yy = labels

    #needed to make parallel training work due to bug in scipy https://github.com/scikit-learn/scikit-learn/issues/6614#issuecomment-209922294
    cX.sort_indices()
    tX.sort_indices()
    return cX, tX, yy

def metrics_v_thresholds(yhat_raw, y):
    names = ["acc", "prec", "rec", "f1"]
    thresholds = np.arange(0,1,.01)[1:]
    metric_threshs = defaultdict(list)
    for thresh in thresholds:
        yhat = (yhat_raw[:,1] > thresh).astype(int)
        acc = accuracy_score(y, yhat)
        prec = precision_score(y, yhat)
        rec = recall_score(y, yhat)
        f1 = f1_score(y, yhat)
        for name, val in zip(names, [acc, prec, rec, f1]):
            metric_threshs[name].append(val)
    best_ix = np.nanargmax(metric_threshs['f1'])
    best_thresh = thresholds[best_ix]
    best_thresh_metrics = {name: vals[best_ix] for name, vals in metric_threshs.items()}
    best_thresh_metrics['auc'] = roc_auc_score(y, yhat_raw[:,1])

    prec_90_ix = np.where(np.array(metric_threshs['prec']) > 0.9)[0][0]
    prec_90_thresh = thresholds[prec_90_ix]
    prec_90_thresh_metrics = {name: vals[prec_90_ix] for name, vals in metric_threshs.items()}

    rec_90_ix = np.where(np.array(metric_threshs['rec']) > 0.9)[0][-1]
    rec_90_thresh = thresholds[rec_90_ix]
    best_thresh_metrics['prec@rec=90'] = metric_threshs['prec'][rec_90_ix]

    rec_95_ix = np.where(np.array(metric_threshs['rec']) > 0.95)[0][-1]
    rec_95_thresh = thresholds[rec_95_ix]
    best_thresh_metrics['prec@rec=95'] = metric_threshs['prec'][rec_95_ix]

    rec_99_ixs = np.where(np.array(metric_threshs['rec']) > 0.99)[0]
    if len(rec_99_ixs > 0):
        rec_99_ix = np.where(np.array(metric_threshs['rec']) > 0.99)[0][-1]
        rec_99_thresh = thresholds[rec_99_ix]
        best_thresh_metrics['prec@rec=99'] = metric_threshs['prec'][rec_99_ix]
    return best_thresh, best_thresh_metrics, prec_90_thresh, rec_90_thresh

def metric_thresholds_multilabel(yhat_raw, yy):
    names = ["prec", "rec", "f1"]
    metric_threshs = {label2abbrev[label]: defaultdict(list) for label in LABEL_TYPES}
    per_label_metrics = {label2abbrev[label]: defaultdict(float) for label in LABEL_TYPES}
    thresholds = np.arange(0,1,.01)[1:]
    rec_90_threshs = {}
    prec_90_threshs = {}
    for ix,label in enumerate(LABEL_TYPES):
        lname = label2abbrev[label]
        for thresh in thresholds:
            yhat = (yhat_raw[:,ix] > thresh).astype(int)
            y = yy[:,ix]
            prec = precision_score(y, yhat)
            rec = recall_score(y, yhat)
            f1 = f1_score(y, yhat)
            for name, val in zip(names, [prec, rec, f1]):
                metric_threshs[lname][name].append(val)
        best_ix = np.nanargmax(metric_threshs[lname]['f1'])
        best_thresh = thresholds[best_ix]
        best_thresh_metrics = {f'{name}_maxf1_thresh': vals[best_ix] for name, vals in metric_threshs[lname].items()}
        best_thresh_metrics['auc'] = roc_auc_score(yy[:,ix], yhat_raw[:,ix])

        rec_90_ixs = np.where(np.array(metric_threshs[lname]['rec']) > 0.90)[0]
        if len(rec_90_ixs) > 0:
            rec_90_ix = rec_90_ixs[-1]
            rec_90_thresh = thresholds[rec_90_ix]
            best_thresh_metrics['prec@rec=90'] = metric_threshs[lname]['prec'][rec_90_ix]
        else:
            rec_90_ix = 0
            rec_90_thresh = thresholds[rec_90_ix]
            highest_rec = metric_threshs[lname]['rec'][rec_90_ix]
            best_thresh_metrics['prec@rec=90'] = metric_threshs[lname]['prec'][rec_90_ix]

        prec_90_ixs = np.where(np.array(metric_threshs[lname]['prec']) > 0.90)[0]
        if len(prec_90_ixs) > 0:
            prec_90_ix = prec_90_ixs[0]
            prec_90_thresh = thresholds[prec_90_ix]
            best_thresh_metrics['rec@prec=90'] = metric_threshs[lname]['rec'][prec_90_ix]
        else:
            prec_90_ix = -1
            prec_90_thresh = thresholds[prec_90_ix]
            highest_prec = metric_threshs[lname]['prec'][prec_90_ix]
            best_thresh_metrics['rec@prec=90'] = metric_threshs[lname]['rec'][prec_90_ix]

        per_label_metrics[lname].update(best_thresh_metrics)
        rec_90_threshs[lname] = rec_90_thresh
        prec_90_threshs[lname] = prec_90_thresh
    return per_label_metrics, rec_90_threshs, prec_90_threshs

def high_rec_false_negatives(X_dv, yhat_raw, y, rec_90_thresh, out_dir, fname, label_name):
    preds = yhat_raw > rec_90_thresh
    fns = np.array(y).astype(bool) & ~preds
    fn_ixs = set(np.where(fns == True)[0])
    fn_sents = []
    with open(fname) as f:
        r = csv.DictReader(f)
        for ix, row in enumerate(r):
            if ix in fn_ixs:
                fn_sents.append(' '.join(eval(row['sentence'])))
    with open(f'{out_dir}/{label_name}_rec_90_fns.txt', 'w') as of:
        for sent in fn_sents:
            of.write(sent + "\n")
    return fn_sents

def high_prec_false_positives(X_dv, yhat_raw, y, prec_90_thresh, out_dir, fname, label_name):
    preds = yhat_raw > prec_90_thresh
    fps = ~np.array(y).astype(bool) & preds
    fp_ixs = set(np.where(fps == True)[0])
    fp_sents = []
    with open(fname) as f:
        r = csv.DictReader(f)
        for ix, row in enumerate(r):
            if ix in fp_ixs:
                fp_sents.append(' '.join(eval(row['sentence'])))
    with open(f'{out_dir}/{label_name}_prec_90_fps.txt', 'w') as of:
        for sent in fp_sents:
            of.write(sent + "\n")
    return fp_sents

def main(args):
    cvec = CountVectorizer()
    tvec = TfidfVectorizer()
    lvec = None
    if args.task == 'finegrained':
        lvec = CountVectorizer(tokenizer=lambda x: x.split(';'), lowercase=False, stop_words=[''], vocabulary=LABEL_TYPES)

    print("building train matrices")
    cX, tX, yy = build_data_and_label_matrices(args.train_fname, cvec, tvec, lvec, fit=True)
    print("building dev matrices")
    dev_fname = args.train_fname.replace('train', 'val')
    cX_dv, tX_dv, yy_dv = build_data_and_label_matrices(dev_fname, cvec, tvec, lvec)
    print("building test matrices")
    test_fname = args.train_fname.replace('train', 'test')
    cX_te, tX_te, yy_te = build_data_and_label_matrices(test_fname, cvec, tvec, lvec)

    solver = 'sag' if args.penalty == 'l2' else 'saga'
    #solver = 'saga'
    #solver = 'liblinear'
    print(f"iterations: {args.max_iter}")
    lr_clf = LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            penalty=args.penalty,
            l1_ratio=args.l1_ratio,
            solver=solver,
            )

    if args.task == 'finegrained':
        clf = OneVsRestClassifier(lr_clf, n_jobs=7)
    else:
        clf = lr_clf

    print(f"training {args.feature_type} BOW")
    X = tX if args.feature_type == 'tfidf' else cX
    X_dv = tX_dv if args.feature_type == 'tfidf' else cX_dv
    vec = tvec if args.feature_type == 'tfidf' else cvec

    clf.fit(X, yy)
    yhat = clf.predict(X_dv)
    yhat_raw = clf.predict_proba(X_dv)
    ix2word = {i:w for w,i in vec.vocabulary_.items()}

    timestamp = time.strftime('%b_%d_%H:%M:%S', time.localtime())
    out_dir = f'results/LR_{timestamp}'

    print("evaluating predictions on dev set")
    if args.task == 'finegrained':
        metrics = multilabel_eval.all_metrics(
                np.asarray(yhat),
                np.asarray(yy_dv), 
                calc_auc=True, 
                yhat_raw=yhat_raw,
                label_order=LABEL_TYPES
                )
        multilabel_eval.print_metrics(metrics, True)

        thresh, best_thresh_metrics = multilabel_eval.metrics_v_thresholds(yhat_raw, np.asarray(yy_dv))
        print(f"best threshold metrics (threshold = {thresh})")
        multilabel_eval.print_metrics(best_thresh_metrics)

        label_type_metrics = multilabel_eval.f1_per_label_type(yhat_raw, np.asarray(yy_dv), LABEL_TYPES, thresh)
        multilabel_eval.print_per_label_metrics(label_type_metrics)

        thresh_metrics, rec_90_threshs, prec_90_threshs = metric_thresholds_multilabel(yhat_raw, yy_dv)

        if out_dir is not None and not os.path.exists(out_dir):
            os.mkdir(out_dir)

        if args.print_feats:
            for ix, label in enumerate(LABEL_TYPES):
                lname = label2abbrev[label]
                print(f"###{lname}###")
                feats = clf.coef_[ix]
                top_10_feats = np.argsort(feats)[::-1][:10]
                for feat_ix in top_10_feats:
                    print(f"{ix2word[feat_ix]}: {feats[feat_ix]}")
                print()

            high_prec_fps = high_prec_false_positives(X_dv, yhat_raw[:,ix], yy_dv[:,ix], prec_90_threshs[lname], out_dir, dev_fname, lname)
            high_rec_fps = high_rec_false_negatives(X_dv, yhat_raw[:,ix], yy_dv[:,ix], rec_90_threshs[lname], out_dir, dev_fname, lname)

    else:
        thresh, best_thresh_metrics, prec_90_thresh, rec_90_thresh = metrics_v_thresholds(yhat_raw, yy_dv)
        acc, prec, rec, f1, auc = best_thresh_metrics['acc'], best_thresh_metrics['prec'], best_thresh_metrics['rec'], best_thresh_metrics['f1'], best_thresh_metrics['auc'], 
        print(f"best threshold metrics (threshold={thresh})")
        print("accuracy, precision, recall, f1, AUROC")
        print(f"{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f},{auc:.4f}")

        if args.print_feats:
            feats = clf.coef_[0]
            top_10_feats = np.argsort(feats)[::-1][:10]
            for feat_ix in top_10_feats:
                print(f"{ix2word[feat_ix]}: {feats[feat_ix]}")
            print()

        prec_at_rec_vals = ['prec@rec=90', 'prec@rec=95']
        if 'prec@rec=99' in best_thresh_metrics:
            prec_at_rec_vals.append('prec@rec=99')
        header_str = ','.join(prec_at_rec_vals)
        values_str = ','.join([f"{best_thresh_metrics[val]:.4f}" for val in prec_at_rec_vals])
        print(header_str)
        print(values_str)

        if out_dir is not None and not os.path.exists(out_dir):
            os.mkdir(out_dir)

        high_prec_fps = high_prec_false_positives(X_dv, yhat_raw[:,1], yy_dv, prec_90_thresh, out_dir, dev_fname, 'binary')
        high_rec_fns = high_rec_false_negatives(X_dv, yhat_raw[:,1], yy_dv, rec_90_thresh, out_dir, dev_fname, 'binary')
    print(f"Finished! Results at {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_fname", type=str)
    parser.add_argument("--task", choices=['binary', 'finegrained'], default='finegrained')
    parser.add_argument("--C", type=float, default=1.0, help="inverse of regularization strength")
    parser.add_argument("--penalty", choices=['l1', 'l2', 'elasticnet', 'none'],  default='l2', help="type of regularization to use")
    parser.add_argument("--l1_ratio", type=float, default=0.5, help="(for elasticnet only) relative strength of l1 reg term")
    parser.add_argument("--max_iter", type=float, default=5000, help="max number of iterations taken for solvers to converge")
    parser.add_argument("--feature_type", choices=['plain', 'tfidf'], default='plain', help="which features to use - tfidf weighted ('tfidf') or not ('plain')")
    parser.add_argument("--print_feats", action="store_true")
    args = parser.parse_args()

    main(args)
