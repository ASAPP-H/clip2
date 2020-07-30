import argparse
import csv

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import multilabel_eval

parser = argparse.ArgumentParser()
parser.add_argument("train_fname", type=str)
parser.add_argument("--C", type=float, default=1.0, help="inverse of regularization strength")
parser.add_argument("--penalty", choices=['l1', 'l2', 'elasticnet', 'none'],  default='l2', help="type of regularization to use")
parser.add_argument("--l1_ratio", type=float, default=0.5, help="(for elasticnet only) relative strength of l1 reg term")
parser.add_argument("--max_iter", type=float, default=100, help="max number of iterations taken for solvers to converge")
parser.add_argument("--feature_type", choices=['plain', 'tfidf'], default='plain', help="which features to use - tfidf weighted ('tfidf') or not ('plain')")
args = parser.parse_args()

label_types = ['I-Imaging-related followup',
 'I-Appointment-related followup',
 'I-Medication-related followups',
 'I-Procedure-related followup',
 'I-Lab-related followup',
 'I-Case-specific instructions for patient',
 'I-Other helpful contextual information',
 ]

def build_data_and_label_matrices(fname, cvec, tvec, lvec, fit=False):
    corpus = []
    labels = []
    with open(fname) as f:
        r = csv.DictReader(f)
        for row in r:
            corpus.append(' '.join(eval(row['sentence'])))
            labels.append(';'.join(eval(row['labels'])))

    if fit:
        cX = cvec.fit_transform(corpus)
        tX = tvec.fit_transform(corpus)
        # create multi-label matrix
        yy = lvec.fit_transform(labels)
    else:
        cX = cvec.transform(corpus)
        tX = tvec.transform(corpus)
        yy = lvec.transform(labels)

    return cX, tX, yy

cvec = CountVectorizer()
tvec = TfidfVectorizer()
lvec = CountVectorizer(tokenizer=lambda x: x.split(';'), lowercase=False, stop_words=[''], vocabulary=label_types)

print("building train matrices")
cX, tX, yy = build_data_and_label_matrices(args.train_fname, cvec, tvec, lvec, fit=True)
print("building dev matrices")
cX_dv, tX_dv, yy_dv = build_data_and_label_matrices(args.train_fname.replace('train', 'val'), cvec, tvec, lvec)
print("building test matrices")
cX_te, tX_te, yy_te = build_data_and_label_matrices(args.train_fname.replace('train', 'test'), cvec, tvec, lvec)

solver = 'sag' if args.penalty == 'l2' else 'saga'
#solver = 'saga'
#solver = 'liblinear'
print(f"iterations: {args.max_iter}")
clf = OneVsRestClassifier(LogisticRegression(
    C=args.C,
    max_iter=args.max_iter,
    penalty=args.penalty,
    l1_ratio=args.l1_ratio,
    solver=solver,
    ))#, n_jobs=-1)
print(f"training {args.feature_type} BOW")
if args.feature_type == 'tfidf':
    clf.fit(tX, yy)
    yhat = clf.predict(tX_dv)
    yhat_raw = clf.predict_proba(tX_dv)
    ix2word = {i:w for w,i in tvec.vocabulary_.items()}
else:
    clf.fit(cX, yy)
    yhat = clf.predict(cX_dv)
    yhat_raw = clf.predict_proba(cX_dv)
    ix2word = {i:w for w,i in cvec.vocabulary_.items()}

metrics = multilabel_eval.all_metrics(
        np.asarray(yhat.todense()),
        np.asarray(yy_dv.todense()), 
        calc_auc=True, 
        yhat_raw=yhat_raw,
        label_order=label_types
        )
multilabel_eval.print_metrics(metrics, True)

thresh, best_thresh_metrics = multilabel_eval.metrics_v_thresholds(yhat_raw, np.asarray(yy_dv.todense()))
print(f"best threshold metrics (threshold = {thresh})")
multilabel_eval.print_metrics(best_thresh_metrics)

label_type_metrics = multilabel_eval.f1_per_label_type(yhat_raw, np.asarray(yy_dv.todense()), label_types, thresh)
multilabel_eval.print_per_label_metrics(label_type_metrics)

for ix, label_name in enumerate(label_types):
    print(f"###{label_name}###")
    feats = clf.coef_[ix]
    top_10_feats = np.argsort(feats)[::-1][:10]
    for feat_ix in top_10_feats:
        print(f"{ix2word[feat_ix]}: {feats[feat_ix]}")
    print()
