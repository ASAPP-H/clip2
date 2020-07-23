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

clf = OneVsRestClassifier(LogisticRegression(C=args.C, solver='sag'))#, n_jobs=-1)
print("training plain BOW")
clf.fit(cX, yy)
yhat = clf.predict(cX_dv)
yhat_raw = clf.predict_proba(cX_dv)

metrics = multilabel_eval.all_metrics(
        np.asarray(yhat.todense()),
        np.asarray(yy_dv.todense()), 
        k=2, 
        calc_auc=True, 
        yhat_raw=yhat_raw,
        label_order=label_types
        )
multilabel_eval.print_metrics(metrics, True)

print("training tfidf BOW")
clf.fit(tX, yy)
yhat = clf.predict(tX_dv)
yhat_raw = clf.predict_proba(tX_dv)

metrics = multilabel_eval.all_metrics(
        np.asarray(yhat.todense()),
        np.asarray(yy_dv.todense()), 
        k=2, 
        calc_auc=True, 
        yhat_raw=yhat_raw,
        label_order=label_types
        )
multilabel_eval.print_metrics(metrics, True)
