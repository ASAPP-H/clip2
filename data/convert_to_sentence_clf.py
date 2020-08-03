"""
Input a finegrained file, this will output both finegrained and binary
"""
import argparse
import csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("fname", type=str)
args = parser.parse_args()

f = open(args.fname)
r = csv.DictReader(f)
sentence_num_spans = []
total_sentences = 0
doc_num_sentences = Counter()
doc_num_span_sentences = Counter()
doc_frac_sentences_span = {}
of = open(args.fname.replace('.csv', '.sentclf.csv'), 'w')
bof = open(args.fname.replace('.csv', '.sentclf.csv').replace('finegrained', 'binary'), 'w')
w = csv.writer(of)
bw = csv.writer(bof)
w.writerow(['doc_id', 'sentence', 'labels'])
bw.writerow(['doc_id', 'sentence', 'labels'])
for row_ix, row in tqdm(enumerate(r)):
    tokens = eval(row['tokens'])
    labels = eval(row['labels'])
    if 'document_id' in row:
        doc_id = row['document_id']
    for sent_ix, (tks, lbls) in enumerate(zip(tokens, labels)):
        span_types = set()
        for ix, (tk, lbl) in enumerate(zip(tks, lbls)):
            span_types.update(set(lbl))
        if 'O' in span_types:
            span_types.remove('O')
        if len(span_types) > 0:
            sentence_num_spans.append(len(span_types))
        w.writerow([doc_id, tks, list(span_types)])
        bw.writerow([doc_id, tks, len(span_types) > 0])
        total_sentences += 1
        doc_num_sentences[doc_id] += 1
        doc_num_span_sentences[doc_id] += len(span_types)
    doc_frac_sentences_span[doc_id] = doc_num_span_sentences[doc_id] / doc_num_sentences[doc_id]

of.close()
bof.close()
plt.hist(doc_frac_sentences_span.values(), bins=10)

print(f"avg # spans, per sentence with a span: {np.mean(sentence_num_spans)}")
num_multiple = np.mean([x > 1 for x in sentence_num_spans])
print(f"num sentences with multiple spans in it: {num_multiple}")
num_mt2 = np.mean([x > 2 for x in sentence_num_spans])
print(f"num sentences with more than 2 spans in it: {num_mt2}")
print(f"frac of sentences that have any spans: {len(sentence_num_spans) / total_sentences}")
