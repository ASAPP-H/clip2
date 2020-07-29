import argparse
import csv
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("fname", type=str)
args = parser.parse_args()

f = open(args.fname)
r = csv.DictReader(f)
spans = []
span_full_sentences_only = []
span_num_sentences = []
span_sentences = []
span_docids = []
doc_ids = []
for row_ix, row in tqdm(enumerate(r)):
    in_span = False
    cur_type = ''
    cur_span = []
    cur_num_sentences = 1
    cur_span_sentences = []
    cur_full_sentences_only = True
    tokens = eval(row['tokens'])
    labels = eval(row['labels'])
    if 'document_id' in row:
        doc_ids.append(row['document_id'])
        doc_id = row['document_id']
    for sent_ix, (tks, lbls) in enumerate(zip(tokens, labels)):
        for ix, (tk, lbl) in enumerate(zip(tks, lbls)):
            if lbl.startswith('B'):
                # start new span
                if in_span:
                    # add current span first
                    spans.append(cur_span)
                    span_full_sentences_only.append(cur_full_sentences_only)
                    span_num_sentences.append(cur_num_sentences)
                    span_sentences.append(cur_span_sentences)
                    span_docids.append(doc_id)
                # make new span
                cur_span = [tk]
                cur_num_sentences = 1
                cur_full_sentences_only = True
                cur_span_sentences = [' '.join(tks)]
                in_span = True
                cur_type = lbl.split('-')[1]
                if ix > 0:
                    # set full sentence flag and fill in span to show length of sentence
                    cur_full_sentences_only = False
                    for ix_ in range(0, ix)[::-1]:
                        cur_span.insert(0, (tks[ix_], '-NOT-IN-SPAN-'))
            elif lbl.startswith('I'):
                in_span = True
                # if different label type
                if lbl.split('-')[1] != cur_type and cur_type != '':
                    if ix > 0:
                        # this means a new followup starts directly after another one, in the middle of a sentence
                        cur_full_sentences_only = False
                        # fill in span to show length
                        for ix_ in range(ix, len(tks)):
                            cur_span.append((tks[ix_], '-NOT-IN-SPAN-'))
                        # add current span
                        spans.append(cur_span)
                        span_full_sentences_only.append(cur_full_sentences_only)
                        span_num_sentences.append(cur_num_sentences)
                        span_sentences.append(cur_span_sentences)
                        span_docids.append(doc_id)

                        cur_span = []
                        # fill in new span with filler too
                        for ix_ in range(0, ix)[::-1]:
                            cur_span.insert(0, (tks[ix_], '-NOT-IN-SPAN-'))
                        cur_num_sentences = 1
                        cur_full_sentences_only = False
                    else:
                        # different label type, starting at start of new sentence
                        cur_span = []
                        cur_num_sentences = 1
                        cur_full_sentences_only = True
                    cur_span_sentences = [' '.join(tks)]
                    cur_type = lbl.split('-')[1]
                else:
                    # same or new label type
                    if ix == 0:
                        if cur_type != '':
                            # same label type
                            # span crosses sentence boundary, so add stuff to that effect
                            cur_span.append("-NEW-SENTENCE-")
                            cur_num_sentences += 1
                            cur_span_sentences.append(' '.join(tks))
                        else:
                            #print("new label type at start of sentence. should have been covered by prev condition")
                            cur_span_sentences.append(' '.join(tks))
                    else:
                        if cur_type == "":
                            # new label in middle of sentence
                            cur_full_sentences_only = False
                            for ix_ in range(0, ix)[::-1]:
                                cur_span.insert(0, (tks[ix_], '-NOT-IN-SPAN-'))
                            cur_span_sentences.append(' '.join(tks))
                cur_span.append(tk)
                cur_type = lbl.split('-')[1]
            elif lbl == 'O':
                if in_span:
                    if ix > 0:
                        # span ended in middle of sentence, so set flag and add filler
                        cur_full_sentences_only = False
                        for ix_ in range(ix, len(tks)):
                            cur_span.append((tks[ix_], '-NOT-IN-SPAN-'))
                    # add current span
                    spans.append(cur_span)
                    span_full_sentences_only.append(cur_full_sentences_only)
                    span_num_sentences.append(cur_num_sentences)
                    span_sentences.append(cur_span_sentences)
                    span_docids.append(doc_id)

                    cur_span = []
                    cur_num_sentences = 1
                    cur_full_sentences_only = True
                    cur_span_sentences = []
                cur_type = ''
                in_span = False
    if len(cur_span) > 0:
        spans.append(cur_span)
        span_full_sentences_only.append(cur_full_sentences_only)
        span_num_sentences.append(cur_num_sentences)
        span_sentences.append(cur_span_sentences)
        span_docids.append(doc_id)
    #import pdb.set_trace()

spans_per_docid = Counter(span_docids)
print(f"num spans: {len(spans)}")
print(f"% spans with full sentences only: {sum(span_full_sentences_only) / len(spans)}")
print(f"% spans with one sentence only: {sum(np.array(span_num_sentences) == 1) / len(spans)}")
print(f"avg # sentences per span: {np.mean(span_num_sentences)}")
fig, axs = plt.subplots(1,3)
axs[0].hist(span_num_sentences, bins=max(span_num_sentences))
axs[0].set_xlabel('number of sentences a span appears in')

spans_per_all_docid = np.array([spans_per_docid[did] for did in doc_ids])
avg_spans_per_doc = np.mean(spans_per_all_docid)
print(f"avg # spans per doc: {avg_spans_per_doc}")
axs[1].hist(spans_per_all_docid, bins=20)
axs[1].set_xlabel('number of spans in a document')

fname = args.fname.replace('binary', 'finegrained')
f = open(fname)
r = csv.DictReader(f)
sentence_num_spans = []
total_sentences = 0
doc_num_sentences = Counter()
doc_num_span_sentences = Counter()
doc_frac_sentences_span = {}
doc_spans = defaultdict(list)
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
            doc_spans[doc_id].append(tks)
        total_sentences += 1
        doc_num_sentences[doc_id] += 1
        doc_num_span_sentences[doc_id] += len(span_types)
    doc_frac_sentences_span[doc_id] = doc_num_span_sentences[doc_id] / doc_num_sentences[doc_id]

import pandas as pd
df = pd.read_csv('MIMIC/source/NOTEEVENTS.csv')
import pdb; pdb.set_trace()

print(f"avg # spans, per sentence with a span: {np.mean(sentence_num_spans)}")
num_multiple = np.mean([x > 1 for x in sentence_num_spans])
print(f"num sentences with multiple spans in it: {num_multiple}")
num_mt2 = np.mean([x > 2 for x in sentence_num_spans])
print(f"num sentences with more than 2 spans in it: {num_mt2}")
print(f"frac of sentences that have any spans: {len(sentence_num_spans) / total_sentences}")
axs[2].hist(doc_frac_sentences_span.values(), bins=50)
axs[2].set_xlabel('fraction of sentences which contain a span, per document')
plt.show()

import sys; sys.exit(0)
with open('blah.bio', 'w') as of:
    for span, fso, numsents, doc_id in zip(spans, span_full_sentences_only, span_num_sentences, span_docids):
        fso_str = 'full sentences only' if fso else 'partial sentences'
        of.write(f'\nnew span: {numsents} sentences, {fso_str}. Doc id: {doc_id}\n')
        for word in span:
            if isinstance(word, tuple):
                of.write(f'{word[0]} O\n')
            else:
                of.write(f'{word} I-followup\n')
            #import pdb; pdb.set_trace()
