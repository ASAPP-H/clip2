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
sentence_num_spans = []
total_sentences = 0
doc_num_sentences = Counter()
doc_num_span_sentences = Counter()
doc_frac_sentences_span = {}
span_frac_of_sentence = []
of = open('low-frac-examples.txt', 'w')
for row_ix, row in tqdm(enumerate(r)):
    tokens = eval(row['tokens'])
    labels = eval(row['labels'])
    if 'document_id' in row:
        doc_id = row['document_id']
    for sent_ix, (tks, lbls) in enumerate(zip(tokens, labels)):
        span_types = set()
        span_type_numchars = Counter()
        span_toks = defaultdict(list)
        for ix, (tk, lbl) in enumerate(zip(tks, lbls)):
            span_types.update(set(lbl))
            for l in lbl:
                span_type_numchars[l] += len(tk) + 1
                span_toks[l].append(tk)
        if 'O' in span_types:
            span_types.remove('O')
        if len(span_types) > 0:
            sentence_num_spans.append(len(span_types))
        del span_type_numchars['O']
        for typ, numchars in span_type_numchars.items():
            #import pdb; pdb.set_trace()
            frac = (numchars - 1) / len(' '.join(tks))
            span_frac_of_sentence.append(frac)
            if frac < 0.5:
                print(f"##### SPAN ({typ}) ({frac:.3f})#####\n{' '.join(span_toks[typ])}\n##### SENTENCE #####\n{' '.join(tks)}\n\n")
                of.write(f"##### SPAN ({typ}) ({frac:.3f})#####\n{' '.join(span_toks[typ])}\n##### SENTENCE #####\n{' '.join(tks)}\n\n")
        total_sentences += 1
        doc_num_sentences[doc_id] += 1
        doc_num_span_sentences[doc_id] += len(span_types)
    doc_frac_sentences_span[doc_id] = doc_num_span_sentences[doc_id] / doc_num_sentences[doc_id]

of.close()
print(f"mean frac of sentence: {np.mean(span_frac_of_sentence)}")
print(f"frac complete sentence: {np.mean([f == 1.0 for f in span_frac_of_sentence])}")
plt.hist(span_frac_of_sentence, bins=50)
plt.xlabel('fraction of the sentence that a span covers, in characters')
plt.show()
#plt.hist(doc_frac_sentences_span.values(), bins=10)

print(f"avg # spans, per sentence with a span: {np.mean(sentence_num_spans)}")
num_multiple = np.mean([x > 1 for x in sentence_num_spans])
print(f"num sentences with multiple spans in it: {num_multiple}")
num_mt2 = np.mean([x > 2 for x in sentence_num_spans])
print(f"num sentences with more than 2 spans in it: {num_mt2}")
print(f"frac of sentences that have any spans: {len(sentence_num_spans) / total_sentences}")
