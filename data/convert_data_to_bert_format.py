"""
This script will map between the dataset files and the input format that our
BERT script expects.
"""
import sys

import pandas as pd
import os
from tqdm import tqdm, trange
import sys

sys.path.append("../")
from CLIP import list_of_label_types
import pandas as pd
import csv
import sys

from collections import Counter


def read_dataset(path, max_sent_len=128):
    data = pd.read_csv(path, header=0)
    documents = data["tokens"].tolist()
    documents = [eval(x) for x in documents]
    tag_docs = data["labels"].tolist()
    tag_docs = [[x for x in eval(tag)] for tag in tag_docs]
    documents, tag_docs = max_len_split(documents, tag_docs, max_sent_len)
    return list(zip(documents, tag_docs))


def max_len_split(documents, tag_docs, max_len):
    ret_docs, ret_tag_docs = [], []
    for doc, tag_doc in zip(documents, tag_docs):
        new_doc, new_tag_doc = [], []
        for sent, tag_sent in zip(doc, tag_doc):
            new_sents = hard_split(truncate_token(sent), max_len)
            new_tag_sents = hard_split(tag_b_to_i(tag_sent), max_len)
            new_doc += new_sents
            new_tag_doc += new_tag_sents
        ret_docs.append(new_doc)
        ret_tag_docs.append(new_tag_doc)
    return ret_docs, ret_tag_docs


def truncate_token(sent, max_len=0):
    ret = []
    for tok in sent:
        if max_len > 0:
            ret.append(tok[:max_len])
        else:
            ret.append(tok)
    return ret


def tag_b_to_i(tag_sent):
    ret = []
    for tag in tag_sent:
        if type(tag) is list:
            inn_ret = []
            for t in tag:
                n_t = t.replace("B-", "I-").replace(" ", "-")
                ts = n_t.split("-")
                if len(ts) > 1:
                    n_t = "-".join(ts[:2])
                inn_ret.append(n_t)
            ret.append(inn_ret)
        else:
            ret.append(tag.replace("B-", "I-"))
    return ret


def hard_split(lst, max_len):
    ret = []
    for i in range(0, len(lst), max_len):
        ret.append(lst[i : i + max_len])
    return ret


def write_conll(ofname, dataset, fine_tag=None):
    with open(ofname, "w") as f:
        for doc, tag_doc in dataset:
            f.write("-DOCSTART- O\n\n")
            for sent, tag_sent in zip(doc, tag_doc):
                for tok, tag in zip(sent, tag_sent):
                    if type(tag) is list:
                        matched = "O"
                        for t in tag:
                            if fine_tag in t:
                                matched = t
                                break
                        f.write("%s %s\n" % (tok, matched))
                    else:
                        f.write("%s %s\n" % (tok, tag))
                f.write("\n")


def get_stats(dataset):
    counter = Counter()
    for _, tag_doc in dataset:
        for tag_sent in tag_doc:
            for tags in tag_sent:
                if type(tags) is list:
                    for tag in tags:
                        counter[tag] += 1
    print(counter.most_common())


def write_to_bert_format(fname, ofname, flag):
    """
    :param fname: str, original file_name
    :param ofname:  output file nmae
    :param flag: binary or finegrained
    :return:
    """
    dataset = read_dataset(fname)
    if flag == "binary":
        write_conll(ofname, dataset)
    elif flag == "stat":
        get_stats(dataset)
    else:
        write_conll(ofname, dataset, flag)


dataset_map = {
    "test": "test.binary.ks.csv",
    "train": "mimic-train.binary.bio.export.csv",
    "val": "mimic-valid.binary.bio.export.csv",
}
os.makedirs("bert_data_v1", exist_ok=True)
os.makedirs("bert_data_v1/binary", exist_ok=True)
for split in ["train", "val", "test"]:
    orig_file_name = os.path.join("final_splits/%s" % dataset_map[split])
    orig_file_name = orig_file_name.replace("finegrained", "binary")
    output_name = os.path.join("bert_data_v1/binary/%s.txt" % split)
    write_to_bert_format(orig_file_name, output_name, "binary")

for tag_name in list_of_label_types:
    os.makedirs("bert_data_v1/%s" % tag_name, exist_ok=True)
    for split in ["train", "val", "test"]:
        output_name = os.path.join("bert_data_v1/%s/%s.txt" % (tag_name, split))
        orig_file_name = os.path.join("final_splits/%s" % dataset_map[split])
        fine_tag = "I-%s" % tag_name
        write_to_bert_format(orig_file_name, output_name, fine_tag)
