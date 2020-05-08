"""
This script gets all the BERT predictions and merges it to the input format expected
by our evaluation code.
"""
import pandas as pd
import sys
import os
import pickle

sys.path.append("../../")
from CLIP import list_of_label_types
from evaluate import evaluation

pwd = os.environ.get("CLIP_DIR")


def get_docids(path):
    data = pd.read_csv(path, header=0)
    doc_ids = data["document_id"].tolist()
    return doc_ids


def get_tags(fname):
    ret = []
    with open(fname) as f:
        temp = []
        for line in f:
            if line.startswith("-DOCSTART-"):
                if len(temp) > 0:
                    ret.append(temp)
                temp = []
            else:
                if line.strip() == "":
                    continue
                else:
                    temp.append(line.split()[1])
        if temp:
            ret.append(temp)
    return ret


def binary_preds(doc_ids, gfname, pfname, ofname):
    glists = get_tags(gfname)
    plists = get_tags(pfname)
    with open(ofname, "wb") as f:
        pickle.dump([doc_ids, glists, plists], f, protocol=pickle.HIGHEST_PROTOCOL)


def fine_preds(doc_ids, gdir, pdir, ofname):
    glists = merge_lists(gdir)
    pdir = os.path.join(pdir)
    plists = merge_lists(pdir, gold=False)
    with open(ofname, "wb") as f:
        pickle.dump([doc_ids, glists, plists], f, protocol=pickle.HIGHEST_PROTOCOL)


def get_evaluation(output_fn, output_dir, gold_path, flag):
    doc_ids, labels, preds = pickle.load(open(output_fn, "rb"))
    pred_rows = []
    gold = pd.read_csv(gold_path)
    label_rows = []
    i2b2_docs = gold[gold["source"] == "i2b2"]["document_id"].unique()
    for i in range(len(doc_ids)):
        if doc_ids[i] in i2b2_docs:
            source = "i2b2"
        else:
            source = "mimic"
        pred_rows.append((preds[i], source))
        label_rows.append((labels[i], source))
    if flag == "fine":
        label_types = list_of_label_types
    else:
        label_types = ["I-followup"]
    f1 = evaluation.get_evaluation(pred_rows, label_rows, label_types)
    pickle.dump(f1, open(os.path.join(output_dir, "%s_metrics.pkl" % flag), "wb"))
    return f1


def merge_lists(path, gold=True):
    # what is happening here?
    fnames = os.listdir(path)
    to_merge_lists = []
    if gold:
        file_name = "test.txt"
    else:
        file_name = "test_predictions.txt"
    for fname in fnames:
        # you have test.txt
        if "DS" in fname:
            continue
        to_merge_lists.append(get_tags(path + "/" + fname + "/" + file_name))
    merged_lists = [[[] for _ in doc] for doc in to_merge_lists[0]]
    for to_list in to_merge_lists:
        for i, doc in enumerate(to_list):
            for j, tag in enumerate(doc):
                if tag == "O":
                    try:
                        if merged_lists[i][j] == []:
                            merged_lists[i][j] = ["O"]
                    except:
                        import pdb

                        pdb.set_trace()
                else:
                    if merged_lists[i][j] == ["O"]:
                        merged_lists[i][j] = [tag]
                    else:
                        merged_lists[i][j].append(tag)
    return merged_lists


if __name__ == "__main__":
    pred_dir, output_dir, flag = sys.argv[1:]
    doc_ids = get_docids("%s/data/processed/test_finegrained.csv" % pwd)
    gold_file = "%s/data/processed/test_finegrained.csv" % pwd
    if flag == "binary":
        gold_file = "bert/bert_data/binary/test.txt"
        binary_preds(doc_ids, gold_file, pred_dir, output_dir)
        pred_dir = "/".join(pred_dir.split("/")[:-1])
    elif flag == "fine":
        directory = pred_dir.split("/")[
            :-3
        ]  # Get only the directory that contains the Case, etc etc ones.
        gdir = "bert/bert_data"
        fine_preds(doc_ids, gdir, pred_dir, output_dir)
    if flag == "fine":
        flag = "finegrained"
    gold_path = os.path.join("%s" % pwd, "data/processed/test_%s.csv" % flag)
    get_evaluation(output_dir, pred_dir, gold_path, flag)
