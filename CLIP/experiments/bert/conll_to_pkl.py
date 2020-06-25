import pandas as pd
import csv
import sys
import os
import pickle


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
                if line.strip() == "": continue
                else:
                    temp.append(line.split()[1])
        if temp: ret.append(temp)
    return ret

def binary_preds(doc_ids, gfname, pfname, ofname):
    glists = get_tags(gfname)
    plists = get_tags(pfname)
    with open(ofname, 'wb') as f:
        pickle.dump([doc_ids, glists, plists], f, protocol=pickle.HIGHEST_PROTOCOL)

def fine_preds(doc_ids, gdir, pdir, ofname):
    glists = merge_lists(gdir)
    plists = merge_lists(pdir)
    with open(ofname, 'wb') as f:
        pickle.dump([doc_ids, glists, plists], f, protocol=pickle.HIGHEST_PROTOCOL)

def merge_lists(path):
    fnames = os.listdir(path)
    to_merge_lists = []
    for fname in fnames:
        to_merge_lists.append(get_tags(path+'/'+fname))
    merged_lists = [[[] for _ in doc] for doc in to_merge_lists[0]]
    for to_list in to_merge_lists:
        for i, doc in enumerate(to_list):
            for j, tag in enumerate(doc):
                if tag == 'O':
                    if merged_lists[i][j] == []:
                        merged_lists[i][j] = ['O']
                else:
                    if merged_lists[i][j] == ['O']:
                        merged_lists[i][j] = [tag]
                    else:
                        merged_lists[i][j].append(tag)
    return merged_lists



if __name__ == "__main__":
    path, gfname, pfname, ofname, flag = sys.argv[1:]
    doc_ids = get_docids(path)
    if flag == 'binary':
        binary_preds(doc_ids, gfname, pfname, ofname)
    elif flag == 'fine':
        fine_preds(doc_ids, gfname, pfname, ofname)

