import sys
sys.path.append('../tagger')
sys.path.append('../../')
from evaluation.evaluation import *


def get_golds_preds(gfname, pfname):
    glists = get_tags(gfname)
    plists = get_tags(pfname)
    #_, macro_f1 = get_evaluation_multiclass_for_documents(plists, glists, ["followup"], "i2b2")
    _, macro_f1 = get_evaluation_multiclass_for_documents(plists, glists, ["followup"], "mimic")
    print (macro_f1)


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

if __name__ == "__main__":
    gfname, pfname = sys.argv[1:]
    get_golds_preds(gfname, pfname)
