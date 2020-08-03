from train import train_model
from utils import *
import os
import sys

pwd = os.environ.get('CLIP_DIR')
DATA_DIR = "%s/data/final_splits/" % pwd
exp_name = "non_multilabel"

run_name = "sentence_structurel_with_crf"
#train_file_name = "MIMIC_train_binary.csv"
#dev_file_name = "MIMIC_val_binary.csv"
#test_file_name = "test_binary.csv"
train_file_name = "mimic-train.binary.bio.export.csv"
dev_file_name = "mimic-valid.binary.bio.export.csv"
test_file_name = "mimic-test.binary.bio.export.csv"

exp_name = "outputs_binary"
train = read_sentence_structure(os.path.join(DATA_DIR, train_file_name))
dev = read_sentence_structure(os.path.join(DATA_DIR, dev_file_name))
test = read_sentence_structure(os.path.join(DATA_DIR, test_file_name))
run_name = "binary"


def main(args):
    train_model(
        train,
        dev,
        test,
        args[0],
        exp_name,
        use_crf=True,
        learning_rate=float(args[1]),
        epochs=int(args[2]),
        writer_preds_freq=10,
        embeddings_type="BioWord",
        list_of_possible_tags=["followup"],
        embeddings_path="%s/CLIP/experiments/tagger/embeddings" % pwd,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
