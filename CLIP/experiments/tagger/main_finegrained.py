from train import train_model
from utils import *
import os
import sys

pwd = os.environ.get('"CLIP_DIR')
DATA_DIR = "%s/data/processed/" % pwd
exp_name = "outputs_finegrained"

train_file_name = "MIMIC_train_finegrained.csv"
dev_file_name = "MIMIC_val_finegrained.csv"
test_file_name = "test_finegrained.csv"


def main(args):
    label_type = args[3]
    train = read_sentence_structure(os.path.join(DATA_DIR, train_file_name), label_type)
    dev = read_sentence_structure(os.path.join(DATA_DIR, dev_file_name), label_type)
    test = read_sentence_structure(os.path.join(DATA_DIR, test_file_name), label_type)
    run_name = label_type.split("-")[0]
    train_model(
        train,
        dev,
        test,
        run_name,
        exp_name,
        use_crf=True,
        learning_rate=float(args[0]),
        epochs=int(args[1]),
        write_preds_freqQ=10,
        list_of_possible_tags=[label_type],
        load_checkpoint="",
        embeddings_type="BioWord",
        embeddings_path="%s/CLIP/experiments/tagger/embeddings" % pwd,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
