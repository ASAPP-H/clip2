#!/bin/bash
# replace with path to mimic train set
python neural_baselines.py ../../../data/processed_syntok_nltk/MIMIC_train_finegrained.sentclf.csv cnn --vocab_file vocab.txt --task binary
python neural_baselines.py ../../../data/processed_syntok_nltk/MIMIC_train_finegrained.sentclf.csv cnn --vocab_file vocab.txt --task finegrained
