#!/bin/bash
python bow.py ../../../data/processed_syntok_nltk_072920/MIMIC_train_finegrained.sentclf.csv --C 1.0 --task binary
python bow.py ../../../data/processed_syntok_nltk_072920/MIMIC_train_finegrained.sentclf.csv --C 1.0 --task finegrained
python bow.py ../../../data/processed_syntok_nltk_072920/MIMIC_train_finegrained.sentclf.csv --C 1.0 --task binary --feature_type tfidf
python bow.py ../../../data/processed_syntok_nltk_072920/MIMIC_train_finegrained.sentclf.csv --C 1.0 --task finegrained --feature_type tfidf
