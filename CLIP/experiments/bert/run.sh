
# Convert csv data files to conll files

python csv_to_conll.py ../../data/gold_standard/final_splits/train-ks.binary.csv data/binary/train.txt binary
python csv_to_conll.py ../../data/gold_standard/final_splits/val.binary.ks.csv data/binary/dev.txt binary
python csv_to_conll.py ../../data/gold_standard/final_splits/test.binary.ks.csv data/binary/test.txt binary

for tag in Appointment Case	Imaging	Lab	Medication Other Procedure
do
    python csv_to_conll.py ../../data/gold_standard/final_splits/train-ks.finegrained.csv data/$tag/train.txt $tag
    python csv_to_conll.py ../../data/gold_standard/final_splits/val.finegrained.ks.csv data/$tag/dev.txt $tag
    python csv_to_conll.py ../../data/gold_standard/final_splits/test.finegrained.ks.csv data/$tag/test.txt $tag
done


# Model training
bash run_bert_all.sh


# copy fine-grained gt files
for tag in Appointment Case Imaging Lab Medication Other Procedure
do
    cp data/$tag/test.txt data/finegrained/$tag.txt
done

# copy pred files
cp outputs/bert-binary-model-epo2/test_predictions.txt predictions/bert/binary.txt
cp outputs/biobert-binary-model-epo2/test_predictions.txt predictions/biobert/binary.txt
for tag in Appointment Case Imaging Lab Medication Other Procedure
do
    cp outputs/bert-$tag-model-epo2/test_predictions.txt predictions/bert/finegrained/$tag.txt
    cp outputs/biobert-$tag-model-epo2/test_predictions.txt predictions/biobert/finegrained/$tag.txt
done

# pred files to pkl
python conll_to_pkl.py ../../data/gold_standard/final_splits/test.binary.ks.csv data/binary/test.txt predictions/bert/binary.txt predictions/bert/binary.pkl binary
python conll_to_pkl.py ../../data/gold_standard/final_splits/test.binary.ks.csv data/binary/test.txt predictions/biobert/binary.txt predictions/biobert/binary.pkl binary
python conll_to_pkl.py ../../data/gold_standard/final_splits/test.finegrained.ks.csv data/finegrained predictions/bert/finegrained predictions/bert/finegrained/finegrained.pkl fine
python conll_to_pkl.py ../../data/gold_standard/final_splits/test.finegrained.ks.csv data/finegrained predictions/biobert/finegrained predictions/biobert/finegrained/finegrained.pkl fine
