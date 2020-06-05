export NUM_EPOCHS=$1
export GENRE=$2 # binary Case Medication Other Procedure Appointment Lab Imaging
export BERT_MODEL=$3  # bert-base-cased emilyalsentzer/Bio_ClinicalBERT

export MAX_LENGTH=256
export BATCH_SIZE=32
export SAVE_STEPS=1000
export SEED=1
export MODEL_NAME_OR_PATH=${5:=\"\"}
export DATA_DIR=bert_data/$2
echo $MODEL_NAME_OR_PATH
if [ $BERT_MODEL = bert-base-cased ]
then
  export OUTPUT_DIR=$4/bert-$2-model-epo$1
else
  export OUTPUT_DIR=$4/biobert-$2-model-epo$1
fi
cat $DATA_DIR/train.txt $DATA_DIR/val.txt $DATA_DIR/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_DIR/labels.txt
if [ $MODEL_NAME_OR_PATH == ""]
then
   python3 run_ner.py --data_dir $DATA_DIR/ \
--model_type bert \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
else
python3 run_ner.py --data_dir $DATA_DIR/ \
--model_type bert \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--model_name_or_path $MODEL_NAME_OR_PATH \
--do_train \
--do_eval \
--do_predict
fi
