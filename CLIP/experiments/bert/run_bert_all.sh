for model in  bert-base-cased emilyalsentzer/Bio_ClinicalBERT
  do
    for i in 3 5 10
    do
      bash run_bert.sh $i  binary $model outputs_binary ""
  done
  done

for model in bert-base-cased emilyalsentzer/Bio_ClinicalBERT
do
  for i in 3 5 10
  do
    for j in Case Medication Other Procedure Appointment Lab Imaging
    do
      # Replace this with the latest checkpoint from the binary experiments
      bash run_bert.sh $i $j $model outputs_finegrained outputs_binary/bert-binary-model-epo$i//checkpoint-750
    done
  done
done