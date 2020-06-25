for model in bert-base-cased emilyalsentzer/Bio_ClinicalBERT;
do
  for i in 3;
  do
    for j in binary; #Case Medication Other Procedure Appointment Lab Imaging
    do
      bash run_bert.sh $i $j $model
    done
  done
done
