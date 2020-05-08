This repository houses all related to CLIP: A Dataset for Extracting Clinical Follow-ups from Discharge Summaries. 

First, you need access to MIMIC-III (https://mimic.physionet.org/gettingstarted/access/) and i2b2-2010 (https://www.i2b2.org/NLP/Relations/). Once you have received access, download the NOTEEVENTS.csv file from MIMIC-III and place it under data/mimic/source. For i2b2, download the "Training Data: Concept assertion relation training data" file from the "2010 Relations Challenge Downloads" tab in DBMI and place it under data/i2b2/source. respectively.  

# Quickstart
First, make sure you have conda installed. Then, run: 
`conda create -n CLIP python=3.7`. 
Then, do `conda activate CLIP`, and finally `pip install -r requirements.txt`. 
Then, to generate the CLIP dataset, do the below. 

`cd data` and follow the instructions there. 
Then, if you are running BERT experiments, run 

`mv data/bert_data CLIP/experiments/bert/` to move the BERT-preprocessed data over into the experiment folder. 

Then, to run the BERT and BiLSTM experiments, go to `CLIP/experiments` and follow the instructions there.

# Features
Other than the experiment and data generation code, we also offer a preprocessing tool for swapping out sentence and word tokenizers
in an easy way, so that people can try various tokenizers on their data seamlessly. Go to the `data` directory to learn more. 


# License 
This package is released under the MIT License. 
