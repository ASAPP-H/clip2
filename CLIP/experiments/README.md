
Before running models, make sure the environment variable `CLIP_DIR` has the path to the current directory. 

For example, if the path to the current directory is `x/y/z/CLIP`, you would run 
`export CLIP_DIR = /x/y/z/CLIP/`. 
## Running BERT
To run the BERT model, first, do:
    `chmod 777 bert/run_bert_all.sh`. 
   
   Then, run ` ./bert/run_bert_all.sh `. You will need to run this on a GPU, preferably a V100 or P40.
   
This will train your dataset for both binary and finegrained experiments and store the results and logs into 
`bert/outputs/` directory.

In order to evaluate the dataset, if your BERT predictions are under the main experiment directory `bert/outputs`, then call 
` python merge_and_evaluate_bert.py  bert/outputs_finegrained finegrained.pkl fine` to get the results for the finegrained cast of the task. 

To evaluate BERT on the binary cast, run 
` python merge_and_evaluate_bert.py  bert/outputs_binary/bert-binary-model-epo3/test_predictions.txt binary.pkl binary`. 

The F1 scores and breakdown will be saved under `bert/outputs` as `fine_metrics.pkl` (for finegrained cast) and `bin_metrics.pkl` (for binary cast).

### Running different BERT models
If you would like to subsitutte in another BERT model, it is relatively easy to do so. The code we use is adapted from the Huggingface 
implementation of a tagger. To switch to another model, go to `bert/run_bert_all.sh`, and replace the model names listed in the first line 
with your model of choice. Make sure that that model is supported by the Huggingface Transformers library. 

## Running BiLSTM

The tagger code is adapted from https://github.com/yi-asapp/seq-tagger. 

We use BioWordVec, so please first go to their repo and get the: version. Then, move it to tagger/embeddings

In order to run, first run 
`cd tagger`

Then run the below:
`./finetune_binary.sh` to run the binary experiments, before running `./finetune_finegrained.sh`.

Then, to evaluate the BiLSTM dataset on the binary cast, run 
`python merge_and_evaluate_bilstm.py tagger/outputs_binary/binary0.005/  tagger/outputs_binary/binary0.005/  binary`. 

To evaluate the BiLSTM dataset on the finegrained cast, run 
`python merge_and_evaluate_bilstm.py tagger/outputs_finegrained  tagger/outputs_finegrained  fine`. 

The metrics will then be saved as .pkl files in the respective directories (for tagger outputs_finegrained, the finegrained metric
will be saved as tagger/outputs_finegrained/fine_metrics.pkl for instance). To render the metrics in an easily digestible format, 
run the View Results Jupyter notebook in this directory.
