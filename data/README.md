
## Data Generation Instructions
Please first put the MIMIC-III and i2b2 data into the source directory. 
Then, run `python data_generation.py`  to generate the dataset, which maps the token-level offsets to the respective subsets of the i2b2 and MIMIC datasets. This will save the data into the `processed` directory.


If you would like to run the BERT experiments, additionally run `python convert_data_to_bert_format.py` after running `python data_generation.py`, which will save the data in the structure that the BERT experimentation code expects. This data will be saved into `bert_data` folder. 


To explore the data, you can also run the code in the examine_data Jupyter notebook. k

## Using different Sentence Tokenizers

We support a variety of different sentence segmentation tokenizers that have been shown to work with biomedical corpus (see [here](https://github.com/ypruksachatkun-asapp/CLIP/blob/master/data/tokenizers.py#L43), making it 
easy to swap out and try a variety of different tokenizers on your corpus. To segment your text into sentences, use [this function](https://github.com/ypruksachatkun-asapp/CLIP/blob/master/data/utils.py#L5). 

It is also possible to try different word tokenizers to split sentences into tokens, which you can do with the use of our `WordTokenizer` class. Given a specific type of WordTokenizer `wt_type`, you can use WordTokenizers as below
`word_tokenizer = WordTokenizer.from_type(wt_type)`

`tokens = word_tokenizer.tokenize(sentence)`
