# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import dataclasses
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, GlueDataset
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from bert_model import BertForSequenceMultilabelClassification, BertCNNForSequenceMultilabelClassification
from constants import *
import multilabel_eval
from neural_baselines import SentDataset

logger = logging.getLogger(__name__)


def collator(batch, tokenizer):
    sents = []
    labels = []
    for sent, label, doc_id in batch:
        sents.append(' '.join(sent))
        label_ixs = [LABEL_TYPES.index(l) for l in label]
        label = np.zeros(len(LABEL_TYPES))
        label[label_ixs] = 1
        labels.append(label)
    tokd = tokenizer(sents, padding=True, max_length = 512, truncation=True)
    input_ids, token_type_ids, attention_mask = tokd['input_ids'], tokd['token_type_ids'], tokd['attention_mask']
    toks = torch.LongTensor(input_ids)
    mask = torch.LongTensor(attention_mask)
    labels = torch.Tensor(labels)
    return {'input_ids': toks, 'attention_mask': mask, 'labels': labels}

def compute_metrics(x):
    yhat_raw = 1 / (1 + np.exp(-x.predictions))
    yhat = np.round(yhat_raw)
    y = x.label_ids
    metrics = multilabel_eval.all_metrics(yhat, y, k=3, yhat_raw=yhat_raw,calc_auc=True,label_order=LABEL_TYPES)
    multilabel_eval.print_metrics(metrics, True)
    return metrics

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = argparse.ArgumentParser()
    parser.add_argument("train_fname", type=str)
    parser.add_argument("model", choices=['bert', 'clinicalbert'])
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--criterion", type=str, default="f1_micro", required=False, help="metric to use for early stopping")
    parser.add_argument("--patience", type=int, default=5, required=False, help="epochs to wait for improved criterion before early stopping (default 3)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11, help="random seed")
    parser.add_argument("--max_steps", type=int, default=-1, help="put a positive number to limit number of training steps for debugging")
    parser.add_argument("--eval_steps", type=int, default=1000, help="number of steps between evaluations during training")
    parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--cnn_on_top", action="store_true", help="set to use CNN on top of bert embedded tokens")
    parser.add_argument("--run_test", action="store_true", help="set to run on test too after running on dev at the end")
    args = parser.parse_args()

    if args.model == 'bert':
        args.model = 'bert-base-uncased'
    elif args.model == 'clinicalbert':
        args.model = 'emilyalsentzer/Bio_ClinicalBERT'

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Training/evaluation parameters %s", args)

    # Set seed
    set_seed(args.seed)

    num_labels = len(LABEL_TYPES)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    label2id = {label:ix for ix,label in enumerate(LABEL_TYPES)}
    id2label = {ix:label for label,ix in label2id.items()}
    config = BertConfig.from_pretrained(
        args.model,
        num_labels=num_labels,
        finetuning_task="text_classification",
        cache_dir=args.cache_dir if args.cache_dir else None,
        label2id=label2id,
        id2label=id2label,
    )
    tokenizer = BertTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.cnn_on_top:
        model = BertCNNForSequenceMultilabelClassification.from_pretrained(
            args.model,
            from_tf=bool(".ckpt" in args.model),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        model = BertForSequenceMultilabelClassification.from_pretrained(
            args.model,
            from_tf=bool(".ckpt" in args.model),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    # Get datasets
    #train
    train_dataset = SentDataset(args.train_fname)
    #dev
    dev_fname = args.train_fname.replace('train', 'val')
    eval_dataset = SentDataset(dev_fname)
    #test
    test_fname = args.train_fname.replace('train', 'test')
    test_dataset = SentDataset(test_fname)

    timestamp = time.strftime('%b_%d_%H:%M:%S', time.localtime())
    out_dir = f"results/{args.model}_{timestamp}"

    training_args = TrainingArguments(
            output_dir = out_dir,
            do_train=True,
            do_eval=True,
            do_predict=args.run_test,
            evaluate_during_training=True,
            max_steps=args.max_steps,
            #learning_rate=args.lr,
            num_train_epochs=args.max_epochs,
            save_total_limit=10,
            eval_steps=args.eval_steps,
            seed=args.seed,
            )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda x: collator(x, tokenizer),
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=args.model if os.path.isdir(args.model) else None
        )
        if args.max_steps < 0:
            trainer.save_model()

    # Evaluation
    eval_results = {}
    logger.info("*** Evaluate ***")

    #trainer.compute_metrics = compute_metrics(eval_dataset.args.task_name)
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    import pdb; pdb.set_trace()

    output_eval_file = os.path.join(
        training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
    )
    if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
            for key, value in eval_result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

    eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]
        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            import pdb; pdb.set_trace()
            predictions = np.argmax(predictions, axis=1)

            output_test_file = os.path.join(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = test_dataset.get_labels()[item]
                        writer.write("%d\t%s\n" % (index, item))
    return eval_results


if __name__ == "__main__":
    main()
