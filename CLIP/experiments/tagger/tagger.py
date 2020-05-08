from tqdm.auto import tqdm

from crf import CRF
import sys

sys.path.append("../../../")
from utils import *
from evaluate import evaluation
import os
import pickle
import pandas as pd
import collections
import csv
import os
import glob
import pickle
from enum import Enum
from dataclasses import dataclass
from functools import lru_cache
import xml.etree.ElementTree as ET
import typing as t
from sklearn.utils.class_weight import compute_class_weight

PAD = "__PAD__"
UNK = "__UNK__"
DIM_EMBEDDING = 200
LSTM_LAYER = 3
LSTM_HIDDEN = 200
CHAR_DIM_EMBEDDING = 25
CHAR_LSTM_HIDDEN = 25
BATCH_SIZE = 16

EPOCHS = 50
KEEP_PROB = 0.5

import torch

torch.manual_seed(0)

device = torch.device("cpu")


class TaggerModel(torch.nn.Module):
    def __init__(
        self,
        nwords,
        nchars,
        ntags,
        pretrained_list,
        run_name,
        exp_name,
        list_of_possible_tags,
        use_char=True,
        use_crf=False,
        class_weights=[],
        learning_rate=0.015,
        learning_decay_rate=0.05,
        weight_decay=1e-8,
    ):
        super().__init__()

        self.run_name = run_name
        self.exp_name = exp_name
        self.class_weights = torch.Tensor(class_weights)
        # Create word embeddings
        pretrained_tensor = torch.FloatTensor(pretrained_list)
        self.word_embedding = torch.nn.Embedding.from_pretrained(
            pretrained_tensor, freeze=False
        )
        self.list_of_possible_tags = list_of_possible_tags
        # Create input dropout parameter
        # self.word_dropout = torch.nn.Dropout(1 - KEEP_PROB)
        char_lstm_hidden = 0
        self.use_char = use_char
        if self.use_char:
            # Character-level LSTMs
            self.char_embedding = torch.nn.Embedding(nchars, CHAR_DIM_EMBEDDING)
            self.char_lstm = torch.nn.LSTM(
                CHAR_DIM_EMBEDDING,
                CHAR_LSTM_HIDDEN,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            char_lstm_hidden = CHAR_LSTM_HIDDEN

        # Create LSTM parameters
        self.lstm = torch.nn.LSTM(
            DIM_EMBEDDING + char_lstm_hidden,
            LSTM_HIDDEN,
            num_layers=LSTM_LAYER,
            batch_first=True,
            bidirectional=True,
        )
        # Create output dropout parameter
        self.lstm_output_dropout = torch.nn.Dropout(1 - KEEP_PROB)

        # Create final matrix multiply parameters
        self.hidden_to_tag = torch.nn.Linear(LSTM_HIDDEN * 2, ntags)
        self.ntags = ntags
        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(target_size=ntags)

    def forward(self, sentences, mask, sent_tokens, labels, lengths, cur_batch_size):
        """
        sent_Tokens is a list of list of lists, where the essential unit is a
        token, and it indices each character in the token. The max token length is
        the extra dimension in sent_Tokens.
        sentences is the sentence embedding.

        """
        max_length = sentences.size(1)
        # Look up word vectors
        word_vectors = self.word_embedding(sentences)
        # Apply dropout
        # dropped_word_vectors = self.word_dropout(word_vectors)
        if self.use_char:
            sent_tokens = sent_tokens.view(cur_batch_size * max_length, -1)
            token_vectors = self.char_embedding(sent_tokens)
            char_lstm_out, (hn, cn) = self.char_lstm(token_vectors, None)
            char_lstm_out = hn[-1].view(cur_batch_size, max_length, CHAR_LSTM_HIDDEN)
            concat_vectors = torch.cat((word_vectors, char_lstm_out), dim=2)
        else:
            concat_vectors = word_vectors

        # Run the LSTM over the input, reshaping data for efficiency
        packed_words = torch.nn.utils.rnn.pack_padded_sequence(
            concat_vectors, lengths, True
        )
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=max_length
        )
        # Apply dropout
        lstm_out_dropped = self.lstm_output_dropout(lstm_out)
        # Matrix multiply to get scores for each tag
        output_scores = self.hidden_to_tag(lstm_out_dropped)
        if self.use_crf:
            loss = self.crf.neg_log_likelihood_loss(output_scores, mask.bool(), labels)
            predicted_tags = self.crf(output_scores, mask.bool())
        else:
            output_scores = output_scores.view(cur_batch_size * max_length, -1)
            flat_labels = labels.view(cur_batch_size * max_length)
            loss_function = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_function(output_scores, flat_labels)
            predicted_tags = torch.argmax(output_scores, 1)
            predicted_tags = predicted_tags.view(cur_batch_size, max_length)

        return loss, predicted_tags


def write_preds_fn(
    pred_list, gold_list, document_ids, name, exp_name, run_name, epoch, doc_tokens
):
    # put predictions in a folder
    pred_list_with_id = list(zip(pred_list, document_ids, gold_list, doc_tokens))
    predictions_dir = os.path.join(exp_name, run_name, "preds", name)
    print("Writing predictions into pkl files to %s" % predictions_dir)
    os.makedirs(f"{predictions_dir}/{str(epoch)}/", exist_ok=True)
    for pred, doc_id, gold, doc_ts in pred_list_with_id:
        if "/" in str(doc_id):
            doc_id = doc_id.split("/")[-1]
        pickle.dump(
            pred,
            open(
                os.path.join(predictions_dir, str(epoch), "%s_pred.pkl" % str(doc_id)),
                "wb",
            ),
        )
        pickle.dump(
            doc_ts,
            open(
                os.path.join(
                    predictions_dir, str(epoch), "%s_tokens.pkl" % str(doc_id)
                ),
                "wb",
            ),
        )

        pickle.dump(
            gold,
            open(
                os.path.join(predictions_dir, str(epoch), "%s_gold.pkl" % str(doc_id)),
                "wb",
            ),
        )


def flatten_into_document(doc_list):
    # flatten into document, this means condensing all the intermediate SOS,
    # EOS tokens
    res_list = []
    for i in range(len(doc_list)):
        if i == 0:
            sent = doc_list[i][:-1]  # no EOS
        elif i == len(doc_list) - 1:
            sent = doc_list[i][1:]  # no SOS
        else:
            sent = doc_list[i][1:-1]  # both no SOS and EOS
        res_list.extend(sent)
    return res_list


def do_pass_chunk_into_sentences(
    data,
    token_to_id,
    char_to_id,
    tag_to_id,
    id_to_tag,
    expressions,
    train,
    write_preds,
    epoch,
    val_or_test="val",
):
    """
    Approximating document-level prediction by doing sentence-level prediction
    indepndently for each sentence in each document, then combining the sentence
    level predictions for each document.
    Data is a list of list of lists, where x[0] is document, x[1] is label
    documetn in turn is a list of lists, where each list is a sentence.
    Labels are similarly split into, for each document, a lsi tof lists.
    """
    model, optimizer = expressions
    # Loop over batches
    loss = 0
    gold_lists, pred_lists = [], []
    for index in tqdm(range(0, len(data)), desc="batch"):
        document = data[index]
        sentences = document[0]
        sentence_tags = document[1]
        doc_ids = document[2]
        doc_pred_list = []
        doc_gold_list = []
        sentence_ids = list(range(len(sentences)))
        sentence_level_exs = list(zip(sentences, sentence_tags, sentence_ids))
        sentence_lengths = [len(x) for x in sentences]
        sentence_tags_lengths = [len(x) for x in sentence_tags]
        assert len(sentence_lengths) == len(sentence_tags_lengths)
        for i in range(len(sentence_lengths)):
            assert sentence_lengths[i] == sentence_tags_lengths[i]
        for start in range(0, len(sentence_level_exs), BATCH_SIZE):
            batch = sentence_level_exs[start : start + BATCH_SIZE]
            batch.sort(key=lambda x: -len(x[0]))
            # Prepare inputs
            cur_batch_size = len(batch)
            max_length = len(batch[0][0])
            lengths = [len(v[0]) for v in batch]
            max_token_length = 0
            for tokens, _, _ in batch:
                for token in tokens:
                    max_token_length = max([max_token_length, len(token), len(token)])
            input_array = torch.zeros((cur_batch_size, max_length)).long()
            mask_array = torch.zeros((cur_batch_size, max_length)).byte()
            input_token_array = torch.zeros(
                (cur_batch_size, max_length, max_token_length)
            ).long()
            output_array = torch.zeros((cur_batch_size, max_length)).long()
            for n, (tokens, tags, _) in enumerate(batch):
                token_ids = [token_to_id.get(t.lower(), 1) for t in tokens]
                input_array[n, : len(tokens)] = torch.LongTensor(token_ids)
                for m, token in enumerate(tokens):
                    char_ids = [char_to_id.get(c, 1) for c in token]
                    input_token_array[n, m, : len(token)] = torch.LongTensor(char_ids)
                tag_ids = [tag_to_id[t] for t in tags]
                mask_ids = [1 for t in tokens]
                try:
                    mask_array[n, : len(tokens)] = torch.LongTensor(mask_ids)
                    output_array[n, : len(tags)] = torch.LongTensor(tag_ids)
                except:
                    import pdb

                    pdb.set_trace()
            model.to(device)
            # Construct computation
            batch_loss, output = model(
                input_array.to(device),
                mask_array.to(device),
                input_token_array.to(device),
                output_array.to(device),
                lengths,
                cur_batch_size,
            )
            # Run computations
            if train:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                loss += batch_loss.item()
            predicted = output.cpu().data.numpy()
            out_dict = list(zip(batch, predicted))
            # reorder sentences back to original order.
            out_dict.sort(key=lambda x: x[0][2])

            # Update the number of correct tags and total tags
            for (_, g, _), a in out_dict:
                gold_list, pred_list = [], []
                for gt, at in zip(g, a):
                    at = id_to_tag[at]
                    gold_list.append(gt)
                    pred_list.append(at)
                doc_gold_list.append(gold_list)
                doc_pred_list.append(pred_list)

        # flatten so each document is a list of document-level predictions
        doc_gold_list = flatten_into_document(doc_gold_list)
        doc_pred_list = flatten_into_document(doc_pred_list)
        gold_lists.append(doc_gold_list)
        pred_lists.append(doc_pred_list)
    if write_preds:
        document_tokens = [flatten_into_document(doc[0]) for doc in data]
        write_preds_fn(
            pred_lists,
            gold_lists,
            [example[2] for example in data],
            val_or_test,
            model.exp_name,
            model.run_name,
            epoch,
            document_tokens,
        )
    if train or val_or_test == "val":
        gold_lists = [(x, "mimic") for x in gold_lists]
        pred_lists = [(x, "mimic") for x in pred_lists]
    else:
        gold_lists = [(gold_lists[i], data[i][3]) for i in range(len(gold_lists))]
        pred_lists = [(pred_lists[i], data[i][3]) for i in range(len(gold_lists))]
    return (
        loss,
        evaluation.get_evaluation(
            pred_lists, gold_lists, model.list_of_possible_tags, source="mimic"
        ),
    )
