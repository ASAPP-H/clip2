import collections
import csv
import os
import glob
import json
import pickle
from enum import Enum
from dataclasses import dataclass
from functools import lru_cache
import xml.etree.ElementTree as ET
import typing as t

import boto3
import spacy
import stanfordnlp
import nltk
import editdistance
import pendulum
import random
from tqdm.auto import tqdm
from nnsplit import NNSplit
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import syntok.segmenter
import numpy as np
import pandas as pd


SPACY_MODELS = [
    "en_core_web_sm",
    "en_core_web_lg",
    "en_trf_bertbaseuncased_lg",
    "en_trf_robertabase_lg",
    "en_trf_distilbertbaseuncased_lg",
    "en_trf_xlnetbasecased_lg",
    "en_core_sci_sm",
    "en_core_sci_md",
    "en_core_sci_lg",
]


@lru_cache()
def load_tokenizer(type_: str):
    print(f"Loading tokenizer: {type_}")
    if type_ in SPACY_MODELS:
        spacy.prefer_gpu()
        return spacy.load(type_)
    elif type_ == "nltk":
        # nltk tokenization functions are called on-demand
        return None
    elif type_ == "stanfordnlp_pretrained":
        download_stanfordnlp_models()
        return stanfordnlp.Pipeline(
            processors="tokenize",
            lang="en",
            tokenize_pretokenized=False,
            models_dir=STANFORDNLP_DIR,
        )
    elif type_ == "stanfordnlp_whitespace":
        download_stanfordnlp_models()
        return stanfordnlp.Pipeline(
            processors="tokenize",
            lang="en",
            tokenize_pretokenized=True,
            models_dir=STANFORDNLP_DIR,
        )
    elif type_ == "nnsplit":
        return NNSplit("en")
    elif type_ == "deepsegment":
        return DeepSegment("en")
    elif type_ == "wboag":
        # wboag (mimic_utils) tokenization function is called on-demand
        return None
    elif type_ == "syntok":
        # syntok tokenization function is called on-demand
        return None
    else:
        raise ValueError(f"Unknown tokenizer type: {type_}")


@dataclass
class SentenceTokenizer:
    type: str
    tokenize: t.Callable[[str], t.List[t.List[str]]]

    @classmethod
    def from_type(cls, type_: t.Optional[str] = "spacy_sm"):
        tokenizer = load_tokenizer(type_)
        if type_ in SPACY_MODELS:

            def tokenize(text: str):
                doc = tokenizer(text)
                return [sentence.text for sentence in doc.sents]

        elif type_ in ("stanfordnlp_pretrained", "stanfordnlp_whitespace"):

            def tokenize(text: str):
                doc = tokenizer(text)
                sentences = []
                cursor = 0
                sentence_start = 0
                for sentence in doc.sentences:
                    sentence_start = cursor
                    for token in sentence.tokens:
                        index = text.find(token.text, cursor)
                        cursor = index + len(token.text)
                    sentences.append([text[sentence_start:cursor]])
                return sentences

        elif type_ == "nltk":

            def tokenize(text: str):
                return nltk.sent_tokenize(text)

        elif type_ == "nnsplit":

            def tokenize(text: str):
                sentences = tokenizer.split([text])[0]
                return [
                    [token.text + token.whitespace for token in sent]
                    for sent in sentences
                ]

        elif type_ == "deepsegment":

            def tokenize(text: str):
                return tokenizer.segment(text)

        elif type_ == "wboag":

            def tokenize(text: str):
                return mimic_utils.sent_tokenize(text)

        elif type_ == "syntok":

            def tokenize(text: str):
                paragraphs = syntok.segmenter.analyze(text)
                all_sentences = []
                for paragraph in paragraphs:
                    for sentence in paragraph:
                        all_sentences.append(
                            "".join(
                                [f"{token.spacing}{token.value}" for token in sentence]
                            )
                        )
                    # NOTE: uncomment to restore more of the original document text
                    # all_sentences.append('\n\n')
                return all_sentences

        else:
            raise ValueError(f"Unknown tokenizer type: {type}")
        return cls(type=type_, tokenize=tokenize)


@dataclass
class WordTokenizer:
    type: str
    tokenize: t.Callable[[str], t.List[t.List[str]]]

    @classmethod
    def from_type(cls, type_: t.Optional[str] = ""):
        if type_ == "":

            def tokenize(text: str):
                return text.split(" ")

            return cls(type=type_, tokenize=tokenize)
        tokenizer = load_tokenizer(type_)
        if type_ in SPACY_MODELS:

            def tokenize(text: str):
                doc = tokenizer(text)
                return [token.text for sentence in doc.sents for token in sentence]

        elif type_ in ("stanfordnlp_pretrained", "stanfordnlp_whitespace"):

            def tokenize(text: str):
                doc = tokenizer(text)
                return [
                    token.text
                    for sentence in doc.sentences
                    for token in sentence.tokens
                ]

        elif type_ == "nltk":

            def tokenize(text: str):
                return nltk.word_tokenize(text)

        elif type_ == "syntok":

            def tokenize(text: str):
                paragraphs = syntok.segmenter.analyze(text)
                all_toks = []
                for paragraph in paragraphs:
                    for sentence in paragraph:
                        for token in sentence:
                            all_toks.append(f"{token.spacing}{token.value}")
                    # NOTE: uncomment to restore more of the original document text
                    # all_sentences.append('\n\n')
                return all_toks

        elif type_ == "nnsplit":

            def tokenize(text: str):
                sentence = tokenizer.split([text])[0][0]
                return [token.text for token in sentence]

        # elif type_ in TRANSFORMERS_TOKENIZERS:
        #     def tokenize(text: str):
        #         doc = tokenizer.encode(text)
        else:
            raise ValueError(f"Unknown tokenizer type: {type}")
        return cls(type=type_, tokenize=tokenize)
