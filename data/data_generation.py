"""
The character offset is in the form
    MIMIC_row_id: Label type -> char offsets.
"""
import os
import json
import csv
import sys

sys.path.append("../")
import pandas as pd


from tqdm.auto import tqdm
from typing import List
from document import Span
from utils import section_tokenize, sentence_to_token
from i2b2 import documents as i2b2_document
from CLIP import list_of_label_types
from os.path import isfile, join
from os import listdir

source_dir = "MIMIC/source"
raw_notes = pd.read_csv(os.path.join(source_dir, "NOTEEVENTS.csv"))
test_document_ids = json.load(open("test_document_ids", "r"))


def get_i2b2_files():
    current_dir = "i2b2/source/concept_assertion_relation_training_data/"
    onlyfiles_beth = [
        os.path.join(current_dir, "beth/txt", f)
        for f in listdir(os.path.join(current_dir, "beth/txt"))
        if isfile(os.path.join(current_dir, "beth/txt", f))
    ]
    onlylabels_beth = [
        os.path.join(current_dir, "beth/concept", f)
        for f in listdir(os.path.join(current_dir, "beth/concept"))
        if isfile(os.path.join(current_dir, "beth/concept", f))
    ]
    onlyfiles_beth = [x for x in onlyfiles_beth if "-" in x]
    onlylabels_beth = [x for x in onlylabels_beth if "-" in x]
    onlyfiles_beth = sorted(
        onlyfiles_beth, key=lambda x: int(x.split("-")[1].split(".")[0])
    )
    onlylabels_beth = sorted(
        onlylabels_beth, key=lambda x: int(x.split("-")[1].split(".")[0])
    )
    onlyfiles_partners = [
        os.path.join(current_dir, "partners/txt", f)
        for f in listdir(os.path.join(current_dir, "partners/txt"))
        if isfile(os.path.join(current_dir, "partners/txt", f))
    ]
    onlyfiles_partners_unann = [
        os.path.join(current_dir, "partners/unannotated", f)
        for f in listdir(os.path.join(current_dir, "partners/unannotated"))
        if isfile(os.path.join(current_dir, "partners/unannotated", f))
    ]
    onlylabels_partners = [
        os.path.join(current_dir, "partners/concept", f)
        for f in listdir(os.path.join(current_dir, "partners/concept"))
        if isfile(os.path.join(current_dir, "partners/concept", f))
    ]
    copy_label = [onlylabels_partners[0] for i in range(len(onlyfiles_partners_unann))]
    onlyfiles_partners.extend(onlyfiles_partners_unann)
    onlylabels_partners.extend(copy_label)
    onlyfiles_partners = sorted(
        onlyfiles_partners,
        key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[0]),
    )
    onlylabels_partners = sorted(
        onlylabels_partners,
        key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[0]),
    )
    onlyfiles_beth = zip(onlyfiles_beth, onlylabels_beth)
    onlyfiles_partners = zip(onlyfiles_partners, onlylabels_partners)
    df = pd.DataFrame(columns=["id", "text", "labels"])
    dataset_mapping = json.load(open("i2b2-test_offset_mapping.jsonl", "r"))
    for dataset in [onlyfiles_beth, onlyfiles_partners]:
        for file, label_file in dataset:
            if file.split(current_dir)[1] not in dataset_mapping:
                # Only include in the df file i2b2 data that is in the CLIP dataset.
                continue
            tokenized_sents, _ = i2b2_document.read_i2b2(file, label_file)
            spans = []
            for tag_name, offsets in dataset_mapping[
                file.split(current_dir)[1]
            ].items():
                for j, offset in enumerate(offsets):
                    spans.append(
                        Span(
                            type=tag_name,
                            id=j,
                            document_id=file.split(current_dir)[1],
                            start=offset["start_offset"],
                            end=offset["end_offset"],
                        )
                    )
            current = pd.DataFrame(
                [[file.split(current_dir)[1], tokenized_sents, spans]],
                columns=["id", "text", "labels"],
            )
            df = df.append(current)
    assert len(df) == len(dataset_mapping)
    return df


def get_dataset_raw_MIMIC_offsets(split="MIMIC_train"):
    if "i2b2" in split:
        dataset = get_i2b2_files()
        return dataset, split
    if split == 'MIMIC_train':
        note_pd = pd.read_csv('mimic-train.csv')
    elif split == 'MIMIC_val':
        note_pd = pd.read_csv('mimic-val.csv')
    elif split == 'MIMIC_test':
        note_pd = pd.read_csv('mimic-test.csv')
    elif split == 'i2b2_test':
        note_pd = pd.read_csv('i2b2-test.csv')
    #dataset_mapping = json.load(open("%s_offset_mapping.jsonl" % split))
    dataset_mapping = json.load(open("v1-%s-mapping.json" % split.lower().replace('_', '-')))
    dataset = []
    for note_id in dataset_mapping.keys():
        #orig_MIMIC_note = raw_notes[raw_notes["ROW_ID"] == int(note_id)]
        orig_MIMIC_note = note_pd[note_pd["id"] == int(note_id)]
        spans = []
        for tag_name, offsets in dataset_mapping[note_id].items():
            for j, offset in enumerate(offsets):
                spans.append(
                    Span(
                        type=tag_name,
                        id=j,
                        document_id=note_id,
                        start=offset["start_offset"],
                        end=offset["end_offset"],
                    )
                )
        # Make sure that the start_offset and end_offset is from token level.
        dataset.append([str(note_id), orig_MIMIC_note["TEXT"].iloc[0], spans])
    return pd.DataFrame(dataset, columns=["id", "text", "labels"]), split


def tags_to_IO_binary(label_sentences):
    bio_labels = []
    for label_sent in label_sentences:
        bio_label_sent = []
        for label in label_sent:
            if len(label) == 1 and label[0] == "NOT":
                bio_label_sent.append("O")
            else:
                # We train with I-O format for binary.
                bio_label_sent.append("I-followup")
        assert len(label_sent) == len(bio_label_sent)
        bio_labels.append(bio_label_sent)
    assert len(bio_labels) == len(label_sentences)
    return bio_labels


def tags_to_IO_finegrained(label_sentences, possible_labels):
    bio_labels = []
    for label_sent in label_sentences:
        bio_label_sent = []
        for label in label_sent:
            if len(label) == 1 and label[0] == "NOT":
                bio_label_sent.append(["O"])
            else:
                bio_label = []
                bio_label.extend(list(set(["I-" + key for key in label])))
                bio_label_sent.append(bio_label)
        assert len(label_sent) == len(bio_label_sent)
        bio_labels.append(bio_label_sent)
    assert len(bio_labels) == len(label_sentences)
    return bio_labels


def preprocess_dataset(
    documents,
    word_tokenizer_type: List[str],
    sentence_tokenizer_type: str,
    cast: str = "binary",
    name=None,
) -> str:
    rows = []
    for i in tqdm(range(len(documents)), desc="preprocessing"):
        doc = documents.iloc[i]
        text = doc["text"]
        if "i2b2" in name:
            text = " ".join([x for y in doc["text"] for x in y])
        sentences = section_tokenize(text, sentence_tokenizer_type)
        spans = doc["labels"]
        tokens, labels = sentence_to_token(text, sentences, word_tokenizer_type, spans)
        list_of_pos_label_types = [
            "Appointment-related followup",
            "Imaging-related followup",
            "Case-specific instructions for patient",
            "Medication-related followups",
            "Other helpful contextual information",
            "Lab-related followup",
            "Procedure-related followup",
        ]
        if cast == "binary":
            labels = tags_to_IO_binary(labels)
        else:
            labels = tags_to_IO_finegrained(labels, list_of_pos_label_types)
        rows.append({"document_id": doc["id"], "tokens": tokens, "labels": labels})
    return pd.DataFrame(rows)


def make_test_files():
    # Merge the MIMIC and i2b2 test set files.
    for cast in ["binary", "finegrained"]:
        mimic_test = pd.read_csv("processed2/MIMIC_test_%s.csv" % cast)
        mimic_test["source"] = "mimic"
        i2b2_test = pd.read_csv("processed2/i2b2-test_%s.csv" % cast)
        i2b2_test["source"] = "i2b2"
        mimic_test = mimic_test.append(i2b2_test)
        current = pd.DataFrame(columns=["document_id", "tokens", "labels", "source"])
        mimic_test["document_id"] = mimic_test["document_id"].apply(lambda x: str(x))
        for doc_id in test_document_ids:
            #import pdb; pdb.set_trace()
            doc = mimic_test[mimic_test["document_id"] == doc_id].iloc[0]
            current = current.append(doc)
        current.to_csv("processed2/test_%s.csv" % cast)


if __name__ == "__main__":
    train_pd, tr_name = get_dataset_raw_MIMIC_offsets(("MIMIC_train"))
    i2b2_test_pd, itest_name = get_dataset_raw_MIMIC_offsets(("i2b2-test"))
    val_pd, val_name = get_dataset_raw_MIMIC_offsets(("MIMIC_val"))
    mimic_test_pd, mtest_name = get_dataset_raw_MIMIC_offsets(("MIMIC_test"))
    datasets = {
        itest_name: i2b2_test_pd,
        tr_name: train_pd,
        val_name: val_pd,
        mtest_name: mimic_test_pd,
    }
    os.makedirs("processed2", exist_ok=True)
    for name, dataset in datasets.items():
        preproc_file = "processed2/%s_finegrained.csv" % name
        preproc_dataset = preprocess_dataset(
            dataset,
            word_tokenizer_type="nltk",
            sentence_tokenizer_type=["nltk"],
            cast="finegrained",
            name=name,
        )
        preproc_dataset.to_csv(preproc_file)
        preproc_file = "processed2/%s_binary.csv" % name
        preproc_dataset = preprocess_dataset(
            dataset,
            word_tokenizer_type="nltk",
            sentence_tokenizer_type=["nltk"],
            cast="binary",
            name=name,
        )
        preproc_dataset.to_csv(preproc_file)

    make_test_files()
