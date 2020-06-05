import hashlib
import os
from pathlib import Path
import pickle
import shutil

import torch
import tqdm
import numpy as np
import glob

from tagger import *
import random
from gensim import models
from gensim.models import KeyedVectors
from sklearn.utils.class_weight import compute_class_weight

"""
This implementation takes in the entire vocabulary. 
"""
import datetime


def write_metrics(run_name, epoch, loss, train_f1, dev_f1, exp_dir):
    with open(f"{exp_dir}/{run_name}/metrics.txt", "a") as f:
        f.write(f"epoch: {epoch}\tloss: {loss}\t\t\n")
        f.write(f"Train F1:  {str(train_f1)}\n")
        f.write(f"Dev F1:  {str(dev_f1)}\n")


def save_model(
    model, epoch, pretrained_list, token_to_id, char_to_id, id_to_tag, exp_dir
) -> str:
    print("Saving model...")
    os.makedirs(f"{exp_dir}/{model.run_name}/models", exist_ok=True)
    model_hash = hashlib.sha256(pickle.dumps(model)).hexdigest()
    model_path = f"{exp_dir}/{model.run_name}/models/{model_hash}.pt"
    torch.save(
        {
            "epoch": epoch,
            "pretrained_list": pretrained_list,
            "token_to_id": token_to_id,
            "char_to_id": char_to_id,
            "id_to_tag": id_to_tag,
            "model_state_dict": model.state_dict(),
            "run_name": model.run_name,
            "model_hash": model_hash,
        },
        model_path,
    )
    shutil.copyfile(model_path, f"{exp_dir}/{model.run_name}/models/master.pt")
    # utils.upload_file_to_s3(model_path, I2B2_2014_S3_BUCKET, model_path)


def get_class_weights(flat_tags):
    class_weights = compute_class_weight("balanced", np.unique(flat_tags), flat_tags)
    return class_weights


def create_vocab_mappings(train, dev, test):
    id_to_token = [PAD, UNK]
    token_to_id = {PAD: 0, UNK: 1}
    id_to_char = [PAD, UNK]
    char_to_id = {PAD: 0, UNK: 1}
    id_to_tag = [PAD, "SOS", "EOS"]
    tag_to_id = {PAD: 0, "SOS": 1, "EOS": 2}

    flat_tags = []
    # Here, we keep the structure of sentences,
    # so we have to treat things slightly differently.
    for token_sentences, tag_sentences, _, _ in train + dev + test:
        for token_sent in token_sentences:
            for token in token_sent:
                for char in token:
                    if char not in char_to_id:
                        char_to_id[char] = len(char_to_id)
                        id_to_char.append(char)
                token = token.lower()  # use lowercased tokens but original characters
                if token not in token_to_id:
                    token_to_id[token] = len(token_to_id)
                    id_to_token.append(token)
        for tag_sent in tag_sentences:
            for tag in tag_sent:
                flat_tags.append(tag)
                if tag not in tag_to_id:
                    tag_to_id[tag] = len(tag_to_id)
                    id_to_tag.append(tag)
    class_weights = get_class_weights(flat_tags)
    return (
        id_to_token,
        token_to_id,
        id_to_char,
        char_to_id,
        id_to_tag,
        tag_to_id,
        class_weights,
    )


def load_model_state(
    model,
    state_path,
    keys_to_skip=["hidden_to_tag.weight", "hidden_to_tag.bias", "crf.transitions"],
):
    model_state = torch.load(state_path)["model_state_dict"]

    for name, param in model.named_parameters():
        # Make sure no trainable params are missing.
        if param.requires_grad:
            assert_for_log(
                name in model_state,
                "In strict mode and failed to find at least one parameter: " + name,
            )
    for key in keys_to_skip:
        del model_state[key]
    model.load_state_dict(model_state, strict=False)


def get_embeddings(type_, path, id_to_token):
    if type_ == "GLOVE":
        pretrained = {}
        for line in open(path):
            parts = line.strip().split()
            word = parts[0]
            vector = [float(v) for v in parts[1:]]
            pretrained[word] = vector
    else:
        pretrained = KeyedVectors.load_word2vec_format(
            os.path.join(path, "BioWordVec_PubMed_MIMICIII_d200.vec.bin"), binary=True
        )
    pretrained_list = []
    scale = np.sqrt(3.0 / DIM_EMBEDDING)
    for word in id_to_token:
        # apply lower() because all GloVe vectors are for lowercase words
        if word.lower() in pretrained:
            pretrained_list.append(np.array(pretrained[word.lower()]))
        else:
            random_vector = np.random.uniform(-scale, scale, [DIM_EMBEDDING])
            pretrained_list.append(random_vector)
    return pretrained_list


def train_model(
    train,
    dev,
    test,
    run_name,
    exp_dir,
    use_crf=False,
    list_of_possible_tags=[],
    learning_rate=0.015,
    learning_decay_rate=0.05,
    weight_decay=1e-8,
    epochs=50,
    writer_preds_freq=10,
    load_checkpoint="",
    embeddings_type="GLOVE",
    embeddings_path="glove.6B.100d.txt",
):
    """
    train = ( list of tuples, where each tuple is the input and tags ).

    """

    def get_filepath(file_type: str, run_name: str) -> str:
        return Path(f"{exp_dir}/{run_name}/{file_type}.bio").resolve()

    # Append to metrics file every epoch
    if os.path.exists(f"{exp_dir}/{run_name}/metrics.txt"):
        os.remove(f"{exp_dir}/{run_name}/metrics.txt")
    vocab_path = f"{exp_dir}/{run_name}/vocab"
    os.makedirs(vocab_path, exist_ok=True)
    if len(glob.glob(os.path.join(vocab_path, "id_to_token"))) == 0:
        print("Creating cached files")
        # Load vocab
        id_to_token, token_to_id, id_to_char, char_to_id, id_to_tag, tag_to_id, class_weights = create_vocab_mappings(
            train, dev, test
        )
        pickle.dump(id_to_token, open(os.path.join(vocab_path, "id_to_token"), "wb"))
        pickle.dump(token_to_id, open(os.path.join(vocab_path, "token_to_id"), "wb"))
        pickle.dump(id_to_char, open(os.path.join(vocab_path, "id_to_char"), "wb"))
        pickle.dump(char_to_id, open(os.path.join(vocab_path, "char_to_id"), "wb"))
        pickle.dump(id_to_tag, open(os.path.join(vocab_path, "id_to_tag"), "wb"))
        pickle.dump(tag_to_id, open(os.path.join(vocab_path, "tag_to_id"), "wb"))
        pickle.dump(
            class_weights, open(os.path.join(vocab_path, "class_weights"), "wb")
        )
    else:
        print("Getting cached vocab files")
        id_to_token = pickle.load(open(os.path.join(vocab_path, "id_to_token"), "rb"))
        token_to_id = pickle.load(open(os.path.join(vocab_path, "token_to_id"), "rb"))
        id_to_char = pickle.load(open(os.path.join(vocab_path, "id_to_char"), "rb"))
        char_to_id = pickle.load(open(os.path.join(vocab_path, "char_to_id"), "rb"))
        id_to_tag = pickle.load(open(os.path.join(vocab_path, "id_to_tag"), "rb"))
        tag_to_id = pickle.load(open(os.path.join(vocab_path, "tag_to_id"), "rb"))
        class_weights = pickle.load(
            open(os.path.join(vocab_path, "class_weights"), "rb")
        )
    # Load pre-trained GloVe vectors
    pretrained_embs_cache = os.path.join(f"{exp_dir}", "embs.pkl")
    if len(glob.glob(pretrained_embs_cache)) == 0:
        pretrained_list = get_embeddings(embeddings_type, embeddings_path, id_to_token)
        pickle.dump(pretrained_list, open(pretrained_embs_cache, "wb"))
    else:
        pretrained_list = pickle.load(open(pretrained_embs_cache, "rb"))
    # Model creation
    model = TaggerModel(
        len(token_to_id),
        len(char_to_id),
        len(tag_to_id),
        pretrained_list,
        run_name,
        exp_dir,
        list_of_possible_tags,
        True,
        use_crf,
        class_weights,
        learning_rate,
        learning_decay_rate,
        weight_decay,
    )
    model_path = f"{exp_dir}/{model.run_name}/models/master.pt"

    if len(glob.glob(model_path)) > 0:
        # load the last checkpoint
        print("Getting checkpoint from %s" % model_path)
        model.load_state_dict(torch.load(model_path)["model_state_dict"])
    if load_checkpoint:
        print("Loading checkpoint from %s" % load_checkpoint)
        load_model_state(model, load_checkpoint)
    # Create optimizer and configure the learning rate
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rescale_lr = lambda epoch: 1 / (1 + learning_decay_rate * epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rescale_lr)

    dev_f1, test_f1 = 0, 0
    expressions = (model, optimizer)
    do_pass = do_pass_chunk_into_sentences
    for epoch in tqdm(range(epochs), desc="epoch"):
        random.shuffle(train)

        model.train()
        model.zero_grad()
        write_preds = False
        if epoch % writer_preds_freq == 0:
            write_preds = True
        loss, trf1 = do_pass(
            train,
            token_to_id,
            char_to_id,
            tag_to_id,
            id_to_tag,
            expressions,
            True,
            write_preds=write_preds,
            epoch=epoch,
            val_or_test="train",
        )
        model.eval()
        _, df1 = do_pass(
            dev,
            token_to_id,
            char_to_id,
            tag_to_id,
            id_to_tag,
            expressions,
            False,
            write_preds=write_preds,
            epoch=epoch,
        )

        write_metrics(run_name, epoch, loss, trf1, df1, exp_dir)
        # Update learning rate
        scheduler.step()
        if float(df1["all"]["strict"][1]["strict"]) > dev_f1:
            dev_f1 = float(df1["all"]["strict"][1]["strict"])
            save_model(
                model,
                epoch,
                pretrained_list,
                token_to_id,
                char_to_id,
                id_to_tag,
                exp_dir,
            )
        print("EPOCH: %s" % epoch)
        print(
            "{0} {1} loss {2} train-f1 {3} dev-f1 {4}".format(
                datetime.datetime.now(), epoch, loss, str(trf1), str(df1)
            )
        )
    # we do the multilabel case for exact-F1 matching.
    _, test1 = do_pass(
        test,
        token_to_id,
        char_to_id,
        tag_to_id,
        id_to_tag,
        expressions,
        False,
        write_preds=True,
        epoch=epoch,
        val_or_test="test",
    )
    _, dev1 = do_pass(
        dev,
        token_to_id,
        char_to_id,
        tag_to_id,
        id_to_tag,
        expressions,
        False,
        write_preds=True,
        epoch=epoch,
    )
