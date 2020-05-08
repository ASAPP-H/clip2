"""
This script gets all the BiLSTM predictions and merges it to the input format expected
by our evaluation code.
"""
import sys
import glob
import os
import pickle
import pandas as pd

sys.path.append("../../")
from CLIP import list_of_label_types
from evaluate import evaluation

pwd = os.environ.get('CLIP_DIR')


def combine_predictions_in_server(experiment_path):
    # Get the highest epoch number in test prediction directory
    eopch_max = 0
    run_paths = {
        tag_name: os.path.join(experiment_path, tag_name, "preds/test/")
        for tag_name in list_of_label_types
    }
    for tag_name in run_paths:
        epoch_max = max([int(x) for x in os.listdir(run_paths[tag_name])])
        run_paths[tag_name] = os.path.join(run_paths[tag_name], str(epoch_max))
    matches = glob.glob(
        os.path.join(experiment_path, "Medication/preds/test/%s" % str(epoch_max), "*.pkl")
    )
    os.makedirs("multilabel_preds", exist_ok=True)
    os.makedirs(os.path.join(experiment_path, "finegrained_preds"), exist_ok=True)
    for match in matches:
        if "_pred" not in match:
            continue
        pickle_for_file = []
        match_fn = match.split("/")[-1]
        for name in list_of_label_types:
            new_pickle = pickle.load(
                open(os.path.join(run_paths[name], match_fn), "rb")
            )
            pickle_for_file.append(new_pickle)
        pickle.dump(
            pickle_for_file,
            open(os.path.join(experiment_path, "finegrained_preds", match_fn), "wb"),
        )
    return


def evaluate_binary_bilstm(directory):
    import pdb; pdb.set_trace()
    file = pd.read_csv(os.path.join(pwd, "data/processed/test_finegrained.csv"))
    matches = glob.glob(os.path.join(directory, "*.pkl"))
    i2b2_docs = file[file["source"] == "i2b2"]
    i2b2_ids = i2b2_docs["document_id"].unique()
    mimic_docs = file[file["source"] == "mimic"]
    mimic_ids = mimic_docs["document_id"].unique()
    preds = []
    golds = []
    matches = [x for x in matches if "bin_" not in x]
    matches = sorted(matches, key=lambda x: int(x.split("/")[1].split("_")[0]))
    for match in matches:
        if "token" in match:
            continue
        doc_id = int(match.split("/")[1].split("_")[0])
        if doc_id in i2b2_ids:
            source = "i2b2"
        elif doc_id in mimic_ids:
            source = "mimic"
        else:
            print("WARNING CANNOT MAP TO SOURCE")
        list_ = pickle.load(open(match, "rb"))
        if "gold" in match:
            golds.append((list_, source))
        else:
            preds.append((list_, source))
    assert len(preds) == len(golds)
    f1 = evaluation.get_evaluation(preds, golds, ["followup"])
    pickle.dump(f1, open(os.path.join(directory, "bin_metric.pkl"), "wb"))
    return f1


def merge_and_evaluate_bilstm(pred_dir, output_dir):
    import pdb; pdb.set_trace()
    gold = pd.read_csv(os.path.join(pwd, "data/processed/test_finegrained.csv"))
    matches = glob.glob(os.path.join(pred_dir, "*.pkl"))
    document_preds = []
    document_gold = []
    i2b2_docs = gold[gold["source"] == "i2b2"]
    i2b2_ids = i2b2_docs["document_id"].unique()
    for pred in matches:
        # for one document
        pred_id = int(pred.split("/")[-1].split("_")[0])
        if pred_id in i2b2_ids:
            source = "i2b2"
        else:
            source = "mimic"
        gold_label = gold[gold["document_id"] == pred_id]
        if len(gold_label) == 0:
            continue
        gold_label = eval(gold_label.iloc[0]["labels"])
        document_gold.append((gold_label, source))
        prediction = pickle.load(open(pred, "rb"))
        num_lengths = [len(x) for x in prediction]
        combined_pred = []
        for i in range(len(prediction[0])):
            curr = []
            for j in range(len(prediction)):
                if (
                    prediction[j][i] != "O"
                    and prediction[j][i] != "SOS"
                    and prediction[j][i] != "EOS"
                ):
                    curr.append(prediction[j][i])
                elif prediction[j][i] == "SOS" or prediction[j][i] == "EOS":
                    if len(curr) == 0:
                        curr = [prediction[j][i]]  # allow one SOS or EOS
            if len(curr) == 0:
                # Only if there are no predictions from any of the taggers
                curr = ["O"]
            combined_pred.append(curr)
        document_preds.append((combined_pred, source))
    pickle.dump(document_preds, open(os.path.join(output_dir, "fine.pkl"), "wb"))
    f1 = evaluation.get_evaluation(document_preds, document_gold, list_of_label_types)
    pickle.dump(f1, open(os.path.join(output_dir, "fine_metrics.pkl"), "wb"))
    return f1


if __name__ == "__main__":
    pred_dir, output_dir, flag = sys.argv[1:]
    if flag == "binary":
        f1 = evaluate_binary_bilstm(pred_dir)
    if flag == "fine":
        combine_predictions_in_server(pred_dir)
        f1 = merge_and_evaluate_bilstm(
            os.path.join(pred_dir, "finegrained_preds"), output_dir
        )
    print("Evaluation results is %s" % str(f1))
