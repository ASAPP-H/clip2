import pandas as pd
import logging
import collections
from collections import namedtuple
from copy import deepcopy

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="DEBUG",
)

Entity = namedtuple("Entity", "e_type start_offset end_offset")

def simplify_token(token):
    if token == "(" or token == "[" or token == "{" or token == "<":
        token = "-LRB-"
    if token == ")" or token == "]" or token == "}" or token == ">":
        token = "-RRB-"
    chars = []
    for char in token:
        if char.isdigit():
            chars.append("0")
        else:
            chars.append(char)
    return "".join(chars)


def assert_for_log(condition, error_message):
    assert condition, error_message


def read_nonmultilabel_dataset(path):
    data = pd.read_csv(path, header=0)
    documents = data["tokens"].tolist()
    documents = [["SOS"] + eval(x) + ["EOS"] for x in documents]
    tags = data["labels"].tolist()
    tags = [[x[0] for x in eval(tag)] for tag in tags]
    res_tags = []
    for tag_seq in tags:
        res_tag_seq = []
        for tag in tag_seq:
            if tag != "O":
                tag = tag.split("-")[0] + "-followup"
            res_tag_seq.append(tag)
        res_tags.append(res_tag_seq)
    tags = res_tags
    tags = [["SOS"] + x + ["EOS"] for x in tags]
    ids = data["document_id"].tolist()
    return list(zip(documents, tags, ids))


def binarize_followup(tag_seq):
    res_tag_seq = []
    for tag in tag_seq:
        # for multilabel cases,
        if isinstance(tag, list):
            if any("followup" in tag_type for tag_type in tag):
               res_tag_seq.append("I-followup")
            else:
                res_tag_seq.append("O")
    return res_tag_seq


def binarize_by_filter(tag_seq, filter_type):
    # given a list of string, returns list of string
    # This function also flattens
    res_tag_seq = []
    for tag in tag_seq:
        # for multilabel cases,
        if isinstance(tag, list):
            if len(tag) == 1 and (tag[0] == "EOS" or tag[0] == "SOS"):
                res_tag_seq.append(tag[0])
            elif any(filter_type in tag_type for tag_type in tag):
                for tag_ in tag:
                    if filter_type in tag_:
                        res_tag_seq.append(tag_)
            else:
                res_tag_seq.append("O")
        elif isinstance(tag, str):
            if filter_type not in tag and tag != "SOS" and tag != "EOS":
                res_tag_seq.append("O")
            elif "B-" in tag and filter_type == "followup":
                tag = tag.replace("B-", "I-")
                res_tag_seq.append(tag)
            else:
                res_tag_seq.append(tag)
    return res_tag_seq


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


def count_filter_type(tag_docs, filter_type):
    num_tags = 0
    for tag_doc in tag_docs:
        for tag_sent in tag_doc:
            for tag in tag_sent:
                if tag != "O" and tag != "SOS" and tag != "EOS":
                    num_tags += 1
    print("Number of non-O tokens in dataset is %s" % str(num_tags))
    flattend_tags = [flatten_into_document(tag_doc) for tag_doc in tag_docs]
    num_segs = 0
    for flat in flattend_tags:
        all_segs = collect_named_entities(flat)
        all_segs = [x for x in all_segs if filter_type in x.e_type]
        num_segs += len(all_segs)
    print("Number of non-O segments is %s" % str(num_segs))


def read_sentence_structure(path, filter_type="followup"):
    """
    Test dataset.
    filter: str, this is for filtering.
    """
    print("Filtering for %s" % filter_type)
    data = pd.read_csv(path, header=0)
    documents = data["tokens"].tolist()
    documents = [eval(x) for x in documents]
    documents = [[["SOS"] + x + ["EOS"] for x in doc] for doc in documents]
    tags = data["labels"].tolist()
    tags = [eval(x) for x in tags]
    tags = [[["SOS"] + x + ["EOS"] for x in tag] for tag in tags]
    # Filter for the various types of tags.
    if filter_type is not None:
        tags = [[binarize_by_filter(x, filter_type) for x in tag] for tag in tags]
    num_of_filter_type = count_filter_type(tags, filter_type)
    ids = data["document_id"].tolist()
    return list(zip(documents, tags, ids))

"""
The below code is taken from David Batista's NER evaluation code:
https://github.com/davidsbatista/NER-Evaluation
"""

class Evaluator:
    def __init__(self, true, pred, tags):
        """
        """

        if len(true) != len(pred):
            raise ValueError("Number of predicted documents does not equal true")

        self.true = true
        self.pred = pred
        self.tags = tags

        # Setup dict into which metrics will be stored.

        self.metrics_results = {
            "correct": 0,
            "incorrect": 0,
            "partial": 0,
            "missed": 0,
            "spurious": 0,
            "possible": 0,
            "actual": 0,
            "precision": 0,
            "recall": 0,
        }

        # Copy results dict to cover the four schemes.

        self.results = {
            "strict": deepcopy(self.metrics_results),
            "ent_type": deepcopy(self.metrics_results),
            "partial": deepcopy(self.metrics_results),
            "exact": deepcopy(self.metrics_results),
        }

        # Create an accumulator to store results

        self.evaluation_agg_entities_type = {e: deepcopy(self.results) for e in tags}

    def evaluate(self):

        logging.info(
            "Imported %s predictions for %s true examples",
            len(self.pred),
            len(self.true),
        )

        for true_ents, pred_ents in zip(self.true, self.pred):

            # Check that the length of the true and predicted examples are the
            # same. This must be checked here, because another error may not
            # be thrown if the lengths do not match.

            if len(true_ents) != len(pred_ents):
                raise ValueError("Prediction length does not match true example length")

            # Compute results for one message

            tmp_results, tmp_agg_results = compute_metrics(
                collect_named_entities(true_ents),
                collect_named_entities(pred_ents),
                self.tags,
            )

            # Cycle through each result and accumulate

            # TODO: Combine these loops below:

            for eval_schema in self.results:

                for metric in self.results[eval_schema]:

                    self.results[eval_schema][metric] += tmp_results[eval_schema][
                        metric
                    ]

            # Calculate global precision and recall

            self.results = compute_precision_recall_wrapper(self.results)

            # Aggregate results by entity type

            for e_type in self.tags:

                for eval_schema in tmp_agg_results[e_type]:

                    for metric in tmp_agg_results[e_type][eval_schema]:

                        self.evaluation_agg_entities_type[e_type][eval_schema][
                            metric
                        ] += tmp_agg_results[e_type][eval_schema][metric]

                # Calculate precision recall at the individual entity level

                self.evaluation_agg_entities_type[
                    e_type
                ] = compute_precision_recall_wrapper(
                    self.evaluation_agg_entities_type[e_type]
                )

        return self.results, self.evaluation_agg_entities_type


def collect_named_entities(tokens, print_=False):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.

    :param tokens: a list of tags
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == "O":
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (
            ent_type == token_tag[2:] and token_tag[:1] == "B"
        ):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type is not None and start_offset is not None and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens) - 1))

    return named_entities


def compute_metrics(true_named_entities, pred_named_entities, tags):
    eval_metrics = {"correct": 0, "precision": 0, "recall": 0}

    # overall results, only for strict

    evaluation = {"strict": deepcopy(eval_metrics)}

    # results by entity type

    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in tags}

    true_named_entities = [ent for ent in true_named_entities if ent.e_type in tags]
    pred_named_entities = [ent for ent in pred_named_entities if ent.e_type in tags]
    # go through each predicted named-entity

    for pred in pred_named_entities:

        # Check each of the potential scenarios in turn. See
        # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        # for scenario explanation.

        # Scenario I: Exact match between true and pred

        if pred in true_named_entities:
            evaluation["strict"]["correct"] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]["strict"]["correct"] += 1
    # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
    # overall results, and use these to calculate precision and recall.
    evaluation["strict"]["possible"] = len(true_named_entities)
    evaluation["strict"]["actual"] = len(pred_named_entities)

    # Compute 'possible', 'actual', and precision and recall on entity level
    # results. Start by cycling through the accumulated results.

    for entity_type, entity_level in evaluation_agg_entities_type.items():

        # Cycle through the evaluation types for each dict containing entity
        # level results.

        evaluation_agg_entities_type[entity_type]["strict"]["possible"] = len(
            [ent for ent in true_named_entities if ent.e_type == entity_type]
        )
        evaluation_agg_entities_type[entity_type]["strict"]["actual"] = len(
            [ent for ent in pred_named_entities if ent.e_type == entity_type]
        )

    return evaluation, evaluation_agg_entities_type


def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges

    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().

    Examples:

    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def compute_actual_possible(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    correct = results["correct"]
    incorrect = results["incorrect"]

    # Possible: number annotations in the gold-standard which contribute to the
    # final score

    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system

    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    return results


def compute_precision_recall(results, partial_or_type=False):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    correct = results["correct"]

    if partial_or_type:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["precision"] = precision
    results["recall"] = recall

    return results


def compute_precision_recall_wrapper(results):
    """
    Wraps the compute_precision_recall function and runs on a dict of results
    """

    results_a = {
        key: compute_precision_recall(value, True)
        for key, value in results.items()
        if key in ["partial", "ent_type"]
    }
    results_b = {
        key: compute_precision_recall(value)
        for key, value in results.items()
        if key in ["strict", "exact"]
    }

    results = {**results_a, **results_b}

    return results
