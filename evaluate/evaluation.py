"""
This evaluation code is adapted from https://github.com/davidsbatista/NER-Evaluation/blob/master/example-full-named-entity-evaluation.ipynb

"""
import sys
sys.path.append("../")
from utils import *
import numpy as np
import copy
from copy import deepcopy


def count_by_label_type(preds, gold, list_of_pos_label_types):
    from tqdm.auto import tqdm

    label_type_dict = {label: 0 for label in list_of_pos_label_types}
    for true_ents, pred_ents in tqdm(zip(gold, preds)):
        for label_type in list_of_pos_label_types:
            filtered_gold_for_label_type = binarize_by_filter(true_ents, label_type)
            filtered_gold_for_label_type = collect_named_entities(
                filtered_gold_for_label_type
            )
            filtered_gold_for_label_type = [
                ent
                for ent in filtered_gold_for_label_type
                if ent.e_type != "S" and ent.e_type != "O" and ent.e_type != "E"
            ]
            label_type_dict[label_type] += len(filtered_gold_for_label_type)
    return {"label_count": label_type_dict}


def get_counts(preds, gold, list_of_pos_label_types):
    preds_i2b2 = [pred[0] for pred in preds if pred[1] == "i2b2"]
    gold_i2b2 = [g_[0] for g_ in gold if g_[1] == "i2b2"]
    i2b2_partial_F1 = count_by_label_type(
        preds_i2b2, gold_i2b2, list_of_pos_label_types
    )
    preds_mimic = [pred[0] for pred in preds if pred[1] == "mimic"]
    gold_mimic = [g_[0] for g_ in gold if g_[1] == "mimic"]
    mimic_partial_F1 = count_by_label_type(
        preds_mimic, gold_mimic, list_of_pos_label_types
    )
    return {
        "i2b2": {"partial": i2b2_partial_F1},
        "mimic": {"partial": mimic_partial_F1},
    }


def calculate_F1_for_F1_type(precision, recall):
    if recall + precision == 0:
        return 0
    f1 = float(2 * (recall * precision)) / float(recall + precision)
    return f1


def convert_to_token_indices_list(entities):
    ranges = [
        [
            str(x) + ent[1]
            for x in list(range(ent[0].start_offset, ent[0].end_offset + 1))
        ]
        for ent in entities
    ]
    ranges = [x for y in ranges for x in y]
    return ranges


def convert_to_token_indices_no_type(entities, is_gold=False, lenient_K=None):
    ranges = [
        list(range(ent[0].start_offset, ent[0].end_offset + 1)) for ent in entities
    ]
    ranges = [x for y in ranges for x in y]
    if lenient_K is not None:
        if is_gold:
            ranges = [
                list(
                    range(
                        min(0, ent[0].start_offset - lenient_K),
                        ent[0].end_offset + 1 + lenient_K,
                    )
                )
                for ent in entities
            ]
            ranges = [x for y in ranges for x in y]
    return ranges


def get_span_counter(entities_list, max_x, max_y):
    count_mtx = [[0] * (max_y + 1) for i in range(max_x + 1)]
    for ent in entities_list:
        count_mtx[ent.start_offset][ent.end_offset] += 1
    return count_mtx


def update_K_leniency(span_counts_g, span_counts_p, g_is_1, K=10):
    for interval in g_is_1:
        start, end = interval[0], interval[1]
        # Look K behind,
        lenient_lower_bound = max(0, start - 10)
        lenient_upper_bound = min(end + 10, len(span_counts_p[0]))
        if span_counts_p[start][end] > 0:
            # It already got the followup, continue
            continue
        for y in range(lenient_lower_bound, start):
            for x in range(y + 1):
                # x <= y
                if span_counts_p[x][y] == 1 and span_counts_p[start][end] == 0:
                    span_counts_p[start][end] += 1  # Leniency, we count it to
                    # have detected the followup.
        # Look K in front
        if span_counts_p[start][end] == 0:
            # If STILL 0, look ahead
            for x in range(end + 1, lenient_lower_bound + 1):
                for y in range(x, len(span_counts_p)):
                    # x <= y
                    if span_counts_p[x][y] == 1 and span_counts_p[start][end] == 0:
                        span_counts_p[start][end] += 1
    return span_counts_p


def get_exact_F1(preds, gold, list_of_pos_label_types):
    """
    Get the exact-F1 followup.
    """
    num_pred_spans = 0
    num_gold_spans = 0
    tp = 0
    for true_ents, pred_ents in zip(gold, preds):
        expanded_pred_entities = []
        expanded_gold_entities = []
        for label_type in list_of_pos_label_types:
            filtered_gold_for_label_type = binarize_by_filter(true_ents, label_type)
            filtered_preds_for_label_type = binarize_by_filter(pred_ents, label_type)
            filtered_gold_for_label_type = collect_named_entities(
                filtered_gold_for_label_type
            )
            filtered_preds_for_label_type = collect_named_entities(
                filtered_preds_for_label_type
            )
            filtered_gold_for_label_type = [
                ent
                for ent in filtered_gold_for_label_type
                if ent.e_type != "S" and ent.e_type != "O" and ent.e_type != "E"
            ]
            filtered_preds_for_label_type = [
                ent
                for ent in filtered_preds_for_label_type
                if ent.e_type != "S" and ent.e_type != "O" and ent.e_type != "E"
            ]
            expanded_gold_entities.extend(filtered_gold_for_label_type)
            expanded_pred_entities.extend(filtered_preds_for_label_type)
        # For this document, get the true positives for exact
        if len(expanded_gold_entities) == 0:
            # if we have a document with no followups, just increase the
            # prediction spans but dont' update correct or num_gold
            num_pred_spans += len(expanded_pred_entities)
            continue
        max_xs = [ent.start_offset for ent in expanded_gold_entities]
        max_xs.extend([ent.start_offset for ent in expanded_pred_entities])
        max_x = max(max_xs)
        max_y = [ent.end_offset for ent in expanded_gold_entities]
        max_y.extend([ent.end_offset for ent in expanded_pred_entities])
        max_y = max(max_y)
        span_counts_g = get_span_counter(expanded_gold_entities, max_x, max_y)
        span_counts_p = get_span_counter(expanded_pred_entities, max_x, max_y)

        import numpy as np

        mask = np.logical_and(span_counts_p, span_counts_g)

        intersect_list = np.where(mask == 1)
        intersect_list = list(zip(intersect_list[0], intersect_list[1]))
        for x, y in intersect_list:
            tp += min(span_counts_g[x][y], span_counts_p[x][y])
        num_pred_spans += len(expanded_pred_entities)
        num_gold_spans += len(expanded_gold_entities)
    # Now, calcualte precision
    if num_pred_spans == 0:
        precision = 0
    else:
        precision = tp / num_pred_spans
    if num_gold_spans == 0:
        recall = 0
    else:
        recall = tp / num_gold_spans
    f1 = calculate_F1_for_F1_type(precision, recall)
    return {
        "tp": tp,
        "actual": num_pred_spans,
        "possible": num_gold_spans,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calculate_tp(
    gold_range, pred_range, gold_range_no_type=None, pred_range_no_type=None
):
    from collections import Counter

    # out of the ones that have
    gold_cntr = Counter(gold_range)
    pred_cntr = Counter(pred_range)
    gold_intersect_pred = set(gold_range).intersection(pred_range)
    curr_tp = 0
    curr_tp_no_type = 0
    for index in gold_intersect_pred:
        curr_tp += min(gold_cntr[index], pred_cntr[index])
    if gold_range_no_type is not None:  # if with tyep
        gold_cntr = Counter(gold_range_no_type)
        pred_cntr = Counter(pred_range_no_type)
        gold_intersect_pred = set(gold_range_no_type).intersection(pred_range_no_type)
        for index in gold_intersect_pred:
            curr_tp_no_type += min(gold_cntr[index], pred_cntr[index])
    return curr_tp, curr_tp_no_type


def get_partial_F1_index_level(preds, gold, list_of_pos_label_types, with_type=False):
    """
    Get partial-F1.
    We may also want to know, given all the predictions that were overlapping with gold,
    how accurate were they in terms of type-classification? In this case, set with_type=True.
    """
    tp = 0
    tp_no_type = 0
    total_gold_tokens = 0
    total_pred_tokens = 0
    label_type_dict = {
        label: {"possible": 0, "actual": 0, "tp": 0}
        for label in list_of_pos_label_types
    }
    for true_ents, pred_ents in zip(gold, preds):
        expanded_pred_entities = []
        expanded_gold_entities = []
        # For each document, calculate the true positive for that document'
        for label_type in list_of_pos_label_types:
            filtered_gold_for_label_type = binarize_by_filter(true_ents, label_type)
            filtered_preds_for_label_type = binarize_by_filter(pred_ents, label_type)
            filtered_gold_for_label_type = collect_named_entities(
                filtered_gold_for_label_type
            )
            filtered_preds_for_label_type = collect_named_entities(
                filtered_preds_for_label_type
            )
            filtered_gold_for_label_type = [
                ent
                for ent in filtered_gold_for_label_type
                if ent.e_type != "S" and ent.e_type != "O" and ent.e_type != "E"
            ]
            filtered_gold_for_label_type = [
                (ent, label_type) for ent in filtered_gold_for_label_type
            ]
            filtered_preds_for_label_type = [
                ent
                for ent in filtered_preds_for_label_type
                if ent.e_type != "S" and ent.e_type != "O" and ent.e_type != "E"
            ]
            filtered_preds_for_label_type = [
                (ent, label_type) for ent in filtered_preds_for_label_type
            ]
            expanded_gold_entities.extend(filtered_gold_for_label_type)
            expanded_pred_entities.extend(filtered_preds_for_label_type)
            # Calculate for this label type, for label type breakdown.
            filtered_pred_range = convert_to_token_indices_list(
                filtered_preds_for_label_type
            )
            filtered_gold_range = convert_to_token_indices_list(
                filtered_gold_for_label_type
            )
            label_type_dict[label_type]["possible"] += len(filtered_gold_range)
            label_type_dict[label_type]["actual"] += len(filtered_pred_range)
            new_tp, _ = calculate_tp(filtered_gold_range, filtered_pred_range)
            label_type_dict[label_type]["tp"] += new_tp
        if with_type:
            gold_range = convert_to_token_indices_list(expanded_gold_entities)
            pred_range = convert_to_token_indices_list(expanded_pred_entities)
            gold_range_no_type = convert_to_token_indices_no_type(
                expanded_gold_entities
            )
            pred_range_no_type = convert_to_token_indices_no_type(
                expanded_pred_entities
            )
        else:
            gold_range = convert_to_token_indices_no_type(expanded_gold_entities)
            pred_range = convert_to_token_indices_no_type(expanded_pred_entities)
            gold_range_no_type = None
            pred_range_no_type = None

        new_tp, new_tp_no_type = calculate_tp(
            gold_range, pred_range, gold_range_no_type, pred_range_no_type
        )
        tp += new_tp
        tp_no_type += new_tp_no_type
        total_gold_tokens += len(gold_range)
        total_pred_tokens += len(pred_range)

    if tp_no_type > 0:
        print(
            "Number of prediction tokens partial correct detection %s out of %s total tokens"
            % (str(tp), total_gold_tokens)
        )
        print(
            "Out of the total partially correct tokens, %s of the overlaps were of the correct type"
            % (str(float(tp) / float(tp_no_type)))
        )
    if total_pred_tokens == 0:
        precision = 0
    else:
        precision = tp / total_pred_tokens
    if total_gold_tokens == 0:
        recall = 0
    else:
        recall = tp / total_gold_tokens
    f1 = calculate_F1_for_F1_type(precision, recall)
    return {
        "tp": tp,
        "actual": total_pred_tokens,
        "possible": total_gold_tokens,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "label_breakdown": label_type_dict,
    }


def get_strict_F1(preds, gold, list_of_pos_label_types):
    """
	list_of_pos_label_types is a list of strings that is all possible labels
	(label space)
	"""
    metrics_results = {
        "correct": 0,
        "possible": 0,
        "actual": 0,
        "precision": 0,
        "recall": 0,
    }
    total_results = {"strict": deepcopy(metrics_results)}
    evaluation_agg_entities_type = {
        e: deepcopy(total_results) for e in list_of_pos_label_types
    }
    for true_ents, pred_ents in zip(gold, preds):
        expanded_gold_entities = []
        expanded_pred_entities = []
        for label_type in list_of_pos_label_types:
            filtered_gold_for_label_type = binarize_by_filter(true_ents, label_type)
            filtered_preds_for_label_type = binarize_by_filter(pred_ents, label_type)
            # we collect the named entities independnetly for multilable case.
            # so same offsets can appear twice with different types in the list of
            # final Entities
            filtered_gold_for_label_type = collect_named_entities(
                filtered_gold_for_label_type
            )
            filtered_preds_for_label_type = collect_named_entities(
                filtered_preds_for_label_type
            )
            expanded_gold_entities.extend(filtered_gold_for_label_type)
            expanded_pred_entities.extend(filtered_preds_for_label_type)
        tmp_results, tmp_agg_results = compute_metrics(
            expanded_gold_entities, expanded_pred_entities, list_of_pos_label_types
        )
        # Update correct, actual, and possible for each.
        for eval_schema in total_results.keys():
            for metric in metrics_results.keys():
                total_results[eval_schema][metric] += tmp_results[eval_schema][metric]
        for e_type in list_of_pos_label_types:
            for eval_schema in tmp_agg_results[e_type]:
                if eval_schema in evaluation_agg_entities_type[e_type]:
                    for metric in tmp_agg_results[e_type][eval_schema]:
                        evaluation_agg_entities_type[e_type][eval_schema][
                            metric
                        ] += tmp_agg_results[e_type][eval_schema][metric]
            evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(
                evaluation_agg_entities_type[e_type]
            )
    total_results = compute_precision_recall_wrapper(total_results)
    total_results["strict"]["f1"] = calculate_F1_for_F1_type(
        total_results["strict"]["precision"], total_results["strict"]["recall"]
    )
    macro_f1 = {}
    for f1_type in total_results:
        f1_sums = []
        # only calculate F1 scores for  label types that appear in gold.
        for ent_type in list(evaluation_agg_entities_type.keys()):
            f1 = calculate_F1_for_F1_type(
                evaluation_agg_entities_type[ent_type][f1_type]["precision"],
                evaluation_agg_entities_type[ent_type][f1_type]["recall"],
            )
            evaluation_agg_entities_type[ent_type]["strict"]["f1"] = f1
            f1_sums.append(f1)
        macro_f1[f1_type] = np.mean(f1_sums)
    return evaluation_agg_entities_type, macro_f1, total_results


def get_K_F1(preds, gold, list_of_pos_label_types):
    """
    K-F1 is partial-F1 based.

    """
    total_gold_tokens = 0
    total_pred_tokens = 0
    recall_tp = 0
    prec_tp = 0
    from tqdm.auto import tqdm

    label_type_dict = {
        label: {"possible": 0, "actual": 0, "tp": 0}
        for label in list_of_pos_label_types
    }
    for true_ents, pred_ents in tqdm(zip(gold, preds)):
        expanded_pred_entities = []
        expanded_gold_entities = []
        for label_type in list_of_pos_label_types:
            filtered_gold_for_label_type = binarize_by_filter(true_ents, label_type)
            filtered_preds_for_label_type = binarize_by_filter(pred_ents, label_type)
            filtered_gold_for_label_type = collect_named_entities(
                filtered_gold_for_label_type
            )
            filtered_preds_for_label_type = collect_named_entities(
                filtered_preds_for_label_type
            )
            filtered_gold_for_label_type = [
                ent
                for ent in filtered_gold_for_label_type
                if ent.e_type != "S" and ent.e_type != "O" and ent.e_type != "E"
            ]
            filtered_preds_for_label_type = [
                ent
                for ent in filtered_preds_for_label_type
                if ent.e_type != "S" and ent.e_type != "O" and ent.e_type != "E"
            ]
            filtered_preds_for_label_type = [
                (ent, label_type) for ent in filtered_preds_for_label_type
            ]
            filtered_gold_for_label_type = [
                (ent, label_type) for ent in filtered_gold_for_label_type
            ]
            expanded_gold_entities.extend(filtered_gold_for_label_type)
            expanded_pred_entities.extend(filtered_preds_for_label_type)
        gold_range = convert_to_token_indices_no_type(
            expanded_gold_entities, is_gold=True, lenient_K=5
        )
        # convert_to_token_indices_no_type
        # Only expand for the predictions that don't overlap.
        new_expanded_pred_entities = []
        pred_range = convert_to_token_indices_no_type(expanded_pred_entities)

        # you expand it by 20 at ea
        gold_range_no_type = None
        pred_range_no_type = None
        new_tp, new_tp_no_type = calculate_tp(
            gold_range, pred_range, True, gold_range_no_type, pred_range_no_type
        )
        recall_tp += new_tp
        # To calculate precision, we go back to the total numb erof tokens, we don't coun tthe expansion.
        gold_range = convert_to_token_indices_no_type(expanded_gold_entities)
        # we add this by 2* this.
        total_pred_tokens += len(pred_range)
        total_gold_tokens += max(
            len(gold_range) + 2 * len(expanded_gold_entities), new_tp
        )  # make sure this never exceeds 1.0
        prec_tp += new_tp

    if total_pred_tokens == 0:
        precision = 0
    else:
        precision = prec_tp / total_pred_tokens
    if total_gold_tokens == 0:
        recall = 0
    else:
        recall = recall_tp / total_gold_tokens
    f1 = calculate_F1_for_F1_type(precision, recall)
    return {
        "tp": recall_tp,
        "actual": total_pred_tokens,
        "possible": total_gold_tokens,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def get_only_partial(preds, gold, list_of_pos_label_types, without_type):
    preds_i2b2 = [pred[0] for pred in preds if pred[1] == "i2b2"]
    gold_i2b2 = [g_[0] for g_ in gold if g_[1] == "i2b2"]
    i2b2_partial_F1 = get_partial_F1_index_level(
        preds_i2b2, gold_i2b2, list_of_pos_label_types, without_type
    )
    preds_mimic = [pred[0] for pred in preds if pred[1] == "mimic"]
    gold_mimic = [g_[0] for g_ in gold if g_[1] == "mimic"]
    mimic_partial_F1 = get_partial_F1_index_level(
        preds_mimic, gold_mimic, list_of_pos_label_types, without_type
    )
    preds = [pred[0] for pred in preds]
    gold = [g_[0] for g_ in gold]
    partial_F1 = get_partial_F1_index_level(
        preds, gold, list_of_pos_label_types, without_type
    )
    return {
        "i2b2": {"partial": i2b2_partial_F1},
        "mimic": {"partial": mimic_partial_F1},
        "all": {"partial": partial_F1},
    }


def get_only_K_F1(preds, gold, list_of_pos_label_types):
    preds_mimic = [pred[0] for pred in preds]
    gold_mimic = [g_[0] for g_ in gold]
    mimic_partial_F1 = get_K_F1(preds_mimic, gold_mimic, list_of_pos_label_types)
    return {"all": {"K": mimic_partial_F1}}


def get_evaluation(preds, gold, list_of_pos_label_types):
    """
    Inputs
    -----------------
		-preds: Tuple (predictions, source), where
		    predictions consist of a 3D list of strings, where each list consists of the
		    document predictions, and each document prediction consists of sentence-level predictions, and
		    each list of sentence-level predictions consist of token-level predictions, where each token-level
		    prediction can have more than one type of followup label in the case of finegrained predictions.
		    source is a string of either 'i2b2' or 'mimic'
		-golds: Tuple (prediction, source), with the same type as preds (except with gold).
	Returns
	------------------
	Dict of strict, exact, partial, and type F1 scores and breakdowns for the data (as well as i2b2 and MIMIC subsets).
	"""
    preds_i2b2 = [pred[0] for pred in preds if pred[1] == "i2b2"]
    gold_i2b2 = [g_[0] for g_ in gold if g_[1] == "i2b2"]
    i2b2_strict_F1 = get_strict_F1(preds_i2b2, gold_i2b2, list_of_pos_label_types)
    i2b2_exact_F1 = get_exact_F1(preds_i2b2, gold_i2b2, list_of_pos_label_types)
    i2b2_partial_F1 = get_partial_F1_index_level(
        preds_i2b2, gold_i2b2, list_of_pos_label_types, True
    )
    i2b2_type_F1 = get_partial_F1_index_level(
        preds_i2b2, gold_i2b2, list_of_pos_label_types
    )
    preds_mimic = [pred[0] for pred in preds if pred[1] == "mimic"]
    gold_mimic = [g_[0] for g_ in gold if g_[1] == "mimic"]
    mimic_strict_F1 = get_strict_F1(preds_mimic, gold_mimic, list_of_pos_label_types)
    mimic_exact_F1 = get_exact_F1(preds_mimic, gold_mimic, list_of_pos_label_types)
    mimic_partial_F1 = get_partial_F1_index_level(
        preds_mimic, gold_mimic, list_of_pos_label_types, True
    )
    mimic_type_F1 = get_partial_F1_index_level(
        preds_mimic, gold_mimic, list_of_pos_label_types
    )
    preds = [pred[0] for pred in preds]
    gold = [g_[0] for g_ in gold]
    strict_F1 = get_strict_F1(preds, gold, list_of_pos_label_types)
    exact_F1 = get_exact_F1(preds, gold, list_of_pos_label_types)
    partial_F1 = get_partial_F1_index_level(preds, gold, list_of_pos_label_types, True)
    type_F1 = get_partial_F1_index_level(preds, gold, list_of_pos_label_types)

    return {
        "i2b2": {
            "strict": i2b2_strict_F1,
            "exact": i2b2_exact_F1,
            "partial": i2b2_partial_F1,
            "type": i2b2_type_F1,
        },
        "mimic": {
            "strict": mimic_strict_F1,
            "exact": mimic_exact_F1,
            "partial": mimic_partial_F1,
            "type": mimic_type_F1,
        },
        "all": {
            "strict": strict_F1,
            "exact": exact_F1,
            "partial": partial_F1,
            "type": type_F1,
        },
    }
