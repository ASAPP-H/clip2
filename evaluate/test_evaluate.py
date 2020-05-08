from evaluation import get_strict_F1, get_partial_F1_index_level, get_exact_F1, get_K_F1
from third_party_utils import *

gold_truths = [
    [["SOS"], ["B-FU2", "B-FU1"], ["I-FU2", "I-FU1"], ["O"], ["O"], ["B-FU4"], ["EOS"]]
]
preds = [
    [
        ["SOS"],
        ["B-FU1"],
        ["I-FU1", "B-FU3"],
        ["I-FU3", "B-FU4"],
        ["I-FU4", "I-FU3"],
        ["B-FU4"],
        ["EOS"],
    ]
]
expanded, _, total_f1 = get_strict_F1(preds, gold_truths, ["FU1", "FU2", "FU3", "FU4"])
assert expanded["FU1"]["strict"]["correct"] == 1
assert expanded["FU2"]["strict"]["actual"] == 0
assert expanded["FU1"]["strict"]["precision"] == 1.0
assert expanded["FU4"]["strict"]["correct"] == 1
assert expanded["FU4"]["strict"]["precision"] == 0.5
assert expanded["FU4"]["strict"]["recall"] == 1.0
assert expanded["FU1"]["strict"]["recall"] == 1.0
assert expanded["FU2"]["strict"]["recall"] == 0.0
assert expanded["FU3"]["strict"]["recall"] == 0.0
assert expanded["FU3"]["strict"]["precision"] == 0.0
assert expanded["FU2"]["strict"]["f1"] == 0
assert expanded["FU1"]["strict"]["f1"] == 1
assert expanded["FU4"]["strict"]["f1"] == 2 / 3
assert total_f1["strict"]["correct"] == 2
assert total_f1["strict"]["precision"] == 0.5
assert total_f1["strict"]["recall"] == 2 / 3

"""
This is for the case 
gold = [Entity("FU2", 1, 2), Entity("FU4", 5, 5)]
pred = [Entity("FU2", 1, 2), Entity("FU4", 3, 5)]
"""
gold_truths = [[["SOS"], ["B-FU2"], ["I-FU2"], ["O"], ["O"], ["B-FU4"], ["EOS"]]]
preds = [[["SOS"], ["B-FU2"], ["I-FU2"], ["B-FU4"], ["I-FU4"], ["I-FU4"], ["EOS"]]]
expanded, _, total_f1 = get_strict_F1(preds, gold_truths, ["FU1", "FU2", "FU3", "FU4"])
assert expanded["FU2"]["strict"]["precision"] == 1.0
assert expanded["FU2"]["strict"]["recall"] == 1.0
assert expanded["FU4"]["strict"]["precision"] == 0.0
assert expanded["FU4"]["strict"]["recall"] == 0.0
assert expanded["FU4"]["strict"]["actual"] == 1
assert total_f1["strict"]["f1"] == 0.5

gold_truths = [[["SOS"], ["B-FU2"], ["I-FU2"], ["EOS"]]]
preds = [[["SOS"], ["B-FU1"], ["I-FU1"], ["EOS"]]]
expanded, _, total_f1 = get_strict_F1(preds, gold_truths, ["FU1", "FU2", "FU3", "FU4"])
assert expanded["FU1"]["strict"]["correct"] == 0

"""
Strict-f1 for binary case
"""
gold_truths = [
    ["SOS", "O", "B-followup", "I-followup", "O", "B-followup", "B-followup", "EOS"]
]
preds = [["SOS", "O", "B-followup", "I-followup", "O", "B-followup", "O", "EOS"]]
expanded, _, total_f1 = get_strict_F1(preds, gold_truths, ["followup"])
assert expanded["followup"]["strict"]["correct"] == 1
assert total_f1["strict"]["f1"] == 0.5

"""
Partial-F1 evaluation

Ground truth: [(1, 3, “Appointment followup”), (1, 3, “Lab followup”), (3, 4, “Other followup”))]
rediction spans: [(1, 3, “Appointment followup”), (3, 4, “Other followup”), (2, 6,”Imaging followup”)]
"""
gold_truths = [
    [["SOS"], ["B-FU2", "B-FU1"], ["I-FU2", "I-FU1"], ["O"], ["O"], ["B-FU4"], ["EOS"]]
]
preds = [
    [
        ["SOS"],
        ["B-FU1"],
        ["I-FU1", "B-FU3"],
        ["I-FU3", "B-FU4"],
        ["I-FU4", "I-FU3"],
        ["B-FU4"],
        ["EOS"],
    ]
]
partial = get_partial_F1_index_level(preds, gold_truths, ["FU1", "FU2", "FU3", "FU4"])

assert partial["tp"] == 4
assert partial["actual"] == 8
assert partial["possible"] == 5


gold_truths = [[["SOS"], ["B-FU2", "B-FU1"], ["O"], ["O"], ["EOS"]]]
preds = [[["SOS"], ["O"], ["O"], ["B-FU2"], ["EOS"]]]
partial = get_partial_F1_index_level(preds, gold_truths, ["FU1", "FU2", "FU3", "FU4"])
assert partial["tp"] == 0
assert partial["actual"] == 1
assert partial["possible"] == 2

gold_truths = [["B-followup", "B-followup", "O", "O"]]
preds = [["O", "O", "B-followup"]]
partial = get_partial_F1_index_level(preds, gold_truths, ["followup"])
assert partial["tp"] == 0
assert partial["actual"] == 1
assert partial["possible"] == 2

gold_truths = [["B-followup", "B-followup", "O", "O"]]
preds = [["O", "B-followup", "O"]]
partial = get_partial_F1_index_level(preds, gold_truths, ["followup"])
assert partial["tp"] == 1
assert partial["actual"] == 1
assert partial["possible"] == 2

gold_truths = [["B-FU1", "B-FU1", "O", "O"]]
preds = [["O", "B-FU2", "O"]]
partial = get_partial_F1_index_level(preds, gold_truths, ["FU1", "FU2"])
assert partial["tp"] == 1
assert partial["actual"] == 1
assert partial["possible"] == 2


"""
Exact-F1 evaluation 
"""
# this should be equal to the strict case
gold_truths = [
    [["SOS"], ["B-FU2", "B-FU1"], ["I-FU2", "I-FU1"], ["O"], ["O"], ["B-FU4"], ["EOS"]]
]
preds = [
    [
        ["SOS"],
        ["B-FU1"],
        ["I-FU1", "B-FU3"],
        ["I-FU3", "B-FU4"],
        ["I-FU4", "I-FU3"],
        ["B-FU4"],
        ["EOS"],
    ]
]

exact = get_exact_F1(preds, gold_truths, ["FU1", "FU2", "FU3", "FU4"])
assert exact["tp"] == 2
assert exact["actual"] == 4
assert exact["possible"] == 3


"""
This is for the case 
gold_truths = [Entity("FU2", 1, 2), Entity("FU1", 1, 2), Entity("FU4", 5, 5)]
preds = [Entity("FU4", 1, 2), Entity("FU4", 3, 4), Entity("FU3", 2, 4), Entity("FU4", 5, 5)]
This is different from the above in that (0,1) prediction is not the correct type. 
"""

gold_truths = [
    [["SOS"], ["B-FU2", "B-FU1"], ["I-FU2", "I-FU1"], ["O"], ["O"], ["B-FU4"], ["EOS"]]
]
preds = [
    [
        ["SOS"],
        ["B-FU1"],
        ["I-FU1", "B-FU3"],
        ["I-FU3", "B-FU4"],
        ["I-FU4", "I-FU3"],
        ["B-FU4"],
        ["EOS"],
    ]
]
exact = get_exact_F1(preds, gold_truths, ["FU1", "FU2", "FU3", "FU4"])
assert exact["tp"] == 2
assert exact["actual"] == 4
assert exact["possible"] == 3

# partial should not be considered correct
gold_truths = [[["SOS"], ["B-FU2"], ["I-FU2"], ["O"], ["EOS"]]]
preds = [[["SOS"], ["B-FU2"], ["I-FU2"], ["I-FU2"], ["EOS"]]]
exact = get_exact_F1(preds, gold_truths, ["FU1", "FU2", "FU3", "FU4"])
assert exact["tp"] == 0
assert exact["actual"] == 1
assert exact["possible"] == 1

gold_truths = [["SOS", "B-followup", "I-followup", "O", "EOS"]]
preds = [["SOS", "B-followup", "I-followup", "I-followup", "EOS"]]
exact = get_exact_F1(preds, gold_truths, ["followup"])
assert exact["tp"] == 0
assert exact["actual"] == 1
assert exact["possible"] == 1

"""
Get K-F1
"""
""""
gold_truths = [[["SOS"], ["O"], ["I-FU2"], ["I-FU2"], ["O"], ["B-FU4"], ["EOS"]]]
preds = [[["SOS"], ["B-FU2"], ["O"], ["O"], ["O"], ["B-FU4"], ["EOS"]]]
exact = get_K_F1(preds, gold_truths, ["FU1", "FU2", "FU3", "FU4"])
assert exact["tp"] == 2  # with leniency constraint
assert exact["actual"] == 2
assert exact["possible"] == 2

gold_truths = [[["SOS"], ["O"], ["B-FU2"], ["I-FU2"], ["O"], ["B-FU4"], ["EOS"]]]
preds = [[["SOS"], ["B-FU2"], ["B-FU2"], ["I-FU2"], ["O"], ["B-FU4"], ["EOS"]]]
exact = get_K_F1(preds, gold_truths, ["FU1", "FU2", "FU3", "FU4"])
assert exact["tp"] == 2  # with leniency constraint
assert exact["actual"] == 3
assert exact["possible"] == 2
"""
