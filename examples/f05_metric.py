"""The competition metric, written out so you can see every step.

Macro F0.5: compute F0.5 for every Source 1 record, then take the plain average.

    python examples/f05_metric.py

No dependencies (standard library only).
"""


def f05_one(predicted: set, truth: set) -> float:
    """F0.5 for ONE Source 1 record.

    - Nothing true and nothing predicted (a correctly spotted singleton) -> 1.0
    - Something predicted but none of it right (or a singleton we wrongly matched) -> 0.0
    - Otherwise the usual formula with beta = 0.5 (precision counts twice as much as recall).
    """
    if not predicted and not truth:
        return 1.0
    tp = len(predicted & truth)          # true positives: predicted AND correct
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)      # of what we said, how much was right
    recall = tp / len(truth)             # of what was true, how much we found
    beta2 = 0.5 ** 2                     # 0.25
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def macro_f05(pred: dict, truth: dict) -> float:
    """Average of f05_one over every Source 1 id in `truth` (singletons included)."""
    return sum(f05_one(pred.get(s1, set()), t) for s1, t in truth.items()) / len(truth)


if __name__ == "__main__":
    # 1. The worked example from the problem statement
    p = {"S2-00047", "S2-00193", "S3-00812"}
    t = {"S2-00047", "S3-00812"}
    print(f"official example            : {f05_one(p, t):.3f}   (statement says 0.714)")

    # 2. Why precision matters more: one wrong extra vs one missed
    t = {"A", "B", "C", "D"}
    print(f"all 4 found + 1 wrong extra : {f05_one({'A', 'B', 'C', 'D', 'X'}, t):.3f}")
    print(f"3 of 4 found, nothing wrong : {f05_one({'A', 'B', 'C'}, t):.3f}")
    print(f"2 of 4 found, nothing wrong : {f05_one({'A', 'B'}, t):.3f}")

    # 3. Singletons are all-or-nothing
    print(f"singleton, predict nothing  : {f05_one(set(), set()):.3f}")
    print(f"singleton, predict anything : {f05_one({'X'}, set()):.3f}")

    # 4. Macro average: every Source 1 record weighs the same, however many copies it has
    truth = {"S1-1": {"a", "b", "c", "d", "e", "f"}, "S1-2": set(), "S1-3": {"g"}}
    pred = {"S1-1": {"a", "b", "c", "d", "e", "f"}, "S1-2": {"z"}, "S1-3": {"g"}}
    print(f"macro example               : {macro_f05(pred, truth):.3f}   (one false merge on a singleton costs 1/3)")
