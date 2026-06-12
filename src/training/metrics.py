import torch


def top1_accuracy(logits, targets):
    """
    Top-1 Accuracy
    """

    preds = torch.argmax(logits, dim=1)

    correct = (preds == targets).sum().item()

    total = targets.size(0)

    return correct / total


def topk_accuracy(
    logits,
    targets,
    k=5
):
    """
    Top-K Accuracy
    """

    k = min(k, logits.size(1))

    _, pred = logits.topk(
        k,
        dim=1
    )

    correct = pred.eq(
        targets.view(-1, 1)
    )

    correct_total = correct.sum().item()

    total = targets.size(0)

    return correct_total / total


def precision_recall_f1(
    logits,
    targets
):
    """
    Macro Precision, Recall, F1
    """

    preds = torch.argmax(
        logits,
        dim=1
    )

    classes = torch.unique(targets)

    precisions = []
    recalls = []
    f1s = []

    for cls in classes:

        tp = (
            (preds == cls)
            &
            (targets == cls)
        ).sum().item()

        fp = (
            (preds == cls)
            &
            (targets != cls)
        ).sum().item()

        fn = (
            (preds != cls)
            &
            (targets == cls)
        ).sum().item()

        precision = tp / (tp + fp + 1e-8)

        recall = tp / (tp + fn + 1e-8)

        f1 = (
            2 * precision * recall
            /
            (precision + recall + 1e-8)
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "precision": sum(precisions) / len(precisions),
        "recall": sum(recalls) / len(recalls),
        "f1": sum(f1s) / len(f1s),
    }