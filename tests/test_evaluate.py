# tests/test_evaluate.py

from src.evaluation.evaluate import (
    Evaluator
)


def test_evaluate():

    evaluator = Evaluator(
    checkpoint_path=
    "weights/checkpoints/best_model.pth",

    test_dir=
    "data/processed/sequences"
)

    results = evaluator.evaluate()

    print(results)


if __name__ == "__main__":
    test_evaluate()