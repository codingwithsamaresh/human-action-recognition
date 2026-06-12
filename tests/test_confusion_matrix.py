from src.evaluation.confusion_matrix import (
    ConfusionMatrixEvaluator
)


def test_confusion_matrix():

    evaluator = (
        ConfusionMatrixEvaluator(
            checkpoint_path=
            "weights/checkpoints/best_model.pth",

            dataset_dir=
            "data/processed/sequences"
        )
    )

    cm = evaluator.generate()

    print(cm)


if __name__ == "__main__":
    test_confusion_matrix()