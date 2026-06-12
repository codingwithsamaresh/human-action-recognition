from src.evaluation.roc_curve import (
    ROCEvaluator
)


def test_roc_curve():

    evaluator = ROCEvaluator(
        checkpoint_path=
        "weights/checkpoints/best_model.pth",

        dataset_dir=
        "data/test"
    )

    result = evaluator.generate()

    print(result)


if __name__ == "__main__":
    test_roc_curve()