from src.evaluation.benchmark import (
    Benchmark
)


def test_benchmark():

    benchmark = Benchmark(
        checkpoint_path=
        "weights/checkpoints/best_model.pth",

        num_classes=1
    )

    report = benchmark.run()

    print(report)


if __name__ == "__main__":
    test_benchmark()