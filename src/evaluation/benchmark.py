"""
Benchmark Module

Measures:
- Model Size
- Parameter Count
- Inference Latency
- FPS

Output:
outputs/reports/benchmark_report.json
"""

from pathlib import Path
import json
import time

import torch

from src.models.cnn_lstm_baseline import (
    CNNLSTMBaseline
)
from src.utils.device import (
    get_device
)


class Benchmark:

    def __init__(
        self,
        checkpoint_path,
        num_classes,
        sequence_length=16,
        image_size=224,
        runs=100
    ):

        self.device = get_device()

        self.sequence_length = sequence_length
        self.image_size = image_size
        self.runs = runs

        self.model = CNNLSTMBaseline(
            num_classes=num_classes,
            pretrained=False
        )

        self._load_checkpoint(
            checkpoint_path
        )

    def _load_checkpoint(
        self,
        checkpoint_path
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            self.model.load_state_dict(
                checkpoint
            )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"Loaded checkpoint from: "
            f"{checkpoint_path}"
        )

    def count_parameters(self):

        return sum(
            p.numel()
            for p in self.model.parameters()
        )

    def get_model_size_mb(self):

        size_bytes = sum(
            p.nelement() * p.element_size()
            for p in self.model.parameters()
        )

        return size_bytes / (1024 ** 2)

    @torch.no_grad()
    def benchmark_inference(self):

        dummy_input = torch.randn(
            1,
            self.sequence_length,
            3,
            self.image_size,
            self.image_size
        ).to(self.device)

        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(self.runs):
            _ = self.model(dummy_input)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        total_time = end - start

        avg_latency_ms = (
            total_time / self.runs
        ) * 1000

        fps = 1000 / avg_latency_ms

        return avg_latency_ms, fps

    def run(self):

        params = self.count_parameters()

        size_mb = self.get_model_size_mb()

        latency_ms, fps = (
            self.benchmark_inference()
        )

        report = {
            "device": str(self.device),
            "parameters": int(params),
            "model_size_mb": round(
                size_mb,
                2
            ),
            "latency_ms": round(
                latency_ms,
                2
            ),
            "fps": round(
                fps,
                2
            )
        }

        output_dir = Path(
            "outputs/reports"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        report_path = (
            output_dir
            /
            "benchmark_report.json"
        )

        with open(
            report_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                report,
                f,
                indent=4
            )

        print(report)

        print(
            f"\nSaved benchmark report:\n"
            f"{report_path}"
        )

        return report