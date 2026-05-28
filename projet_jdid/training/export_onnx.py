"""Export an SB3 PPO THS-II policy to ONNX and validate with ONNX Runtime."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
from stable_baselines3 import PPO
import torch


class PPOLogitsWrapper(torch.nn.Module):
    """Small export wrapper returning 4 action logits from an SB3 PPO policy."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        dist = self.policy.get_distribution(obs)
        return dist.distribution.logits


def export_onnx(model_path: str, output_path: str, opset: int = 17) -> dict:
    model = PPO.load(model_path, device="cpu")
    model.policy.set_training_mode(False)
    wrapper = PPOLogitsWrapper(model.policy)
    wrapper.eval()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    obs_ex = torch.zeros(1, 8, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        obs_ex,
        str(output),
        opset_version=opset,
        input_names=["obs"],
        output_names=["action_logits"],
        dynamic_axes={"obs": {0: "batch"}, "action_logits": {0: "batch"}},
    )
    return validate_onnx(str(output))


def validate_onnx(path: str, n_runs: int = 100) -> dict:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    obs = np.zeros((1, 8), dtype=np.float32)
    outputs = sess.run(None, {"obs": obs})
    if len(outputs) != 1:
        raise ValueError(f"Expected one ONNX output, got {len(outputs)}")
    if outputs[0].shape != (1, 4):
        raise ValueError(f"Expected ONNX output shape (1, 4), got {outputs[0].shape}")

    t0 = time.perf_counter()
    for _ in range(max(1, n_runs)):
        sess.run(None, {"obs": obs})
    latency_ms = (time.perf_counter() - t0) / max(1, n_runs) * 1000.0
    return {
        "onnx_path": path,
        "input_shape": [1, 8],
        "output_shape": list(outputs[0].shape),
        "latency_ms": latency_ms,
        "logits": outputs[0].tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SB3 PPO policy to ONNX.")
    parser.add_argument("--model", default="models/best_model.zip")
    parser.add_argument("--output", default="models/ths_policy.onnx")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_onnx(args.model, args.output, opset=args.opset)
    print(result)


if __name__ == "__main__":
    main()

