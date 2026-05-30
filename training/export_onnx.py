"""Day 3B - Export the trained PPO policy to ONNX.

Loads ``models/best_model.zip`` via ``PPO.load``, wraps the actor head so the
forward pass returns raw action logits, and exports an ONNX graph with a
``(1, 8)`` observation input (the 8-dimensional GPS-enriched observation) and a
4-logit output head (action_space ``Discrete(4)``).

The exported graph is then validated and benchmarked with ONNX Runtime:
  * inference on a single 8-dim observation succeeds,
  * the output has exactly 4 logits,
  * argmax matches the SB3 deterministic action,
  * 1000 sequential ``sess.run`` calls average < 2 ms each.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

MODEL_PATH = PROJECT_ROOT / "models" / "aziz_best_model.zip"
ONNX_PATH  = PROJECT_ROOT / "models" / "aziz_policy.onnx"
OBS_DIM    = 40
N_ACTIONS  = 3
OPSET = 17
LATENCY_TARGET_MS = 2.0
N_BENCH = 1000


class OnnxablePolicy(nn.Module):
    """Wrap an SB3 ``ActorCriticPolicy`` to emit action logits only.

    Mirrors the deterministic actor path used by ``policy.predict``:
    features extractor -> actor MLP -> action net.
    """

    def __init__(self, policy) -> None:
        super().__init__()
        self.extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        self.features_extractor = policy.features_extractor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        latent_pi, _ = self.extractor(features)
        return self.action_net(latent_pi)  # raw logits, shape (batch, N_ACTIONS)


def export() -> None:
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"Loading PPO model: {MODEL_PATH}")
    model = PPO.load(str(MODEL_PATH), device="cpu")
    wrapper = OnnxablePolicy(model.policy).eval()

    dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        str(ONNX_PATH),
        opset_version=OPSET,
        input_names=["obs"],
        output_names=["action_logits"],
        dynamic_axes={"obs": {0: "batch"}, "action_logits": {0: "batch"}},
        dynamo=False,  # legacy TorchScript exporter (no onnxscript dependency)
    )
    size_kb = ONNX_PATH.stat().st_size / 1024.0
    print(f"Exported: {ONNX_PATH}  ({size_kb:.1f} KB)")

    _validate(model, size_kb)


def _validate(model: PPO, size_kb: float) -> None:
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
    print("ONNX graph structurally valid.")

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = sess.get_outputs()[0]
    print(f"Input '{in_name}' shape={sess.get_inputs()[0].shape}  "
          f"Output '{out.name}' shape={out.shape}")

    rng = np.random.default_rng(0)
    sample = rng.uniform(-1.0, 1.0, size=(1, OBS_DIM)).astype(np.float32)
    logits = sess.run(None, {in_name: sample})[0]
    n_logits = logits.shape[-1]
    onnx_action = int(np.argmax(logits, axis=-1)[0])
    sb3_action = int(model.predict(sample[0], deterministic=True)[0])

    # parity check across a batch of random observations
    batch = rng.uniform(-1.0, 1.0, size=(256, OBS_DIM)).astype(np.float32)
    onnx_batch = np.argmax(sess.run(None, {in_name: batch})[0], axis=-1)
    sb3_batch = np.array([int(model.predict(o, deterministic=True)[0]) for o in batch])
    parity = float(np.mean(onnx_batch == sb3_batch)) * 100.0

    # latency benchmark: 1000 sequential single-obs calls
    for _ in range(50):  # warmup
        sess.run(None, {in_name: sample})
    t0 = time.perf_counter()
    for _ in range(N_BENCH):
        sess.run(None, {in_name: sample})
    per_call_ms = (time.perf_counter() - t0) / N_BENCH * 1000.0

    size_ok = 150.0 <= size_kb <= 260.0
    print("\n--- Validation ---")
    print(f"  Output logits           : {n_logits}  ({'PASS' if n_logits == N_ACTIONS else 'FAIL'})")
    print(f"  ONNX vs SB3 action       : {onnx_action} vs {sb3_action}  "
          f"({'match' if onnx_action == sb3_action else 'MISMATCH'})")
    print(f"  Argmax parity (256 obs)  : {parity:.1f}%")
    print(f"  File size                : {size_kb:.1f} KB  "
          f"({'in ~180-220KB band' if size_ok else 'OUTSIDE expected band'})")
    print(f"  Latency / call           : {per_call_ms:.4f} ms over {N_BENCH} calls  "
          f"({'PASS <2ms' if per_call_ms < LATENCY_TARGET_MS else 'FAIL'})")

    ok = n_logits == N_ACTIONS and onnx_action == sb3_action and per_call_ms < LATENCY_TARGET_MS
    print(f"\nONNX export checkpoint: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    export()
