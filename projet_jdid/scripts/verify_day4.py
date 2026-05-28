"""Day 4 verification for SIL results, ONNX export, and dashboard wiring."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
from stable_baselines3 import PPO


def main() -> None:
    model_path = ROOT / "models" / "best_model.zip"
    assert model_path.is_file(), f"Missing {model_path}"
    model = PPO.load(model_path)
    assert model.action_space.n == 4
    print("ppo_model=PASS")

    results_path = ROOT / "eval" / "results" / "sil_results.json"
    assert results_path.is_file(), f"Missing {results_path}"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["routes"], "No evaluated routes"
    route = payload["routes"][0]
    for policy in ("ppo", "rule", "normal"):
        assert policy in route["policies"]
    for key in (
        "ppo_vs_rule_fuel_savings_pct",
        "ppo_vs_rule_co2_savings_pct",
        "ppo_vs_normal_fuel_savings_pct",
        "ppo_vs_normal_co2_savings_pct",
    ):
        assert key in route["comparisons"]
    print("sil_results=PASS")

    onnx_path = ROOT / "models" / "ths_policy.onnx"
    assert onnx_path.is_file(), f"Missing {onnx_path}"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"obs": np.zeros((1, 8), dtype=np.float32)})
    assert len(out) == 1
    assert out[0].shape == (1, 4)
    print("onnx_runtime=PASS")

    dashboard = (ROOT / "app" / "streamlit_dashboard.py").read_text(encoding="utf-8")
    assert "Evaluation JSON" in dashboard
    assert "SIL Evaluation" in dashboard
    print("dashboard_metrics=PASS")
    print("day4_checks=PASS")


if __name__ == "__main__":
    main()

