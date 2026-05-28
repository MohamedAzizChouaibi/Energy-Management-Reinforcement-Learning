"""Day 5 verification for GPS fine-tuning, ONNX, and pre-trip planning."""

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

from training.train_ppo import RandomRouteTHSEnv


VALID_MODES = {"EV", "ECO", "NORMAL", "PWR"}


def main() -> None:
    env = RandomRouteTHSEnv(["gps/cache/sample_phase0_route.json"], seed=7)
    first = env.route_cache
    env.reset()
    second = env.route_cache
    assert first and second
    print("gps_route_pool_reset=PASS")

    gps_model = ROOT / "models" / "ths_agent_gps.zip"
    assert gps_model.is_file(), f"Missing {gps_model}"
    model = PPO.load(gps_model)
    assert model.action_space.n == 4
    print("gps_model=PASS")

    onnx_path = ROOT / "models" / "ths_policy_gps.onnx"
    assert onnx_path.is_file(), f"Missing {onnx_path}"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = sess.run(None, {"obs": np.zeros((1, 8), dtype=np.float32)})
    assert outputs[0].shape == (1, 4)
    print("gps_onnx=PASS")

    plan_path = ROOT / "gps" / "pre_trip_plan.json"
    assert plan_path.is_file(), f"Missing {plan_path}"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["segments"]
    assert "gps_anticipation_rate_pct" in plan
    for seg in plan["segments"]:
        assert seg["recommended_mode"] in VALID_MODES
        assert "expected_CO2_g" in seg
        assert seg["expected_CO2_g"] >= 0.0
    print("pre_trip_plan=PASS")

    dashboard = (ROOT / "app" / "streamlit_dashboard.py").read_text(encoding="utf-8")
    assert "Pre-trip plan JSON" in dashboard
    assert "Pre-Trip Plan" in dashboard
    print("dashboard_pre_trip=PASS")
    print("day5_checks=PASS")


if __name__ == "__main__":
    main()
