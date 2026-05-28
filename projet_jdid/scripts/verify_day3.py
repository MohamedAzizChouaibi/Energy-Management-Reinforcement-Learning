"""Day 3 verification for PPO training artifacts and route-cache-only setup."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from gps.cache_utils import load_tomtom_cache


def main() -> None:
    cache_files = sorted((ROOT / "gps" / "cache").glob("*.json"))
    valid_caches = []
    for path in cache_files:
        try:
            load_tomtom_cache(path)
        except Exception:
            continue
        valid_caches.append(path)
    print(f"valid_route_caches={len(valid_caches)}")
    assert valid_caches, "No valid TomTom route caches found"

    train_script = (ROOT / "training" / "train_ppo.py").read_text(encoding="utf-8")
    forbidden = ("WLTC", "FTP75", "US06", "drive_cycles")
    assert not any(token in train_script for token in forbidden)
    print("no_csv_training_refs=PASS")

    best_model = ROOT / "models" / "best_model.zip"
    final_model = ROOT / "models" / "final_model.zip"
    assert best_model.is_file(), f"Missing {best_model}"
    assert final_model.is_file(), f"Missing {final_model}"
    print("model_files=PASS")

    model = PPO.load(best_model)
    assert model.action_space.n == 4
    print(f"action_space={model.action_space}")

    runs_dir = ROOT / "runs"
    assert runs_dir.is_dir(), "Missing TensorBoard runs/ directory"
    run_files = [p for p in runs_dir.rglob("*") if p.is_file()]
    assert run_files, "TensorBoard runs/ contains no files"
    print("tensorboard_logs=PASS")
    print("day3_checks=PASS")


if __name__ == "__main__":
    main()

