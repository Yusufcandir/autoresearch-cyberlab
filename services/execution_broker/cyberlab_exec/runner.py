"""Subprocess entry point for one experiment. Invoked only by the broker with a fixed template."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from cyberlab_schemas import ExperimentManifest
from cyberlab_tabular import run_experiment


def main(argv: list[str]) -> int:
    manifest_path, prepared_dir, out_dir = (Path(a) for a in argv[:3])
    manifest = ExperimentManifest.model_validate_json(manifest_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_experiment(manifest, prepared_dir, out_dir)
    except Exception as exc:  # noqa: BLE001 - recorded as the job failure reason
        (out_dir / "error.json").write_text(
            json.dumps({"error": repr(exc), "traceback": traceback.format_exc()[-4000:]})
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
