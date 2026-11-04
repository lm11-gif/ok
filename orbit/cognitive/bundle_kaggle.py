#!/usr/bin/env python3
"""Bundle Cognitive v1 modules into one self-contained Kaggle script."""
from __future__ import annotations

import argparse
from pathlib import Path


def strip_future_import(source: str) -> str:
    return source.replace("from __future__ import annotations\n", "", 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    core = (root / "core.py").read_text(encoding="utf-8")
    models = strip_future_import((root / "models.py").read_text(encoding="utf-8"))
    trainer = strip_future_import((root / "train.py").read_text(encoding="utf-8"))

    import_block = """try:
    from .core import build_seed_data, encode_examples, encode_facts
    from .models import PerceptionNet, AssociationNet, MemoryNet
except ImportError:  # Kaggle script mode: modules are side-by-side.
    from core import build_seed_data, encode_examples, encode_facts
    from models import PerceptionNet, AssociationNet, MemoryNet
"""
    if import_block not in trainer:
        raise SystemExit("Could not locate the expected module-import block in train.py.")
    trainer = trainer.replace(import_block, "", 1)

    # The core module's future import must remain the first Python statement.
    bundled = "\n".join([
        "# Generated self-contained Kaggle entry point. Do not edit directly.",
        core.rstrip(),
        models.rstrip(),
        trainer.lstrip(),
        "",
    ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(bundled, encoding="utf-8")
    compile(bundled, str(output), "exec")
    print(f"Bundled Kaggle entry point: {output} ({len(bundled)} characters)")


if __name__ == "__main__":
    main()
