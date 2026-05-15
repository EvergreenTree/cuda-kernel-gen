#!/usr/bin/env python3
"""Run the original baseline benchmark and save a small JSON summary."""

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout


def parse_output(text):
    time_match = re.search(r"^A:\s*([0-9.]+)\s*ms", text, re.MULTILINE)
    return {
        "correct": "Results are correct" in text,
        "time_ms": float(time_match.group(1)) if time_match else None,
        "raw_output": text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./baseline.x")
    parser.add_argument("--output-dir", default="reports/latest")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.build:
        code, text = run(["make", "baseline.x"])
        if code != 0:
            print(text, end="")
            raise SystemExit(code)

    code, text = run([args.binary])
    if code != 0:
        print(text, end="")
        raise SystemExit(code)

    result = parse_output(text)
    result["command"] = args.binary
    result["role"] = "strict_original_problem_definition"
    path = output_dir / "baseline.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
