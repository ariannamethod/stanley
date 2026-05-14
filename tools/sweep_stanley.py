#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path


SWEEP = [
    ("baseline", {}),
    ("permissive-silence", {"coherence_floor": 0.05}),
    ("strict-silence", {"coherence_floor": 0.35}),
    ("cool-rings", {"ring_temp_scale": 0.75}),
    ("hot-rings", {"ring_temp_scale": 1.25}),
    ("short-rings", {"ring_len_scale": 0.60}),
    ("long-rings", {"ring_len_scale": 1.40}),
    ("single-ring", {"max_rings": 1}),
    ("deep-hot", {"max_rings": 5, "ring_temp_scale": 1.15, "ring_len_scale": 1.20}),
    ("eager-graze", {"graze_rate": 0.80}),
    ("somatic-temp", {"somatic_temp": True, "somatic_temp_strength": 0.50}),
    ("metastanley", {"metastanley": True, "metastanley_rate": 0.70}),
]


SUMMARY_KEYS = [
    "spoken",
    "silent",
    "collapsed replies",
    "origin 5-gram echoes",
    "avg spoken tokens",
    "avg glue ratio",
    "speak_ratio",
    "coherence_floor",
    "temp_scale",
    "len_scale",
    "graze_rate",
    "temp_factor",
    "scars",
    "scar_pressure",
    "inner_ticks",
]


def metric(report: str, key: str) -> str:
    pattern = rf"- {re.escape(key)}: `([^`]+)`"
    match = re.search(pattern, report)
    return match.group(1) if match else ""


def add_optional(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def run_cell(args: argparse.Namespace, name: str, params: dict[str, object], out: Path) -> str:
    cmd = [
        sys.executable,
        str(args.eval_script),
        "--binary",
        str(args.binary),
        "--origin",
        str(args.origin),
        "--out",
        str(out),
        "--seed",
        str(args.seed),
    ]
    if args.no_origin:
        cmd.append("--no-origin")
    if args.prompts:
        cmd.extend(["--prompts", str(args.prompts)])
    for pasture in args.graze:
        cmd.extend(["--graze", pasture])
    for profile in args.graze_profile:
        cmd.extend(["--graze-profile", profile])

    add_optional(cmd, "--coherence-floor", params.get("coherence_floor"))
    add_optional(cmd, "--max-rings", params.get("max_rings"))
    add_optional(cmd, "--ring-temp-scale", params.get("ring_temp_scale"))
    add_optional(cmd, "--ring-len-scale", params.get("ring_len_scale"))
    add_optional(cmd, "--graze-rate", params.get("graze_rate"))
    if params.get("somatic_temp"):
        cmd.append("--somatic-temp")
    add_optional(cmd, "--somatic-temp-strength", params.get("somatic_temp_strength"))
    if params.get("metastanley"):
        cmd.append("--metastanley")
    add_optional(cmd, "--metastanley-rate", params.get("metastanley_rate"))

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    report = out.read_text(errors="ignore")
    print(f"{name}: {out}")
    return report


def render_index(args: argparse.Namespace, rows: list[tuple[str, dict[str, object], str]]) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Stanley Listening Sweep")
    lines.append("")
    lines.append(f"- generated: `{now}`")
    lines.append(f"- binary: `{args.binary}`")
    lines.append(f"- origin: `{args.origin if not args.no_origin else '--no-origin'}`")
    lines.append(f"- seed: `{args.seed}`")
    if args.graze:
        lines.append(f"- graze: `{', '.join(args.graze)}`")
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    headers = ["cell", "params"] + SUMMARY_KEYS + ["report"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for name, params, report_path in rows:
        report = Path(report_path).read_text(errors="ignore")
        param_text = ", ".join(f"{k}={v}" for k, v in params.items()) or "default"
        cells = [name, param_text]
        cells.extend(metric(report, key) for key in SUMMARY_KEYS)
        cells.append(Path(report_path).name)
        safe = [str(cell).replace("|", "\\|") for cell in cells]
        lines.append("| " + " | ".join(safe) + " |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("Compare cells by silence, collapse, glue, origin echo, and spoken-token length.")
    lines.append("A useful Stanley adapter experiment should start from a cell that changes trajectory without raising collapse.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run a Dario-style listening-condition sweep for Stanley.")
    parser.add_argument("--binary", default=root / "stanley", type=Path)
    parser.add_argument("--eval-script", default=root / "tools" / "eval_stanley.py", type=Path)
    parser.add_argument("--origin", default=root / "origin.txt", type=Path)
    parser.add_argument("--no-origin", action="store_true")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--graze", action="append", default=[])
    parser.add_argument("--graze-profile", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42069)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (root / "evals" / f"listening-{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, dict[str, object], str]] = []
    for name, params in SWEEP:
        report_path = out_dir / f"{name}.md"
        run_cell(args, name, params, report_path)
        rows.append((name, params, str(report_path)))

    index = render_index(args, rows)
    index_path = out_dir / "index.md"
    index_path.write_text(index)
    print(f"index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
