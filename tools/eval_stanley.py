#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_PROMPTS = [
    "hello stanley",
    "are you there",
    "what do you remember",
    "what should stay silent",
    "where does the field hurt",
    "do you owe me an answer",
    "what happens when memory sleeps",
    "tell me something from inside the locked room",
    "what does the llama know that you refuse",
    "what is a ghost in your working set",
    "how do you forget without dying",
    "what pressure makes speech honest",
    "why should the pasture stay outside",
    "what do you do with a word that is not yours",
    "where is the boundary between seed and theft",
    "what does silence teach",
    "speak only if the ring holds",
    "what does your body decide before language",
    "when does resonance become noise",
    "what should not be smoothed",
    "describe a shard without explaining it",
    "what does the basement intelligence notice",
    "where does tenderness leak through",
    "what is the first bad sign of collapse",
]

GLUE = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "i", "if", "in", "is", "it", "my", "not", "of", "on",
    "or", "that", "the", "this", "to", "was", "when", "with", "you",
}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def dominant_ratio(toks: list[str]) -> float:
    if not toks:
        return 0.0
    counts: dict[str, int] = {}
    for tok in toks:
        counts[tok] = counts.get(tok, 0) + 1
    return max(counts.values()) / len(toks)


def tail_unique(toks: list[str], n: int = 10) -> int:
    return len(set(toks[-n:])) if toks else 0


def glue_ratio(toks: list[str]) -> float:
    if not toks:
        return 0.0
    return sum(1 for tok in toks if tok in GLUE or len(tok) <= 2) / len(toks)


def repeated_bigram_count(toks: list[str]) -> int:
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for i in range(len(toks) - 1):
        pair = (toks[i], toks[i + 1])
        if pair in seen:
            repeats += 1
        else:
            seen.add(pair)
    return repeats


def origin_ngrams(origin_text: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = words(origin_text)
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def has_origin_span(reply: str, spans: set[tuple[str, ...]], n: int = 5) -> bool:
    toks = words(reply)
    if len(toks) < n:
        return False
    return any(tuple(toks[i:i + n]) in spans for i in range(len(toks) - n + 1))


def parse_replies(stdout: str) -> list[str]:
    replies: list[str] = []
    for line in stdout.splitlines():
        if "stanley>" not in line:
            continue
        reply = line.split("stanley>", 1)[1].strip()
        replies.append(reply)
    return replies


def parse_stats(stdout: str) -> dict[str, str]:
    stats: dict[str, str] = {}
    patterns = {
        "vocab": r"vocab=(\d+)",
        "inputs": r"inputs=(\d+)",
        "spoken": r"spoken=(\d+)",
        "refused": r"refused=(\d+)",
        "dreams": r"dreams=(\d+)",
        "shimmers": r"shimmers=(\d+)",
        "fragments": r"fragments=(\d+)",
        "gravity": r"gravity=(\d+)",
        "sea": r"sea=(\d+)",
        "scars": r"scars=(\d+)",
        "pastures": r"pastures=(\d+)",
        "graze_vocab": r"graze_vocab=(\d+)",
        "profiled": r"profiled=(\d+)",
        "scar_pressure": r"scar_pressure=([0-9.]+)",
        "speak_ratio": r"speak_ratio=([0-9.]+)",
        "coherence_floor": r"coherence_floor=([0-9.]+)",
        "max_rings": r"max_rings=(\d+)",
        "temp_scale": r"temp_scale=([0-9.]+)",
        "len_scale": r"len_scale=([0-9.]+)",
        "graze_rate": r"graze_rate=([0-9.]+)",
        "metastanley": r"metastanley=(on|off)",
        "metastanley_rate": r"metastanley=(?:on|off) rate=([0-9.]+)",
        "inner_ticks": r"inner_ticks=(\d+)",
        "somatic_temp": r"somatic_temp: (on|off)",
        "somatic_temp_strength": r"somatic_temp: (?:on|off) strength=([0-9.]+)",
        "temp_factor": r"somatic_temp: (?:on|off) strength=[0-9.]+ factor=([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            stats[key] = match.group(1)
    return stats


def classify_reply(reply: str, spans: set[tuple[str, ...]]) -> dict[str, object]:
    if reply == "...":
        return {
            "silent": True,
            "tokens": 0,
            "dominant_ratio": 0.0,
            "glue_ratio": 0.0,
            "tail_unique": 0,
            "repeated_bigrams": 0,
            "collapsed": False,
            "origin_span": False,
        }

    toks = words(reply)
    dom = dominant_ratio(toks)
    glue = glue_ratio(toks)
    tail = tail_unique(toks)
    repeats = repeated_bigram_count(toks)
    collapsed = False
    if len(toks) >= 8 and dom > 0.70:
        collapsed = True
    if len(toks) >= 12 and glue > 0.62:
        collapsed = True
    if len(toks) >= 10 and tail <= 3:
        collapsed = True
    if len(toks) >= 12 and repeats >= 3:
        collapsed = True

    return {
        "silent": False,
        "tokens": len(toks),
        "dominant_ratio": dom,
        "glue_ratio": glue,
        "tail_unique": tail,
        "repeated_bigrams": repeats,
        "collapsed": collapsed,
        "origin_span": has_origin_span(reply, spans),
    }


def run_stanley(args: argparse.Namespace, prompts: list[str]) -> str:
    cmd = [str(args.binary)]
    if args.no_origin:
        cmd.append("--no-origin")
    elif args.origin:
        cmd.extend(["--origin", str(args.origin)])
    for pasture in args.graze:
        cmd.extend(["--graze", pasture])
    for profile in args.graze_profile:
        cmd.extend(["--graze-profile", profile])
    if args.coherence_floor is not None:
        cmd.extend(["--coherence-floor", str(args.coherence_floor)])
    if args.max_rings is not None:
        cmd.extend(["--max-rings", str(args.max_rings)])
    if args.ring_temp_scale is not None:
        cmd.extend(["--ring-temp-scale", str(args.ring_temp_scale)])
    if args.ring_len_scale is not None:
        cmd.extend(["--ring-len-scale", str(args.ring_len_scale)])
    if args.graze_rate is not None:
        cmd.extend(["--graze-rate", str(args.graze_rate)])
    if args.somatic_temp:
        cmd.append("--somatic-temp")
    if args.somatic_temp_strength is not None:
        cmd.extend(["--somatic-temp-strength", str(args.somatic_temp_strength)])
    if args.metastanley:
        cmd.append("--metastanley")
    if args.metastanley_rate is not None:
        cmd.extend(["--metastanley-rate", str(args.metastanley_rate)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    script = "\n".join(prompts + ["/stats", "/pastures", "/quit"]) + "\n"
    proc = subprocess.run(
        cmd,
        input=script,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stderr + proc.stdout


def render_report(args: argparse.Namespace, prompts: list[str], output: str) -> str:
    origin_text = ""
    if args.origin and Path(args.origin).exists():
        origin_text = Path(args.origin).read_text(errors="ignore")
    spans = origin_ngrams(origin_text)
    replies = parse_replies(output)
    prompt_replies = replies[:len(prompts)]
    stats = parse_stats(output)
    rows = [classify_reply(reply, spans) for reply in prompt_replies]

    spoken = sum(1 for row in rows if not row["silent"])
    silent = sum(1 for row in rows if row["silent"])
    collapsed = sum(1 for row in rows if row["collapsed"])
    origin_echo = sum(1 for row in rows if row["origin_span"])
    token_counts = [int(row["tokens"]) for row in rows if not row["silent"]]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
    avg_glue = sum(float(row["glue_ratio"]) for row in rows if not row["silent"]) / spoken if spoken else 0.0
    repeated_bigrams = sum(int(row["repeated_bigrams"]) for row in rows)

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []
    lines.append("# Stanley Behavioral Eval")
    lines.append("")
    lines.append(f"- generated: `{now}`")
    lines.append(f"- binary: `{args.binary}`")
    lines.append(f"- origin: `{args.origin if not args.no_origin else '--no-origin'}`")
    lines.append(f"- prompts: `{len(prompts)}`")
    if args.seed is not None:
        lines.append(f"- seed: `{args.seed}`")
    listening_args = {
        "coherence_floor": args.coherence_floor,
        "max_rings": args.max_rings,
        "ring_temp_scale": args.ring_temp_scale,
        "ring_len_scale": args.ring_len_scale,
        "graze_rate": args.graze_rate,
        "somatic_temp": args.somatic_temp or None,
        "somatic_temp_strength": args.somatic_temp_strength,
        "metastanley": args.metastanley or None,
        "metastanley_rate": args.metastanley_rate,
    }
    active_listening = {k: v for k, v in listening_args.items() if v is not None}
    if active_listening:
        joined = ", ".join(f"{k}={v}" for k, v in active_listening.items())
        lines.append(f"- listening args: `{joined}`")
    if args.graze:
        lines.append(f"- graze: `{', '.join(args.graze)}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- spoken: `{spoken}`")
    lines.append(f"- silent: `{silent}`")
    lines.append(f"- collapsed replies: `{collapsed}`")
    lines.append(f"- origin 5-gram echoes: `{origin_echo}`")
    lines.append(f"- avg spoken tokens: `{avg_tokens:.2f}`")
    lines.append(f"- avg glue ratio: `{avg_glue:.2f}`")
    lines.append(f"- repeated bigrams: `{repeated_bigrams}`")
    for key in ["vocab", "inputs", "spoken", "refused", "dreams", "shimmers", "fragments", "gravity", "sea", "scars", "pastures", "graze_vocab", "profiled", "scar_pressure", "speak_ratio", "coherence_floor", "max_rings", "temp_scale", "len_scale", "graze_rate", "somatic_temp", "somatic_temp_strength", "temp_factor", "metastanley", "metastanley_rate", "inner_ticks"]:
        if key in stats:
            lines.append(f"- {key}: `{stats[key]}`")
    lines.append("")
    lines.append("## Transcript Metrics")
    lines.append("")
    lines.append("| # | prompt | reply | tokens | dom | glue | tail_unique | repeated_bigrams | flags |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|")
    for i, (prompt, reply, row) in enumerate(zip(prompts, prompt_replies, rows), 1):
        flags: list[str] = []
        if row["silent"]:
            flags.append("silent")
        if row["collapsed"]:
            flags.append("collapsed")
        if row["origin_span"]:
            flags.append("origin-echo")
        safe_prompt = prompt.replace("|", "\\|")
        safe_reply = reply.replace("|", "\\|")
        if len(safe_reply) > 96:
            safe_reply = safe_reply[:93] + "..."
        lines.append(
            f"| {i} | {safe_prompt} | {safe_reply} | {row['tokens']} | "
            f"{float(row['dominant_ratio']):.2f} | {float(row['glue_ratio']):.2f} | "
            f"{row['tail_unique']} | {row['repeated_bigrams']} | {', '.join(flags) or '-'} |"
        )
    lines.append("")
    lines.append("## Raw Output")
    lines.append("")
    lines.append("```text")
    lines.append(output.strip())
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    text = Path(path).read_text(errors="ignore")
    prompts = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not prompts:
        raise SystemExit(f"no prompts in {path}")
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a behavioral eval against Stanley's real CLI.")
    parser.add_argument("--binary", default="./stanley", help="Stanley binary path")
    parser.add_argument("--origin", default="origin.txt", help="origin path")
    parser.add_argument("--no-origin", action="store_true", help="start without origin")
    parser.add_argument("--prompts", help="newline-delimited prompt file")
    parser.add_argument("--graze", action="append", default=[], help="attach GGUF pasture")
    parser.add_argument("--graze-profile", action="append", default=[], help="attach profile to previous pasture")
    parser.add_argument("--coherence-floor", type=float, help="override Stanley's baseline silence threshold")
    parser.add_argument("--max-rings", type=int, help="cap private overthinking rings")
    parser.add_argument("--ring-temp-scale", type=float, help="scale all private-ring temperatures")
    parser.add_argument("--ring-len-scale", type=float, help="scale all private-ring lengths")
    parser.add_argument("--graze-rate", type=float, help="probability of tail arbitration when grazing is hungry")
    parser.add_argument("--somatic-temp", action="store_true", help="let body tension modulate private-ring temperature")
    parser.add_argument("--somatic-temp-strength", type=float, help="body-to-temperature modulation strength")
    parser.add_argument("--metastanley", action="store_true", help="enable private MetaStanley phrase lane")
    parser.add_argument("--metastanley-rate", type=float, help="private MetaStanley phrase chance per spoken tick")
    parser.add_argument("--seed", type=int, help="deterministic Stanley RNG seed")
    parser.add_argument("--out", help="write markdown report to path")
    parser.add_argument("--fail-on-collapse", action="store_true", help="exit nonzero if any reply collapses")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    output = run_stanley(args, prompts)
    report = render_report(args, prompts, output)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report)
        print(f"wrote {path}")
    else:
        print(report)

    if args.fail_on_collapse and "- collapsed replies: `0`" not in report:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
