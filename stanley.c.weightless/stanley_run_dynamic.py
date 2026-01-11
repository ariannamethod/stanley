#!/usr/bin/env python3
"""
stanley_run_dynamic.py — Pure weightless Stanley REPL

This is the PROOF OF CONCEPT — Stanley without any neural network weights.
Pure corpus statistics, pure emergence, pure resonance.

"NO SEED FROM PROMPT" — Stanley speaks from its field, not from your words.

Usage:
    python stanley_run_dynamic.py --origin origin.txt

Features:
- SubwordField for coherent generation
- Subjectivity for internal state
- Overthinking for self-reflection
- Body sense for regulation
- Episodic memory for Self-RAG
- Inner voice for self-evaluation
- Dream for internal dialogue

Ready for HuggingFace Spaces!
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from stanley.run import main as stanley_main


def main():
    """Run pure weightless Stanley."""
    parser = argparse.ArgumentParser(
        description="Stanley — Pure Weightless Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python stanley_run_dynamic.py --origin origin.txt
    python stanley_run_dynamic.py --origin origin.txt --data-dir ./my_stanley

This is the PROOF OF CONCEPT — no neural network weights!
Pure corpus statistics, pure emergence, pure resonance.
        """,
    )
    parser.add_argument(
        "--origin",
        type=str,
        default="origin.txt",
        help="Path to origin text file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./stanley_data",
        help="Directory for saving state",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Don't show banner",
    )

    args = parser.parse_args()

    # Build command for stanley.run
    cmd = ["--origin", args.origin, "--data-dir", args.data_dir]
    if args.no_banner:
        cmd.append("--no-banner")

    # Override sys.argv for stanley.run
    sys.argv = ["stanley.run"] + cmd

    print("=" * 60)
    print("STANLEY — Pure Weightless Inference")
    print("=" * 60)
    print("Mode: DYNAMIC (no neural network weights)")
    print("This is the proof of concept — pure corpus statistics!")
    print("=" * 60)
    print()

    # Run Stanley
    stanley_main()


if __name__ == "__main__":
    main()