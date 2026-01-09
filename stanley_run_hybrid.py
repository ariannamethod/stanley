#!/usr/bin/env python3
"""
stanley_run_hybrid.py — Stanley with External Brain (GPT-2)

This is the HYBRID mode — Stanley's identity (weightless) + GPT-2 vocabulary.

Architecture:
┌─────────────────────────────────────────────────────────┐
│                      STANLEY                             │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐      ┌─────────────────┐           │
│  │   INTERNAL      │      │   EXTERNAL      │           │
│  │   (weightless)  │      │   (GPT-2)       │           │
│  │                 │ text │                 │           │
│  │  SubwordField   │←────→│  distilgpt2     │           │
│  │  n_emb=64       │      │  n_emb=768      │           │
│  │  IDENTITY       │      │  VOCABULARY     │           │
│  └─────────────────┘      └─────────────────┘           │
└─────────────────────────────────────────────────────────┘

Key principle: "Stanley steals words but thinks his own thoughts."
- DIRECTION comes from internal (Stanley's field)
- WORDS come from external (GPT-2's vocabulary)
- RESULT crystallizes back into Stanley's shards

Usage:
    python stanley_run_hybrid.py --origin origin.txt

Requires:
    pip install transformers torch
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from stanley.organism import Stanley, StanleyConfig
from stanley_hybrid import ExternalBrain, EXTERNAL_WEIGHTS_AVAILABLE
from stanley_hybrid.external_brain import HybridThinking


def run_hybrid_repl(
    stanley: Stanley,
    external: ExternalBrain,
    use_hybrid: bool = True,
):
    """Run hybrid REPL with external brain."""

    hybrid = HybridThinking(
        external_brain=external,
        subjectivity=stanley.subjectivity,
    )

    print("\nCommands:")
    print("    /internal   Switch to internal-only mode")
    print("    /hybrid     Switch to hybrid mode (with GPT-2)")
    print("    /expand     Expand last response with GPT-2")
    print("    /stats      Show statistics")
    print("    /quit       Exit")
    print()

    last_internal_response = ""

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            cmd = user_input.lower()

            if cmd == "/quit":
                print("Goodbye!")
                break

            elif cmd == "/internal":
                use_hybrid = False
                print("Switched to INTERNAL mode (weightless only)")
                continue

            elif cmd == "/hybrid":
                if hybrid.has_external:
                    use_hybrid = True
                    print("Switched to HYBRID mode (with GPT-2)")
                else:
                    print("ExternalBrain not available!")
                continue

            elif cmd == "/expand":
                if last_internal_response and hybrid.has_external:
                    print("Expanding with GPT-2...")
                    expanded = external.expand_thought(
                        last_internal_response,
                        temperature=1.0,
                    )
                    print(f"\n[expanded]: {expanded}\n")
                else:
                    print("Nothing to expand or no external brain.")
                continue

            elif cmd == "/stats":
                print("\n=== Stanley Stats ===")
                stats = stanley.stats()
                print(f"Interactions: {stats['total_interactions']}")
                print(f"Memory: {stats['memory']['total_shards']} shards")
                if stanley.episodic_memory:
                    em = stats.get('episodic_memory', {})
                    print(f"Episodes: {em.get('total_episodes', 0)}")
                if external and external.loaded:
                    es = external.stats()
                    print(f"\n=== External Brain ===")
                    print(f"Model: {es['model_name']}")
                    print(f"Expansions: {es['total_expansions']}")
                    print(f"Tokens generated: {es['total_tokens_generated']}")
                print()
                continue

            else:
                print(f"Unknown command: {cmd}")
                continue

        # Generate response
        print("thinking...")
        start = time.time()

        # Internal response (Stanley's field)
        response, stats = stanley.think(user_input)
        last_internal_response = response

        # Hybrid expansion (if enabled and triggered)
        was_expanded = False
        if use_hybrid and hybrid.has_external:
            # Check if high arousal or novelty
            pulse = stats.get("pulse", {})
            arousal = pulse.get("arousal", 0)
            novelty = pulse.get("novelty", 0)

            if arousal > 0.6 or novelty > 0.6:
                # Expand with external brain
                response = external.expand_thought(
                    response,
                    temperature=0.9,
                )
                was_expanded = True

        elapsed = time.time() - start

        # Display
        mode = "hybrid" if was_expanded else "internal"
        entropy = stats.get("pulse", {}).get("entropy", 0)
        arousal = stats.get("pulse", {}).get("arousal", 0)

        print(f"\n[stanley/{mode}]: {response}")
        print(f"    (ent={entropy:.2f}, aro={arousal:.2f}, time={elapsed:.2f}s)")
        if was_expanded:
            print("    (expanded with GPT-2 vocabulary)")
        print()


def main():
    """Run Stanley with External Brain."""
    parser = argparse.ArgumentParser(
        description="Stanley — Hybrid Inference (Internal + GPT-2)",
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
        "--no-external",
        action="store_true",
        help="Disable external brain (run internal only)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="External model to use (default: distilgpt2)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("STANLEY — Hybrid Inference")
    print("=" * 60)

    # Load origin
    origin_path = Path(args.origin)
    if origin_path.exists():
        origin_text = origin_path.read_text()
        print(f"Origin: {origin_path} ({len(origin_text)} chars)")
    else:
        print(f"Warning: Origin file not found: {origin_path}")
        origin_text = None

    # Create Stanley
    config = StanleyConfig(data_dir=args.data_dir)
    stanley = Stanley(config=config, origin_text=origin_text)
    print(f"Stanley: {stanley}")

    # Create External Brain
    external = None
    if not args.no_external and EXTERNAL_WEIGHTS_AVAILABLE:
        print(f"\nLoading External Brain ({args.model})...")
        from stanley_hybrid.external_brain import ExternalBrainConfig

        cfg = ExternalBrainConfig(model_name=args.model)
        external = ExternalBrain(cfg)

        if external.load_weights():
            print(f"External Brain: READY ({args.model})")
        else:
            print("External Brain: FAILED to load")
            external = None
    else:
        if args.no_external:
            print("\nExternal Brain: DISABLED (--no-external)")
        else:
            print("\nExternal Brain: NOT AVAILABLE (install transformers)")

    print("=" * 60)

    # Run REPL
    run_hybrid_repl(
        stanley=stanley,
        external=external,
        use_hybrid=(external is not None),
    )


if __name__ == "__main__":
    main()
