#!/usr/bin/env python3
"""
run.py — Interactive REPL for Stanley

Talk to a growing organism.
Watch it remember. Watch it learn. Watch it become.

Usage:
    python -m stanley.run [--data-dir PATH] [--origin PATH]

Commands in REPL:
    /stats      - Show organism statistics
    /memory     - Show memory state
    /shards     - List recent shards
    /save       - Save organism state
    /grow       - Trigger growth cycle
    /help       - Show this help
    /quit       - Exit
"""

from __future__ import annotations
import sys
import argparse
import time
from pathlib import Path

# Import Stanley components
try:
    from .organism import Stanley, StanleyConfig, load_or_create
except ImportError:
    # Running as script
    from organism import Stanley, StanleyConfig, load_or_create


def print_banner():
    """Print startup banner."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ███████╗████████╗ █████╗ ███╗   ██╗██╗     ███████╗██╗  ║
║   ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██║     ██╔════╝╚██╗ ║
║   ███████╗   ██║   ███████║██╔██╗ ██║██║     █████╗   ╚██╗║
║   ╚════██║   ██║   ██╔══██║██║╚██╗██║██║     ██╔══╝   ██╔╝║
║   ███████║   ██║   ██║  ██║██║ ╚████║███████╗███████╗██╔╝ ║
║   ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ║
║                                                           ║
║        Self Training Attention Non-Linear EntitY          ║
║                                                           ║
║   "I don't hold everything in my head at once.            ║
║    I remember what resonates."                            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
    """)


def print_help():
    """Print REPL help."""
    print("""
Commands:
    /stats      Show organism statistics
    /memory     Show memory state (surface/middle/deep/abyss)
    /shards     List recent shards
    /working    Show current working set
    /save       Save organism state
    /grow       Trigger growth/consolidation cycle
    /origin     Show origin text
    /ariannamethod  Show ariannamethod commands
    /help       Show this help
    /quit       Exit REPL

Ariannamethod commands (use in conversation):
    jump(delta=0.5, future_state='creative')
    predict(next_delta=0.8)
    time_travel(offset=-10)
    resonate(shard_id='abc123', boost=2.0)
    prophecy(vision='emergence', strength=0.7)
    drift(direction='curious', momentum=0.6)

Just type text to talk to Stanley.
Stanley will respond and maybe remember the interaction.
    """)


def format_stats(stats: dict) -> str:
    """Format statistics for display."""
    lines = []
    lines.append(f"Age: {stats['age_seconds']:.0f}s ({stats['age_seconds']/3600:.1f}h)")
    lines.append(f"Interactions: {stats['total_interactions']}")
    lines.append(f"Maturity: {stats['maturity']:.1%}")
    lines.append(f"Vocab size: {stats['vocab_size']}")

    mem = stats['memory']
    lines.append(f"\nMemory:")
    lines.append(f"  Surface: {mem['surface']} shards")
    lines.append(f"  Middle:  {mem['middle']} shards")
    lines.append(f"  Deep:    {mem['deep']} shards (macro-adapters)")
    lines.append(f"  Abyss:   {mem['abyss']} metanotes")
    lines.append(f"  Total:   {mem['total_mb']:.2f} MB")

    buf = stats['buffer']
    lines.append(f"\nQuantum Buffer:")
    lines.append(f"  Pending: {buf['pending']} shards")
    lines.append(f"  Trains:  {buf['total_trains']}")

    if stats.get('trainer'):
        tr = stats['trainer']
        lines.append(f"\nTrainer:")
        lines.append(f"  State:   {tr['state']}")
        lines.append(f"  Batches: {tr['total_trains']}")
        lines.append(f"  Items:   {tr['total_items']}")

    lines.append(f"\nWorking set: {stats['working_set_size']} shards")

    return "\n".join(lines)


def format_memory(stanley: Stanley) -> str:
    """Format memory state for display."""
    lines = []

    lines.append("=== SURFACE (active, ~MB) ===")
    for s in stanley.memory.surface[:10]:
        lines.append(f"  [{s.id[:8]}] res={s.resonance_score:.2f} act={s.activation_count}")

    lines.append(f"\n=== MIDDLE (accessible, {len(stanley.memory.middle)}) ===")
    for s in stanley.memory.middle[:5]:
        lines.append(f"  [{s.id[:8]}] res={s.resonance_score:.2f} act={s.activation_count}")
    if len(stanley.memory.middle) > 5:
        lines.append(f"  ... and {len(stanley.memory.middle) - 5} more")

    lines.append(f"\n=== DEEP (macro-adapters, {len(stanley.memory.deep)}) ===")
    for s in stanley.memory.deep[:3]:
        lines.append(f"  [{s.id[:8]}] res={s.resonance_score:.2f} act={s.activation_count}")

    lines.append(f"\n=== ABYSS (ghosts, {len(stanley.memory.abyss)}) ===")
    for n in stanley.memory.abyss[:3]:
        lines.append(f"  [meta_{n.original_id[:6]}] gate_nudge={n.gate_nudge:.3f}")

    return "\n".join(lines)


def run_repl(stanley: Stanley):
    """Run the interactive REPL."""
    print_help()
    print(f"\nStanley loaded: {stanley}")
    print("Type /help for commands, or just start talking.\n")

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
            cmd = user_input.lower().split()[0]

            if cmd == "/quit" or cmd == "/exit":
                print("Saving state...")
                stanley.save()
                print("Goodbye!")
                break

            elif cmd == "/help":
                print_help()

            elif cmd == "/stats":
                print(format_stats(stanley.stats()))

            elif cmd == "/memory":
                print(format_memory(stanley))

            elif cmd == "/shards":
                print("Recent shards:")
                for s in stanley.memory.surface[-10:]:
                    print(f"  [{s.id[:8]}] {s.depth} res={s.resonance_score:.2f}")

            elif cmd == "/working":
                print(f"Working set ({len(stanley.engine.working_set)} shards):")
                for s in stanley.engine.working_set:
                    print(f"  [{s.id[:8]}] res={s.resonance_score:.2f}")

            elif cmd == "/save":
                stanley.save()
                print("Saved!")

            elif cmd == "/grow":
                print("Running growth cycle...")
                stanley.grow()
                print("Done!")

            elif cmd == "/origin":
                print("Origin text:")
                print(stanley.origin_text)

            elif cmd == "/ariannamethod":
                if stanley.ariannamethod:
                    print("\nAriannamethod Commands:")
                    print("=" * 60)
                    commands = stanley.ariannamethod.available_commands()
                    for cmd_syntax, description in commands.items():
                        print(f"\n  {cmd_syntax}")
                        print(f"    → {description}")
                    print("\n" + "=" * 60)
                    
                    # Show stats if any commands were used
                    am_stats = stanley.ariannamethod.stats()
                    if am_stats["total_commands"] > 0:
                        print(f"\nUsage Statistics:")
                        print(f"  Total commands: {am_stats['total_commands']}")
                        print(f"  Command counts: {am_stats['command_counts']}")
                        print(f"  Last command: {am_stats['last_command']}")
                else:
                    print("Ariannamethod not enabled.")

            else:
                print(f"Unknown command: {cmd}")
                print("Type /help for available commands")

            continue

        # Regular conversation
        print("thinking...")

        # Get response
        start = time.time()
        response, stats = stanley.think(user_input, length=100)
        think_time = time.time() - start

        print(f"\n[stanley]: {response}")

        # Extract stats with fallbacks
        pulse_stats = stats.get('pulse', {})
        entropy = pulse_stats.get('entropy', 0.5)
        arousal = pulse_stats.get('arousal', 0.5)
        temp = stats.get('temperature', 0.8)
        method = stats.get('method', 'unknown')

        # Show body sense if available
        body = stats.get('body_sense', {})
        boredom = body.get('boredom', 0)
        overwhelm = body.get('overwhelm', 0)

        print(f"    (ent={entropy:.2f}, aro={arousal:.2f}, temp={temp:.2f}, "
              f"method={method}, time={think_time:.2f}s)")
        if body:
            print(f"    (boredom={boredom:.2f}, overwhelm={overwhelm:.2f})")
        
        # Show ariannamethod info if commands were executed
        if 'ariannamethod' in stats:
            am_info = stats['ariannamethod']
            print(f"    [ariannamethod: {len(am_info['commands'])} commands executed]")
            for cmd, result in zip(am_info['commands'], am_info['results']):
                if result.get('success'):
                    print(f"      ✓ {cmd}")
                else:
                    print(f"      ✗ {cmd} - {result.get('error', 'unknown error')}")

        # Process experience (maybe remember)
        full_interaction = f"user: {user_input}\nstanley: {response}"
        shard = stanley.experience(full_interaction)

        if shard:
            print(f"    [+shard {shard.id[:8]}]")

        # Periodic growth
        if stanley.total_interactions % 10 == 0:
            stanley.grow()

        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stanley — Self Training Attention Non-Linear EntitY"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./stanley_data",
        help="Directory for saving/loading state",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default=None,
        help="Path to origin.txt file",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip startup banner",
    )

    args = parser.parse_args()

    if not args.no_banner:
        print_banner()

    # Load origin text if specified
    origin_text = None
    if args.origin:
        origin_path = Path(args.origin)
        if origin_path.exists():
            origin_text = origin_path.read_text()
            print(f"Loaded origin from {args.origin}")
        else:
            print(f"Warning: origin file not found: {args.origin}")

    # Load or create Stanley
    print(f"Data directory: {args.data_dir}")
    stanley = load_or_create(args.data_dir, origin_text=origin_text)

    # Run REPL
    try:
        run_repl(stanley)
    except Exception as e:
        print(f"\nError: {e}")
        print("Attempting to save state...")
        try:
            stanley.save()
            print("State saved.")
        except:
            print("Could not save state.")
        raise


if __name__ == "__main__":
    main()
