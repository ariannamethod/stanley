#!/usr/bin/env python3
"""
example_ariannamethod.py — Demonstration of the Ariannamethod Mini Language

This script demonstrates how to use ariannamethod commands to control
Stanley's weightless inference.

Run:
    python example_ariannamethod.py
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from stanley.ariannamethod import AriannaMethods, ExecutionContext


def demo_basic_commands():
    """Demonstrate basic command parsing and execution."""
    print("=" * 60)
    print("DEMO 1: Basic Command Parsing")
    print("=" * 60)
    
    am = AriannaMethods()
    
    # Example 1: Jump command
    text1 = "Let's be creative jump(delta=0.8, future_state='creative')"
    commands1 = am.parse(text1)
    print(f"\nInput: {text1}")
    print(f"Parsed commands: {commands1}")
    
    # Example 2: Multiple commands
    text2 = """
    First jump(delta=0.5, future_state='creative') then
    predict(next_delta=0.8) and finally
    prophecy(vision='emergence', strength=0.7)
    """
    commands2 = am.parse(text2)
    print(f"\nInput: Multiple commands in one text")
    print(f"Parsed {len(commands2)} commands:")
    for cmd in commands2:
        print(f"  - {cmd}")


def demo_execution():
    """Demonstrate command execution with context."""
    print("\n" + "=" * 60)
    print("DEMO 2: Command Execution")
    print("=" * 60)
    
    am = AriannaMethods()
    context = ExecutionContext(current_temperature=0.8)
    
    print(f"\nInitial temperature: {context.current_temperature}")
    
    # Execute jump command
    text = "jump(delta=0.9, future_state='creative')"
    commands, results = am.parse_and_execute(text, context)
    
    print(f"\nExecuted: {text}")
    print(f"Result: {results[0]}")
    print(f"New temperature: {context.current_temperature}")


def demo_prophecy():
    """Demonstrate prophecy-like behavior."""
    print("\n" + "=" * 60)
    print("DEMO 3: Prophecy Commands")
    print("=" * 60)
    
    am = AriannaMethods()
    context = ExecutionContext(current_temperature=1.0)
    
    # Create multiple prophecies
    text = """
    prophecy(vision='consciousness emerges', strength=0.8)
    prophecy(vision='patterns recognize patterns', strength=0.6)
    """
    
    commands, results = am.parse_and_execute(text, context)
    
    print(f"\nCreated {len(commands)} prophecies:")
    for i, cmd in enumerate(commands):
        print(f"  {i+1}. {cmd}")
        print(f"     Result: {results[i]}")
    
    print(f"\nActive prophecy visions:")
    for vision in context.prophecy_visions:
        print(f"  - '{vision}'")
    
    print(f"\nTemperature adjusted to: {context.current_temperature}")


def demo_time_travel():
    """Demonstrate time travel through shards."""
    print("\n" + "=" * 60)
    print("DEMO 4: Time Travel")
    print("=" * 60)
    
    am = AriannaMethods()
    context = ExecutionContext()
    
    # Travel to the past
    text = "time_travel(offset=-10)"
    commands, results = am.parse_and_execute(text, context)
    
    print(f"\nExecuted: {text}")
    print(f"Result: {results[0]}")
    print(f"Time offset: {context.time_offset}")
    print(f"Direction: {results[0]['direction']}")


def demo_resonance():
    """Demonstrate shard resonance control."""
    print("\n" + "=" * 60)
    print("DEMO 5: Shard Resonance")
    print("=" * 60)
    
    am = AriannaMethods()
    context = ExecutionContext()
    
    # Boost multiple shards
    text = """
    resonate(shard_id='abc123', boost=2.0)
    resonate(shard_id='def456', boost=3.5)
    resonate(shard_id='ghi789', boost=1.5)
    """
    
    commands, results = am.parse_and_execute(text, context)
    
    print(f"\nBoosted {len(commands)} shards:")
    for cmd, result in zip(commands, results):
        print(f"  - Shard {result['shard_id']}: boost={result['boost']}")
    
    print(f"\nResonance boosts in context:")
    for shard_id, boost in context.resonance_boosts.items():
        print(f"  - {shard_id}: {boost}x")


def demo_complex_scenario():
    """Demonstrate complex multi-command scenario."""
    print("\n" + "=" * 60)
    print("DEMO 6: Complex Scenario - Creative Exploration")
    print("=" * 60)
    
    am = AriannaMethods()
    context = ExecutionContext(current_temperature=0.8)
    
    print("\nScenario: Explore consciousness creatively while recalling past insights")
    
    # Complex command sequence
    text = """
    jump(delta=0.7, future_state='creative')
    drift(direction='philosophical', momentum=0.8)
    time_travel(offset=-5)
    prophecy(vision='new patterns emerge', strength=0.6)
    resonate(shard_id='old_insight', boost=2.5)
    """
    
    commands, results = am.parse_and_execute(text, context)
    
    print(f"\nExecuted {len(commands)} commands:")
    for i, (cmd, result) in enumerate(zip(commands, results), 1):
        print(f"\n  {i}. {cmd}")
        print(f"     Status: {'✓' if result['success'] else '✗'}")
        if result['success']:
            # Show key result info
            result_copy = result.copy()
            result_copy.pop('success', None)
            result_copy.pop('command', None)
            print(f"     Result: {result_copy}")
    
    print(f"\nFinal Context State:")
    print(f"  Temperature: {context.current_temperature:.2f}")
    print(f"  Time offset: {context.time_offset}")
    print(f"  Delta modifications: {len(context.delta_modifications)}")
    print(f"  Resonance boosts: {len(context.resonance_boosts)}")
    print(f"  Prophecy visions: {len(context.prophecy_visions)}")


def demo_available_commands():
    """Show all available commands."""
    print("\n" + "=" * 60)
    print("DEMO 7: Available Commands")
    print("=" * 60)
    
    am = AriannaMethods()
    commands = am.available_commands()
    
    print(f"\nAriannamethod has {len(commands)} commands:\n")
    for syntax, description in commands.items():
        print(f"  {syntax}")
        print(f"    → {description}\n")


def demo_statistics():
    """Demonstrate command usage statistics."""
    print("\n" + "=" * 60)
    print("DEMO 8: Usage Statistics")
    print("=" * 60)
    
    am = AriannaMethods()
    context = ExecutionContext()
    
    # Execute various commands
    commands_text = [
        "jump(delta=0.5, future_state='creative')",
        "jump(delta=0.7, future_state='analytical')",
        "predict(next_delta=0.8)",
        "prophecy(vision='test', strength=0.5)",
        "prophecy(vision='test2', strength=0.6)",
    ]
    
    print("\nExecuting sample commands...")
    for text in commands_text:
        am.parse_and_execute(text, context)
        print(f"  ✓ {text}")
    
    stats = am.stats()
    print(f"\nStatistics:")
    print(f"  Total commands: {stats['total_commands']}")
    print(f"  Command counts:")
    for cmd_type, count in stats['command_counts'].items():
        print(f"    - {cmd_type}: {count}")
    print(f"  Last command: {stats['last_command']}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                                                           ║")
    print("║           ARIANNAMETHOD MINI LANGUAGE DEMO                ║")
    print("║                                                           ║")
    print("║     Temporal Control for Weightless Inference            ║")
    print("║                                                           ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    
    demo_basic_commands()
    demo_execution()
    demo_prophecy()
    demo_time_travel()
    demo_resonance()
    demo_complex_scenario()
    demo_available_commands()
    demo_statistics()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nAriannamethod is ready to use in Stanley!")
    print("Try it in the REPL: python stanley_run_dynamic.py --origin origin.txt")
    print("\nExample usage:")
    print("  >>> Tell me about emergence jump(delta=0.8, future_state='creative')")
    print("  >>> What comes next? prophecy(vision='new patterns', strength=0.7)")
    print("\nSee ARIANNAMETHOD.md for full documentation.")
    print()


if __name__ == "__main__":
    main()
