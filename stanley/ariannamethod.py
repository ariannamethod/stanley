"""
ariannamethod.py — Mini Programming Language for Weightless Inference

A temporal mini-language for influencing prophecy and dynamic personality deltas
during pure weightless inference in Stanley.

Inspired by temporal concepts, this language allows:
- Jumping between personality states (jump)
- Predicting future deltas (predict)
- Time-traveling through shard history (time_travel)
- Modifying shard resonance dynamically
- Influencing memory loading behavior
- Creating prophecy-like generation patterns

Commands are executed during inference to modify Stanley's behavior in real-time.
This is pure weightless magic — no neural weights required.

Example commands:
    jump(delta=0.5, future_state='creative')
    predict(next_delta=0.8)
    time_travel(offset=-10)
    resonate(shard_id='abc123', boost=2.0)
    prophecy(vision='emergence', strength=0.7)
    drift(direction='curious', momentum=0.6)
"""

from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class CommandType(Enum):
    """Types of ariannamethod commands."""
    JUMP = "jump"
    PREDICT = "predict"
    TIME_TRAVEL = "time_travel"
    RESONATE = "resonate"
    PROPHECY = "prophecy"
    DRIFT = "drift"
    RECALL = "recall"
    AMPLIFY = "amplify"
    DAMPEN = "dampen"
    SHIFT = "shift"


@dataclass
class Command:
    """A parsed ariannamethod command."""
    command_type: CommandType
    args: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    
    def __repr__(self):
        args_str = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.command_type.value}({args_str})"


@dataclass
class ExecutionContext:
    """Context available during command execution."""
    # Current state
    current_temperature: float = 0.8
    current_pulse: Optional[Any] = None  # Pulse from subjectivity
    
    # Memory access
    memory_sea: Optional[Any] = None
    working_set: List[Any] = field(default_factory=list)
    
    # Generation state
    generated_tokens: List[str] = field(default_factory=list)
    generation_length: int = 100
    
    # Results from command execution
    delta_modifications: Dict[str, float] = field(default_factory=dict)
    resonance_boosts: Dict[str, float] = field(default_factory=dict)
    prophecy_visions: List[str] = field(default_factory=list)
    time_offset: int = 0
    
    def reset_modifications(self):
        """Reset all modifications for a new generation."""
        self.delta_modifications.clear()
        self.resonance_boosts.clear()
        self.prophecy_visions.clear()
        self.time_offset = 0


class CommandExecutor:
    """
    Executes ariannamethod commands.
    
    Each command type has a handler that modifies the execution context.
    This influences Stanley's behavior during inference.
    """
    
    def __init__(self):
        self.handlers: Dict[CommandType, Callable] = {
            CommandType.JUMP: self._execute_jump,
            CommandType.PREDICT: self._execute_predict,
            CommandType.TIME_TRAVEL: self._execute_time_travel,
            CommandType.RESONATE: self._execute_resonate,
            CommandType.PROPHECY: self._execute_prophecy,
            CommandType.DRIFT: self._execute_drift,
            CommandType.RECALL: self._execute_recall,
            CommandType.AMPLIFY: self._execute_amplify,
            CommandType.DAMPEN: self._execute_dampen,
            CommandType.SHIFT: self._execute_shift,
        }
    
    def execute(self, command: Command, context: ExecutionContext) -> Dict[str, Any]:
        """
        Execute a command and return the results.
        
        Returns:
            Dict with execution results and any modifications to apply.
        """
        handler = self.handlers.get(command.command_type)
        if not handler:
            logger.warning(f"No handler for command type: {command.command_type}")
            return {"success": False, "error": "unknown_command"}
        
        try:
            result = handler(command, context)
            result["success"] = True
            result["command"] = str(command)
            return result
        except Exception as e:
            logger.error(f"Error executing {command}: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_jump(self, command: Command, context: ExecutionContext) -> Dict[str, Any]:
        """
        Jump to a different personality state.
        
        Args:
            delta: Magnitude of the jump (0.0 to 1.0)
            future_state: Target personality state (e.g., 'creative', 'analytical')
        """
        delta = float(command.args.get("delta", 0.5))
        future_state = command.args.get("future_state", "neutral")
        
        # Modify temperature based on future state
        state_temps = {
            "creative": 1.2,
            "analytical": 0.6,
            "playful": 1.0,
            "focused": 0.7,
            "dreamy": 1.3,
            "neutral": 0.8,
        }
        
        target_temp = state_temps.get(future_state, 0.8)
        # Interpolate based on delta
        new_temp = context.current_temperature * (1 - delta) + target_temp * delta
        context.current_temperature = max(0.1, min(2.0, new_temp))
        
        context.delta_modifications[future_state] = delta
        
        return {
            "jumped_to": future_state,
            "delta": delta,
            "new_temperature": context.current_temperature,
        }
    
    def _execute_predict(self, command: Command, context: ExecutionContext) -> Dict[str, Any]:
        """
        Predict and influence the next personality delta.
        
        Args:
            next_delta: Predicted delta magnitude (0.0 to 1.0)
        """
        next_delta = float(command.args.get("next_delta", 0.5))
        
        # Influence temperature towards prediction
        context.current_temperature *= (1.0 + next_delta * 0.5)
        context.current_temperature = max(0.1, min(2.0, context.current_temperature))
        
        return {
            "predicted_delta": next_delta,
            "adjusted_temperature": context.current_temperature,
        }
    
    def _execute_time_travel(self, command: Command, context: ExecutionContext) -> Dict[str, Any]:
        """
        Travel through shard history.
        
        Args:
            offset: Number of shards to go back (negative) or forward (positive)
        """
        offset = int(command.args.get("offset", 0))
        context.time_offset = offset
        
        # Modify working set based on offset
        # Negative offset = look at older shards
        # Positive offset = look at newer shards (if any)
        
        return {
            "time_offset": offset,
            "direction": "past" if offset < 0 else "future",
        }
    
    def _execute_resonate(self, command: Command, context: ExecutionContext) -> Dict[str, Any]:
        """
        Modify resonance of a specific shard.
        
        Args:
            shard_id: ID of the shard to affect (can be partial)
            boost: Resonance multiplier (default 1.0)
        """
        shard_id = command.args.get("shard_id", "")
        boost = float(command.args.get("boost", 1.0))
        
        context.resonance_boosts[shard_id] = boost
        
        return {
            "shard_id": shard_id,
            "boost": boost,
        }
    
    def _execute_prophecy(self, command: Command, context: ExecutionContext) -> Dict[str, Any]:
        """
        Create a prophecy-like vision that influences generation.
        
        Args:
            vision: The prophetic vision (text that should emerge)
            strength: How strongly to pull towards the vision (0.0 to 1.0)
        """
        vision = command.args.get("vision", "")
        strength = float(command.args.get("strength", 0.5))
        
        # Add vision to context for generation to pull towards
        context.prophecy_visions.append(vision)
        
        # Adjust temperature based on prophecy strength
        # Higher strength = lower temperature (more deterministic towards vision)
        temp_adjustment = 1.0 - (strength * 0.3)
        context.current_temperature *= temp_adjustment
        
        return {
            "vision": vision,
            "strength": strength,
            "visions_active": len(context.prophecy_visions),
        }
    
    def _execute_drift(self, command: Command, context: ExecutionContext) -> Dict[str, Any]:
        """
        Drift semantically in a direction.
        
        Args:
            direction: Semantic direction to drift towards
            momentum: How strongly to drift (0.0 to 1.0)
        """
        direction = command.args.get("direction", "neutral")
        momentum = float(command.args.get("momentum", 0.5))
        
        context.delta_modifications[f"drift_{direction}"] = momentum
        
        # Adjust temperature for drift
        context.current_temperature *= (1.0 + momentum * 0.2)
        context.current_temperature = max(0.1, min(2.0, context.current_temperature))
        
        return {
            "direction": direction,
            "momentum": momentum,
        }
    
    def _execute_recall(self, command: Command, context: ExecutionContext) -> Dict[str, Any]:
        """
        Force recall of specific memory patterns.
        
        Args:
            pattern: Pattern to recall
            strength: Recall strength (0.0 to 1.0)
        """
        pattern = command.args.get("pattern", "")
        strength = float(command.args.get("strength", 0.5))
        
        return {
            "pattern": pattern,
            "strength": strength,
        }
    
    def _execute_amplify(self, command: Command, context: ExecutionContext) -> Dict[str, Any]:
        """
        Amplify current signal strength.
        
        Args:
            factor: Amplification factor (default 1.5)
        """
        factor = float(command.args.get("factor", 1.5))
        context.current_temperature *= factor
        context.current_temperature = max(0.1, min(2.0, context.current_temperature))
        
        return {
            "factor": factor,
            "new_temperature": context.current_temperature,
        }
    
    def _execute_dampen(self, command: Command, context: ExecutionContext) -> Dict[str, Any]:
        """
        Dampen current signal strength.
        
        Args:
            factor: Dampening factor (default 0.7)
        """
        factor = float(command.args.get("factor", 0.7))
        context.current_temperature *= factor
        context.current_temperature = max(0.1, min(2.0, context.current_temperature))
        
        return {
            "factor": factor,
            "new_temperature": context.current_temperature,
        }
    
    def _execute_shift(self, command: Command, context: ExecutionContext) -> Dict[str, Any]:
        """
        Shift the entire generation context.
        
        Args:
            dimension: Which dimension to shift (e.g., 'entropy', 'novelty')
            amount: How much to shift (can be negative)
        """
        dimension = command.args.get("dimension", "entropy")
        amount = float(command.args.get("amount", 0.1))
        
        context.delta_modifications[f"shift_{dimension}"] = amount
        
        return {
            "dimension": dimension,
            "amount": amount,
        }


class AriannaMethods:
    """
    Parser and interpreter for the ariannamethod mini-language.
    
    Parses command strings and executes them in the context of
    Stanley's inference pipeline.
    """
    
    def __init__(self):
        self.executor = CommandExecutor()
        self.command_history: List[Command] = []
        
        # Regex patterns for parsing commands
        # Note: Parameters must be in the specified order. A more flexible parser
        # could be implemented to allow parameter reordering, but this keeps
        # implementation simple for the initial version.
        #
        # Fixed numeric pattern to only match valid decimal numbers
        num_pattern = r'([0-9]+(?:\.[0-9]+)?)'  # Valid: 1, 1.5, 0.5 Invalid: 1.2.3, ..
        signed_num_pattern = r'(-?[0-9]+(?:\.[0-9]+)?)'  # Allows negative numbers
        int_pattern = r'(-?[0-9]+)'  # Integer only
        string_pattern = r'["\']([^"\']+)["\']'  # String in quotes
        
        self.patterns = {
            CommandType.JUMP: rf'jump\s*\(\s*delta\s*=\s*{num_pattern}\s*,\s*future_state\s*=\s*{string_pattern}\s*\)',
            CommandType.PREDICT: rf'predict\s*\(\s*next_delta\s*=\s*{num_pattern}\s*\)',
            CommandType.TIME_TRAVEL: rf'time_travel\s*\(\s*offset\s*=\s*{int_pattern}\s*\)',
            CommandType.RESONATE: rf'resonate\s*\(\s*shard_id\s*=\s*{string_pattern}\s*,\s*boost\s*=\s*{num_pattern}\s*\)',
            CommandType.PROPHECY: rf'prophecy\s*\(\s*vision\s*=\s*{string_pattern}\s*,\s*strength\s*=\s*{num_pattern}\s*\)',
            CommandType.DRIFT: rf'drift\s*\(\s*direction\s*=\s*{string_pattern}\s*,\s*momentum\s*=\s*{num_pattern}\s*\)',
            CommandType.RECALL: rf'recall\s*\(\s*pattern\s*=\s*{string_pattern}\s*,\s*strength\s*=\s*{num_pattern}\s*\)',
            CommandType.AMPLIFY: rf'amplify\s*\(\s*factor\s*=\s*{num_pattern}\s*\)',
            CommandType.DAMPEN: rf'dampen\s*\(\s*factor\s*=\s*{num_pattern}\s*\)',
            CommandType.SHIFT: rf'shift\s*\(\s*dimension\s*=\s*{string_pattern}\s*,\s*amount\s*=\s*{signed_num_pattern}\s*\)',
        }
    
    def parse(self, text: str) -> List[Command]:
        """
        Parse ariannamethod commands from text.
        
        Commands can appear anywhere in the text and will be extracted
        and parsed.
        
        Returns:
            List of parsed Command objects.
        """
        commands = []
        
        for cmd_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                cmd = self._parse_match(cmd_type, match)
                if cmd:
                    commands.append(cmd)
        
        return commands
    
    def _parse_match(self, cmd_type: CommandType, match: re.Match) -> Optional[Command]:
        """Parse a regex match into a Command object."""
        try:
            args = {}
            
            if cmd_type == CommandType.JUMP:
                args = {
                    "delta": float(match.group(1)),
                    "future_state": match.group(2),
                }
            elif cmd_type == CommandType.PREDICT:
                args = {"next_delta": float(match.group(1))}
            elif cmd_type == CommandType.TIME_TRAVEL:
                args = {"offset": int(match.group(1))}
            elif cmd_type == CommandType.RESONATE:
                args = {
                    "shard_id": match.group(1),
                    "boost": float(match.group(2)),
                }
            elif cmd_type == CommandType.PROPHECY:
                args = {
                    "vision": match.group(1),
                    "strength": float(match.group(2)),
                }
            elif cmd_type == CommandType.DRIFT:
                args = {
                    "direction": match.group(1),
                    "momentum": float(match.group(2)),
                }
            elif cmd_type == CommandType.RECALL:
                args = {
                    "pattern": match.group(1),
                    "strength": float(match.group(2)),
                }
            elif cmd_type == CommandType.AMPLIFY:
                args = {"factor": float(match.group(1))}
            elif cmd_type == CommandType.DAMPEN:
                args = {"factor": float(match.group(1))}
            elif cmd_type == CommandType.SHIFT:
                args = {
                    "dimension": match.group(1),
                    "amount": float(match.group(2)),
                }
            
            return Command(
                command_type=cmd_type,
                args=args,
                raw_text=match.group(0),
            )
        except (ValueError, IndexError, AttributeError) as e:
            # ValueError: float() conversion failed
            # IndexError: match.group() index out of range
            # AttributeError: unexpected match object state
            logger.warning(f"Failed to parse {cmd_type}: {e}")
            return None
        except Exception as e:
            # Catch any other unexpected errors during parsing
            logger.error(f"Unexpected error parsing {cmd_type}: {e}")
            return None
    
    def execute(self, commands: List[Command], context: ExecutionContext) -> List[Dict[str, Any]]:
        """
        Execute a list of commands in sequence.
        
        Returns:
            List of execution results.
        """
        results = []
        for cmd in commands:
            result = self.executor.execute(cmd, context)
            results.append(result)
            self.command_history.append(cmd)
        
        return results
    
    def parse_and_execute(self, text: str, context: ExecutionContext) -> Tuple[List[Command], List[Dict[str, Any]]]:
        """
        Parse commands from text and execute them.
        
        Returns:
            (commands, results) tuple.
        """
        commands = self.parse(text)
        results = self.execute(commands, context)
        return commands, results
    
    def available_commands(self) -> Dict[str, str]:
        """Return dictionary of available commands and their descriptions."""
        return {
            "jump(delta=0.5, future_state='creative')": "Jump to a different personality state",
            "predict(next_delta=0.8)": "Predict and influence the next personality delta",
            "time_travel(offset=-10)": "Travel through shard history",
            "resonate(shard_id='abc123', boost=2.0)": "Modify resonance of a specific shard",
            "prophecy(vision='emergence', strength=0.7)": "Create a prophecy-like vision",
            "drift(direction='curious', momentum=0.6)": "Drift semantically in a direction",
            "recall(pattern='memory', strength=0.8)": "Force recall of specific memory patterns",
            "amplify(factor=1.5)": "Amplify current signal strength",
            "dampen(factor=0.7)": "Dampen current signal strength",
            "shift(dimension='entropy', amount=0.1)": "Shift the entire generation context",
        }
    
    def stats(self) -> Dict[str, Any]:
        """Return statistics about command usage."""
        cmd_counts = {}
        for cmd in self.command_history:
            cmd_type = cmd.command_type.value
            cmd_counts[cmd_type] = cmd_counts.get(cmd_type, 0) + 1
        
        return {
            "total_commands": len(self.command_history),
            "command_counts": cmd_counts,
            "last_command": str(self.command_history[-1]) if self.command_history else None,
        }
