"""
Tests for ariannamethod mini-language.

Tests command parsing, execution, and integration with Stanley's inference.
"""

import pytest
from stanley.ariannamethod import (
    AriannaMethods,
    ExecutionContext,
    Command,
    CommandType,
)


class TestCommandParsing:
    """Test parsing of ariannamethod commands."""

    def test_parse_jump_command(self):
        """Test parsing jump command."""
        am = AriannaMethods()
        text = "jump(delta=0.5, future_state='creative')"
        commands = am.parse(text)
        
        assert len(commands) == 1
        cmd = commands[0]
        assert cmd.command_type == CommandType.JUMP
        assert cmd.args["delta"] == 0.5
        assert cmd.args["future_state"] == "creative"

    def test_parse_predict_command(self):
        """Test parsing predict command."""
        am = AriannaMethods()
        text = "predict(next_delta=0.8)"
        commands = am.parse(text)
        
        assert len(commands) == 1
        cmd = commands[0]
        assert cmd.command_type == CommandType.PREDICT
        assert cmd.args["next_delta"] == 0.8

    def test_parse_time_travel_command(self):
        """Test parsing time_travel command."""
        am = AriannaMethods()
        text = "time_travel(offset=-10)"
        commands = am.parse(text)
        
        assert len(commands) == 1
        cmd = commands[0]
        assert cmd.command_type == CommandType.TIME_TRAVEL
        assert cmd.args["offset"] == -10

    def test_parse_prophecy_command(self):
        """Test parsing prophecy command."""
        am = AriannaMethods()
        text = "prophecy(vision='emergence', strength=0.7)"
        commands = am.parse(text)
        
        assert len(commands) == 1
        cmd = commands[0]
        assert cmd.command_type == CommandType.PROPHECY
        assert cmd.args["vision"] == "emergence"
        assert cmd.args["strength"] == 0.7

    def test_parse_multiple_commands(self):
        """Test parsing multiple commands in one text."""
        am = AriannaMethods()
        text = """
        First jump(delta=0.5, future_state='creative') then
        predict(next_delta=0.8) and finally
        prophecy(vision='growth', strength=0.6)
        """
        commands = am.parse(text)
        
        assert len(commands) == 3
        assert commands[0].command_type == CommandType.JUMP
        assert commands[1].command_type == CommandType.PREDICT
        assert commands[2].command_type == CommandType.PROPHECY

    def test_parse_with_surrounding_text(self):
        """Test parsing commands embedded in regular text."""
        am = AriannaMethods()
        text = "I want to jump(delta=0.7, future_state='focused') into the future"
        commands = am.parse(text)
        
        assert len(commands) == 1
        assert commands[0].command_type == CommandType.JUMP


class TestCommandExecution:
    """Test execution of ariannamethod commands."""

    def test_execute_jump_modifies_temperature(self):
        """Test that jump command modifies temperature."""
        am = AriannaMethods()
        context = ExecutionContext(current_temperature=0.8)
        
        cmd = Command(
            command_type=CommandType.JUMP,
            args={"delta": 0.5, "future_state": "creative"}
        )
        
        result = am.executor.execute(cmd, context)
        
        assert result["success"]
        assert "jumped_to" in result
        assert result["jumped_to"] == "creative"
        # Temperature should have changed
        assert context.current_temperature != 0.8

    def test_execute_predict_adjusts_temperature(self):
        """Test that predict command adjusts temperature."""
        am = AriannaMethods()
        context = ExecutionContext(current_temperature=0.8)
        original_temp = context.current_temperature
        
        cmd = Command(
            command_type=CommandType.PREDICT,
            args={"next_delta": 0.5}
        )
        
        result = am.executor.execute(cmd, context)
        
        assert result["success"]
        assert context.current_temperature != original_temp

    def test_execute_time_travel_sets_offset(self):
        """Test that time_travel command sets offset."""
        am = AriannaMethods()
        context = ExecutionContext()
        
        cmd = Command(
            command_type=CommandType.TIME_TRAVEL,
            args={"offset": -5}
        )
        
        result = am.executor.execute(cmd, context)
        
        assert result["success"]
        assert context.time_offset == -5
        assert result["direction"] == "past"

    def test_execute_prophecy_adds_vision(self):
        """Test that prophecy command adds vision."""
        am = AriannaMethods()
        context = ExecutionContext(current_temperature=1.0)
        
        cmd = Command(
            command_type=CommandType.PROPHECY,
            args={"vision": "emergence", "strength": 0.7}
        )
        
        result = am.executor.execute(cmd, context)
        
        assert result["success"]
        assert "emergence" in context.prophecy_visions
        assert len(context.prophecy_visions) == 1
        # Temperature should be reduced by prophecy strength
        assert context.current_temperature < 1.0

    def test_execute_resonate_boosts_shard(self):
        """Test that resonate command boosts shard."""
        am = AriannaMethods()
        context = ExecutionContext()
        
        cmd = Command(
            command_type=CommandType.RESONATE,
            args={"shard_id": "abc123", "boost": 2.0}
        )
        
        result = am.executor.execute(cmd, context)
        
        assert result["success"]
        assert "abc123" in context.resonance_boosts
        assert context.resonance_boosts["abc123"] == 2.0

    def test_execute_drift_modifies_context(self):
        """Test that drift command modifies context."""
        am = AriannaMethods()
        context = ExecutionContext(current_temperature=0.8)
        
        cmd = Command(
            command_type=CommandType.DRIFT,
            args={"direction": "curious", "momentum": 0.6}
        )
        
        result = am.executor.execute(cmd, context)
        
        assert result["success"]
        assert "drift_curious" in context.delta_modifications
        assert context.delta_modifications["drift_curious"] == 0.6


class TestExecutionContext:
    """Test execution context behavior."""

    def test_context_reset_modifications(self):
        """Test resetting modifications in context."""
        context = ExecutionContext()
        
        # Add some modifications
        context.delta_modifications["test"] = 1.0
        context.resonance_boosts["shard1"] = 2.0
        context.prophecy_visions.append("vision1")
        context.time_offset = -5
        
        # Reset
        context.reset_modifications()
        
        assert len(context.delta_modifications) == 0
        assert len(context.resonance_boosts) == 0
        assert len(context.prophecy_visions) == 0
        assert context.time_offset == 0


class TestIntegration:
    """Test integration of parse and execute."""

    def test_parse_and_execute_single_command(self):
        """Test parsing and executing a single command."""
        am = AriannaMethods()
        context = ExecutionContext()
        
        text = "jump(delta=0.5, future_state='creative')"
        commands, results = am.parse_and_execute(text, context)
        
        assert len(commands) == 1
        assert len(results) == 1
        assert results[0]["success"]

    def test_parse_and_execute_multiple_commands(self):
        """Test parsing and executing multiple commands."""
        am = AriannaMethods()
        context = ExecutionContext()
        
        text = """
        jump(delta=0.5, future_state='creative')
        predict(next_delta=0.8)
        prophecy(vision='emergence', strength=0.7)
        """
        commands, results = am.parse_and_execute(text, context)
        
        assert len(commands) == 3
        assert len(results) == 3
        assert all(r["success"] for r in results)

    def test_command_history_tracking(self):
        """Test that command history is tracked."""
        am = AriannaMethods()
        context = ExecutionContext()
        
        text = "jump(delta=0.5, future_state='creative')"
        commands, results = am.parse_and_execute(text, context)
        
        stats = am.stats()
        assert stats["total_commands"] == 1
        assert "jump" in stats["command_counts"]

    def test_available_commands(self):
        """Test that available commands can be listed."""
        am = AriannaMethods()
        commands = am.available_commands()
        
        assert len(commands) > 0
        assert any("jump" in cmd for cmd in commands.keys())
        assert any("prophecy" in cmd for cmd in commands.keys())


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_command_syntax(self):
        """Test that invalid syntax doesn't crash."""
        am = AriannaMethods()
        text = "jump(this is not valid syntax"
        commands = am.parse(text)
        
        # Should return empty list for invalid syntax
        assert len(commands) == 0

    def test_temperature_bounds(self):
        """Test that temperature stays within bounds."""
        am = AriannaMethods()
        context = ExecutionContext(current_temperature=0.1)
        
        # Try to set very high temperature
        cmd = Command(
            command_type=CommandType.AMPLIFY,
            args={"factor": 100.0}
        )
        
        result = am.executor.execute(cmd, context)
        
        assert result["success"]
        # Temperature should be capped at 2.0
        assert context.current_temperature <= 2.0

    def test_multiple_prophecies(self):
        """Test multiple prophecies can be added."""
        am = AriannaMethods()
        context = ExecutionContext()
        
        text = """
        prophecy(vision='emergence', strength=0.5)
        prophecy(vision='growth', strength=0.6)
        """
        commands, results = am.parse_and_execute(text, context)
        
        assert len(context.prophecy_visions) == 2
        assert "emergence" in context.prophecy_visions
        assert "growth" in context.prophecy_visions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
