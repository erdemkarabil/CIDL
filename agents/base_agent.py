"""
VoltOptimizer - Base Agent Class
=================================
Base class inherited by all agents. Inspired by CrewAI architecture:
  - Each agent has a role and a goal
  - Can use tools
  - Can reason (ReAct pattern)
  - Can self-correct
  - Can exchange messages with other agents

Architecture Pattern: ReAct (Reasoning + Acting)
    1. Observe → 2. Think → 3. Act → 4. Evaluate → 5. Repeat
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AGENT_CONFIG
from utils.logger import logger


class BaseAgent:
    """
    Base class for all VoltOptimizer agents.

    Provides a CrewAI-like interface:
      - role: The agent's role (e.g. "Battery Health Guardian")
      - goal: The agent's objective
      - backstory: Background context for the agent
      - tools: List of tools the agent can use
    """

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: list = None,
        verbose: bool = None,
        max_reasoning_steps: int = None,
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.verbose = (verbose if verbose is not None
                        else AGENT_CONFIG["verbose"])
        self.max_reasoning_steps = (
            max_reasoning_steps or AGENT_CONFIG["max_reasoning_steps"]
        )

        self.reasoning_history = []
        self.inbox = []
        self.outputs = {}

        if self.verbose:
            logger.log("system", f"Agent created: {self.role}")

    def use_tool(self, tool_name: str, tool_instance, method: str,
                 **kwargs) -> dict:
        """
        Invokes a tool and returns its result.

        Args:
            tool_name: Display name of the tool
            tool_instance: Tool object
            method: Method name to call
            **kwargs: Parameters to pass to the method

        Returns:
            Tool output
        """
        if self.verbose:
            logger.agent_action(self.role,
                                f"Using tool '{tool_name}' → {method}()")

        tool_method = getattr(tool_instance, method)
        result = tool_method(**kwargs)

        if self.verbose:
            logger.agent_action(self.role,
                                f"Result received from '{tool_name}' ✓")

        return result

    def reason(self, observation: dict, step: int = 0) -> str:
        """
        Reasoning based on an observation.

        Overridden by subclasses.

        Args:
            observation: Observation data
            step: Reasoning step number

        Returns:
            Reasoning text
        """
        thought = "Observation received, analysing..."
        self.reasoning_history.append({
            "step": step,
            "observation": observation,
            "thought": thought,
        })

        if self.verbose:
            logger.agent_thinking(self.role, thought, step)

        return thought

    def self_correct(self, previous_output: dict, feedback: str) -> dict:
        """
        Self-correction mechanism.

        Re-evaluates the previous output against the feedback
        and produces a corrected output.

        Args:
            previous_output: Previous output
            feedback: Feedback text

        Returns:
            Corrected output
        """
        if self.verbose:
            logger.agent_thinking(
                self.role,
                f"Self-correction active: {feedback}"
            )

        corrected = previous_output.copy()
        corrected["self_corrected"] = True
        corrected["correction_reason"] = feedback

        return corrected

    def receive_message(self, from_agent: str, message: dict):
        """
        Receives a message from another agent.

        Args:
            from_agent: Sender agent name
            message: Message content
        """
        self.inbox.append({
            "from": from_agent,
            "message": message,
        })

        if self.verbose:
            logger.log("system",
                       f"[{self.role}] ← Message received from: [{from_agent}]")

    def send_message(self, to_agent, message: dict):
        """
        Sends a message to another agent.

        Args:
            to_agent: Target agent object
            message: Message content
        """
        if self.verbose:
            logger.log("system",
                       f"[{self.role}] → Sending message to: [{to_agent.role}]")

        to_agent.receive_message(self.role, message)

    def execute(self, task_input: dict) -> dict:
        """
        Main task execution method.
        Overridden by subclasses.

        Args:
            task_input: Task input data

        Returns:
            Task output
        """
        raise NotImplementedError(
            "execute() must be implemented in a subclass"
        )

    def get_summary(self) -> dict:
        """Returns a summary of the agent."""
        return {
            "role": self.role,
            "goal": self.goal,
            "tools": [type(t).__name__ for t in self.tools],
            "reasoning_steps": len(self.reasoning_history),
            "messages_received": len(self.inbox),
            "outputs": list(self.outputs.keys()),
        }
