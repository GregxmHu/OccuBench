"""
Agent execution loop.

Runs an LLM agent against an LWM environment using the OpenAI tool-calling API.
"""

import json
import logging
import os
from typing import Dict, List, Optional
from openai import OpenAI

from .lwm import LWMEnvironment, create_client
from .debug import debug_print

logger = logging.getLogger(__name__)

EXECUTOR_SYSTEM_PROMPT = """You are a professional AI assistant executing tasks in a simulated environment.
You have access to a set of tools that interact with the environment.
Your goal is to complete the given task by making appropriate tool calls.

Task Scenario: {task_scenario_name}

Please remember the current actual time: {current_time}

IMPORTANT:
- Think step by step before acting.
- Use the available tools to gather information and take actions.
- Do not hallucinate tool responses - always call the tool to get real data.
- Complete the task as thoroughly as possible.
"""


class ExecutorAgent:
    """
    Executes a task by running an LLM agent against an LWM environment.
    Uses the OpenAI tool-calling API for the agent loop.
    """

    def __init__(
        self,
        agent_model: str,
        agent_client: OpenAI = None,
        api_key: str = None,
        base_url: str = None,
        max_steps: int = 200,
        max_tokens: int = 12288,
    ):
        self.agent_model = agent_model
        self.client = agent_client or create_client(api_key, base_url)
        self.max_steps = max_steps
        self.max_tokens = max_tokens

    def execute(
        self,
        env: LWMEnvironment,
        task_scenario_name: str,
        agent_instruction: str,
        solution_plan: str = None,
    ) -> Dict:
        """
        Execute a task and return the trajectory.

        Args:
            env: LWM environment instance
            task_scenario_name: Name of the task scenario
            agent_instruction: The task instruction for the agent
            solution_plan: Optional solution plan hint

        Returns:
            Dict with 'trajectory' (formatted string) and 'step_count'
        """
        env.reset()

        # Build initial messages
        from datetime import datetime
        current_time = datetime.now().strftime("%A, %B %d, %Y")

        system_msg = EXECUTOR_SYSTEM_PROMPT.format(
            task_scenario_name=task_scenario_name,
            current_time=current_time,
        )

        prompt = f"Task Instruction:\n{agent_instruction}\n"
        if solution_plan:
            prompt += f"\nHere is a recommended solution plan you should **strictly** follow:\n{solution_plan}\n"
            prompt += "Please execute the plan step by step."
        else:
            prompt += "\nPlease solve this task."

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        tools = env.get_tool_schemas()
        trajectory_parts = []
        step_count = 0

        for step in range(self.max_steps):
            debug_print(f"[AGENT] Step {step+1}, messages={len(messages)}")
            try:
                response = self.client.chat.completions.create(
                    model=self.agent_model,
                    messages=messages,
                    tools=tools if tools else None,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                debug_print(f"[AGENT] LLM call failed: {e}")
                logger.error(f"Agent LLM call failed at step {step}: {e}")
                break

            choice = response.choices[0]
            message = choice.message
            debug_print(f"[AGENT] finish_reason={choice.finish_reason}, has_content={bool(message.content)}, tool_calls={len(message.tool_calls) if message.tool_calls else 0}")

            # Append assistant message
            assistant_msg = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_msg)

            # Log agent response
            if message.content:
                trajectory_parts.append(f"\n[Agent Response]: {message.content}\n")

            # If no tool calls, the agent is done
            if not message.tool_calls:
                break

            # Process each tool call
            for tc in message.tool_calls:
                tool_name = tc.function.name
                tool_args = tc.function.arguments
                step_count += 1

                debug_print(f"[AGENT] Tool call: {tool_name}({tool_args[:100]})")

                trajectory_parts.append(
                    f"\n[Agent Action]: Call tool `{tool_name}` with arguments: {tool_args}\n"
                )

                # Simulate the tool call via LWM
                observation = env.simulate(tool_name, tool_args)

                trajectory_parts.append(
                    f"\n[Environment Observation] (from {tool_name}): {observation}\n"
                )

                # Append tool response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": observation,
                })

            if choice.finish_reason == "stop":
                break

        return {
            "trajectory": "".join(trajectory_parts),
            "step_count": step_count,
        }
