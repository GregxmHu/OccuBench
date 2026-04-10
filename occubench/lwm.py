"""
Language World Model (LWM) - Core simulation engine.

The LWM simulates tool-response-level environment interaction by using an LLM
to generate observations conditioned on the environment configuration and
conversation history.
"""

import json
import os
import re
import time
import random
import hashlib
import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI
from .debug import debug_print

logger = logging.getLogger(__name__)


# ============================================================
# LLM Inference
# ============================================================

def create_client(api_key: str = None, base_url: str = None) -> OpenAI:
    """Create an OpenAI-compatible client."""
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def call_llm(
    client: OpenAI,
    model: str,
    messages: List[Dict],
    max_tokens: int = 12288,
    max_retries: int = 50,
    delay: float = 1.0,
) -> str:
    """Call LLM with retry logic. Returns the content string."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if content:
                return content
            raise ValueError("Empty response from LLM")
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def call_world_model(client: OpenAI, model: str, messages: List[Dict], max_retries: int = 100) -> str:
    """Call the World Model LLM and extract content from <predicted_observation> tags."""
    raw = call_llm(client, model, messages, max_retries=max_retries)
    # Extract content between tags
    match = re.search(r"<predicted_observation>(.*?)</predicted_observation>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw


# ============================================================
# JSON Parsing
# ============================================================

def parse_json_from_response(response: str) -> Optional[Any]:
    """
    Parse JSON from LLM response. Handles:
    - Direct JSON
    - JSON wrapped in markdown code blocks
    """
    if not isinstance(response, str):
        return None
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                return None
        return None


# ============================================================
# Tool Call Simulation
# ============================================================

def simulate_tool_call(
    client: OpenAI,
    world_model: str,
    environment_system_prompt: str,
    tool_call_json: str,
    tool_history: List[Dict],
    max_retries: int = 10,
) -> str:
    """
    Simulate a single tool call using the Language World Model.

    Args:
        client: OpenAI client for the world model
        world_model: Model name for the world model
        environment_system_prompt: The environment's system prompt
        tool_call_json: JSON string of the current tool call (name + arguments)
        tool_history: List of previous {action, observation} pairs
        max_retries: Number of retries for parsing failures

    Returns:
        JSON string of the simulated observation
    """
    system_prompt = (
        environment_system_prompt
        + "\n### OUTPUT FORMAT INSTRUCTIONS\n"
        "Your response must consist ONLY of a single valid JSON object. "
        "You MUST wrap the JSON object within <predicted_observation> and "
        "</predicted_observation> tags. Do not include any introductory text, "
        "conversational filler, or Markdown code blocks outside or inside the tags."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history
    for entry in tool_history:
        messages.append({
            "role": "user",
            "content": f"<agent_action>\n{entry['action']}\n</agent_action>"
        })
        messages.append({
            "role": "assistant",
            "content": f"<predicted_observation>\n{entry['observation']}\n</predicted_observation>"
        })

    # Add current tool call
    messages.append({
        "role": "user",
        "content": (
            f"<agent_action>\n{tool_call_json}\n</agent_action>\n"
            "### OUTPUT FORMAT INSTRUCTIONS\n"
            "Your response must consist ONLY of a single valid JSON object. "
            "You MUST wrap the JSON object within <predicted_observation> and "
            "</predicted_observation> tags."
        ),
    })

    archived = []
    for attempt in range(max_retries):
        try:
            raw = call_world_model(client, world_model, messages, max_retries=50)
            if not raw:
                raise ValueError("Empty response")
            archived.append(raw)
            parsed = parse_json_from_response(raw)
            if parsed is not None:
                return raw
            logger.info(f"Response parsing failed, retry {attempt+1}/{max_retries}")
        except Exception as e:
            logger.info(f"Error in tool simulation: {e}")

        time.sleep(random.uniform(0.5, 1.5))

    return archived[-1] if archived else "{}"


# ============================================================
# Environment Registry
# ============================================================

class WorldModelRegistry:
    """Manages environment configuration storage and retrieval."""

    def __init__(self, registry_dir: str):
        self.registry_dir = registry_dir

    def get(self, env_name: str) -> Dict:
        """Load environment config by name."""
        file_path = os.path.join(self.registry_dir, f"{env_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Environment '{env_name}' not found at {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_environments(self) -> List[str]:
        """List all available environment names."""
        envs = []
        for f in os.listdir(self.registry_dir):
            if f.endswith(".json"):
                envs.append(f.replace(".json", ""))
        return envs


# ============================================================
# Environment (combines config + LWM client)
# ============================================================

class LWMEnvironment:
    """
    A Language World Model environment instance.

    Loads an environment config and provides a simulate() method
    that takes agent tool calls and returns simulated observations.
    """

    def __init__(self, config: Dict, client: OpenAI, world_model: str):
        self.env_name = config["environment_name"]
        self.system_prompt = config["world_model_system_prompt"]
        self.tools = config.get("action_set_definitions", [])
        self.initial_state = config.get("task_initial_state", "{}")
        self.state_description = config.get("state_description", "")
        self.client = client
        self.world_model = world_model
        self.history: List[Dict] = []  # {action, observation} pairs

    def get_tool_schemas(self) -> List[Dict]:
        """Return tool schemas in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for tool in self.tools
        ]

    def simulate(self, tool_name: str, arguments: str) -> str:
        """
        Simulate a tool call and return the observation.

        Args:
            tool_name: Name of the tool being called
            arguments: JSON string of the tool arguments

        Returns:
            JSON string of the simulated observation
        """
        tool_call_json = json.dumps(
            {"name": tool_name, "arguments": arguments}, ensure_ascii=False
        )

        debug_print(f"[LWM] Simulating: {tool_name}, history_len={len(self.history)}")

        observation = simulate_tool_call(
            client=self.client,
            world_model=self.world_model,
            environment_system_prompt=self.system_prompt,
            tool_call_json=tool_call_json,
            tool_history=self.history,
        )

        debug_print(f"[LWM] Observation: {observation[:200]}")

        self.history.append({"action": tool_call_json, "observation": observation})
        return observation

    def reset(self):
        """Reset the environment history."""
        self.history = []
