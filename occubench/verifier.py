"""
Verifier - Evaluates agent trajectories against verification rubrics.
Uses 3-vote majority for noise reduction.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from openai import OpenAI

from .lwm import create_client
from .debug import debug_print

logger = logging.getLogger(__name__)

VERIFIER_PROMPT = """You are a strict verifier for an AI agent evaluation benchmark.

## Task Scenario
{task_scenario_name}

## Initial State
{task_initial_state}

## State Description
{state_description}

## Agent Instruction
{agent_instruction}

## Verification Plan
{verification_plan}

## Agent Trajectory
{trajectory}

---

Based on the verification plan above, evaluate whether the agent successfully completed the task.

You must:
1. Check each criterion in the verification plan against the agent's trajectory.
2. Verify that the agent's actions and observations satisfy all required conditions.
3. Be strict: if any required criterion is not met, the task is a failure.

Respond in the following JSON format:
```json
{{
    "is_correct": true/false,
    "feedback": "Detailed explanation of why the task passed or failed."
}}
```
"""


class Verifier:
    """Evaluates agent trajectories using 3-vote majority verification."""

    def __init__(
        self,
        verifier_model: str,
        client: OpenAI = None,
        api_key: str = None,
        base_url: str = None,
        num_votes: int = 3,
    ):
        self.model = verifier_model
        self.client = client or create_client(api_key, base_url)
        self.num_votes = num_votes

    def _single_check(
        self,
        task_scenario_name: str,
        task_initial_state: str,
        state_description: str,
        agent_instruction: str,
        verification_plan: str,
        trajectory: str,
    ) -> Dict:
        """Run a single verification check."""
        prompt = VERIFIER_PROMPT.format(
            task_scenario_name=task_scenario_name,
            task_initial_state=task_initial_state,
            state_description=state_description,
            agent_instruction=agent_instruction,
            verification_plan=verification_plan,
            trajectory=trajectory,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )
            content = response.choices[0].message.content

            # Parse JSON from response
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return {
                    "is_correct": bool(result.get("is_correct", False)),
                    "feedback": result.get("feedback", ""),
                }
        except Exception as e:
            logger.error(f"Verification failed: {e}")

        return {"is_correct": False, "feedback": "Verification error"}

    def check(
        self,
        task_scenario_name: str,
        task_initial_state: str,
        state_description: str,
        agent_instruction: str,
        verification_plan: str,
        trajectory: str,
    ) -> Dict:
        """
        Run majority-vote verification.

        Returns:
            Dict with 'is_correct' (bool) and 'feedback' (str)
        """
        with ThreadPoolExecutor(max_workers=self.num_votes) as pool:
            futures = [
                pool.submit(
                    self._single_check,
                    task_scenario_name,
                    task_initial_state,
                    state_description,
                    agent_instruction,
                    verification_plan,
                    trajectory,
                )
                for _ in range(self.num_votes)
            ]
            results = [f.result() for f in futures]

        true_votes = [r for r in results if r["is_correct"]]
        false_votes = [r for r in results if not r["is_correct"]]

        debug_print(f"[VERIFIER] Votes: {len(true_votes)} TRUE, {len(false_votes)} FALSE")
        for i, r in enumerate(results):
            debug_print(f"[VERIFIER] Vote {i+1}: correct={r['is_correct']}, feedback={r['feedback'][:200]}")

        if len(true_votes) > len(false_votes):
            return true_votes[0]
        else:
            return false_votes[0]
