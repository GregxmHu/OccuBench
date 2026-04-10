"""
OccuBench Evaluation Harness.

Usage:
    python -m occubench.evaluate \
        --agent-model gpt-4o \
        --world-model gpt-4o \
        --env-mode E0 \
        --api-key YOUR_KEY \
        --max-workers 8
"""

import argparse
import json
import os
import copy
import threading
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .lwm import WorldModelRegistry, LWMEnvironment, create_client
from .agent import ExecutorAgent
from .verifier import Verifier
from .fault_injection import build_fault_prompt

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


def load_tasks(eval_data_path: str, scenario_pool_path: str = None, task_ids: list = None, categories: list = None):
    """Load evaluation tasks and join with scenario metadata."""
    # Load scenario metadata
    scenario_meta = {}
    if scenario_pool_path and os.path.exists(scenario_pool_path):
        with open(scenario_pool_path, "r") as f:
            for line in f:
                d = json.loads(line)
                scenario_meta[d["task_scenario_name"]] = {
                    "category": d.get("category", "Unknown"),
                    "domain": d.get("domain", "Unknown"),
                }

    # Load tasks
    tasks = []
    with open(eval_data_path, "r") as f:
        for line in f:
            task = json.loads(line)
            # Join metadata
            meta = scenario_meta.get(task["task_scenario_name"], {})
            task["category"] = meta.get("category", "Unknown")
            task["domain"] = meta.get("domain", "Unknown")
            tasks.append(task)

    # Filter by task_ids
    if task_ids:
        task_id_set = set(task_ids)
        tasks = [t for t in tasks if t["task_id"] in task_id_set]

    # Filter by categories
    if categories:
        cat_set = set(categories)
        tasks = [t for t in tasks if t["category"] in cat_set]

    return tasks


def load_completed(results_path: str) -> set:
    """Load already-completed task IDs for resume support."""
    completed = set()
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    completed.add(d["task_id"])
                except:
                    pass
    return completed


def evaluate_single(
    task: dict,
    registry: WorldModelRegistry,
    agent_model: str,
    world_model: str,
    verifier_model: str,
    api_key: str,
    base_url: str,
    env_mode: str,
    fault_count: int,
    fault_duration: int,
    with_plan: bool,
    config_dir: str,
) -> dict:
    """Evaluate a single task."""
    # Load environment config
    config = registry.get(task["env_name"])

    # Inject faults if needed
    if env_mode != "E0":
        fault_prompt = build_fault_prompt(env_mode, fault_count, fault_duration)
        config = copy.deepcopy(config)
        config["world_model_system_prompt"] += "\n\n" + fault_prompt

    # Create clients
    wm_client = create_client(api_key, base_url)
    agent_client = create_client(api_key, base_url)
    verifier_client = create_client(api_key, base_url)

    # Create environment
    env = LWMEnvironment(config, wm_client, world_model)

    # Run agent
    agent = ExecutorAgent(agent_model, agent_client)
    solution_plan = task.get("solution_plan") if with_plan else None

    result = agent.execute(
        env=env,
        task_scenario_name=task["task_scenario_name"],
        agent_instruction=task["agent_instruction"],
        solution_plan=solution_plan,
    )

    trajectory = result["trajectory"]
    step_count = result["step_count"]

    # Verify
    verifier = Verifier(verifier_model, verifier_client)
    initial_state = config.get("task_initial_state", "{}")
    state_description = config.get("state_description", "")
    if isinstance(initial_state, dict):
        initial_state = json.dumps(initial_state, ensure_ascii=False)

    verification = verifier.check(
        task_scenario_name=task["task_scenario_name"],
        task_initial_state=initial_state,
        state_description=state_description,
        agent_instruction=task["agent_instruction"],
        verification_plan=task["verification_plan"],
        trajectory=trajectory,
    )

    return {
        "task_id": task["task_id"],
        "task_scenario_name": task["task_scenario_name"],
        "category": task.get("category", "Unknown"),
        "domain": task.get("domain", "Unknown"),
        "difficulty_level": task.get("difficulty_level"),
        "agent_model": agent_model,
        "world_model": world_model,
        "env_mode": env_mode,
        "fault_count": fault_count if env_mode != "E0" else 0,
        "fault_duration": fault_duration if env_mode != "E0" else 0,
        "with_plan": with_plan,
        "is_correct": verification["is_correct"],
        "feedback": verification["feedback"],
        "step_count": step_count,
        "trajectory": trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description="OccuBench Evaluation")
    parser.add_argument("--agent-model", required=True, help="Agent model name")
    parser.add_argument("--world-model", default="gpt-4o", help="World model name")
    parser.add_argument("--verifier-model", default=None, help="Verifier model (default: same as world-model)")
    parser.add_argument("--env-mode", default="E0", choices=["E0", "E1", "E2", "E3"])
    parser.add_argument("--fault-count", type=int, default=2)
    parser.add_argument("--fault-duration", type=int, default=2)
    parser.add_argument("--with-plan", action="store_true")
    parser.add_argument("--eval-data", default="data/eval_benchmark_solvable.jsonl")
    parser.add_argument("--scenario-pool", default="data/task_scenario_all_pool.jsonl")
    parser.add_argument("--config-dir", default="data/world_model_configs")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--task-ids", type=int, nargs="+", default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if not args.verifier_model:
        args.verifier_model = args.world_model

    # Build run name
    if not args.run_name:
        args.run_name = f"{args.agent_model}__{args.world_model}__{args.env_mode}"
        if args.env_mode != "E0":
            args.run_name += f"__fc{args.fault_count}_fd{args.fault_duration}"

    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.jsonl")

    if args.debug:
        from .debug import set_debug
        debug_log_path = os.path.join(run_dir, "debug.log")
        set_debug(True, debug_log_path)
        print(f"Debug logging enabled -> {debug_log_path}")

    # Save args
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load tasks
    tasks = load_tasks(args.eval_data, args.scenario_pool, args.task_ids, args.categories)
    print(f"Loaded {len(tasks)} tasks")

    # Resume support
    completed = load_completed(results_path)
    pending = [t for t in tasks if t["task_id"] not in completed]
    print(f"Completed: {len(completed)}, Pending: {len(pending)}")

    if not pending:
        print("No tasks to evaluate.")
        return

    # Initialize registry
    registry = WorldModelRegistry(args.config_dir)

    # Print info
    print(f"\nRun: {args.run_name}")
    print(f"Agent: {args.agent_model}")
    print(f"World Model: {args.world_model}")
    print(f"Env Mode: {args.env_mode}")
    print(f"Workers: {args.max_workers}\n")

    # Evaluate
    def eval_task(task):
        return evaluate_single(
            task=task,
            registry=registry,
            agent_model=args.agent_model,
            world_model=args.world_model,
            verifier_model=args.verifier_model,
            api_key=args.api_key,
            base_url=args.base_url,
            env_mode=args.env_mode,
            fault_count=args.fault_count,
            fault_duration=args.fault_duration,
            with_plan=args.with_plan,
            config_dir=args.config_dir,
        )

    correct = 0
    total = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(eval_task, t): t for t in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = future.result()
                total += 1
                if result["is_correct"]:
                    correct += 1

                with _write_lock:
                    with open(results_path, "a") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                task = futures[future]
                logger.error(f"Task {task['task_id']} failed: {e}")

    # Print summary
    total_all = len(completed) + total
    correct_all = correct  # We don't know previous correct count without re-reading
    print(f"\nResults: {args.run_name}")
    print(f"New: {correct}/{total} ({correct/total*100:.1f}%)" if total > 0 else "")
    print(f"Total tasks evaluated: {total_all}")
    print(f"Results saved to: {results_path}")

    if args.debug:
        from .debug import close_debug
        close_debug()
        print(f"Debug log saved to: {os.path.join(run_dir, 'debug.log')}")


if __name__ == "__main__":
    main()
