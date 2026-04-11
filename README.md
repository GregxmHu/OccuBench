# OccuBench: Evaluating AI Agents on Real-World Professional Tasks via Language World Models

OccuBench is a benchmark for evaluating AI agents on **100 real-world professional task scenarios** across **10 industry categories** and **65 specialized domains**, using **Language World Models (LWMs)** to simulate domain-specific environments.

## Important Notice

> **This repository provides reference implementation code and benchmark data for the OccuBench evaluation framework.** The internal evaluation system used in our paper is built on a proprietary agent framework that cannot be publicly released due to commercial restrictions. The code provided here is a **clean, standalone reimplementation** designed to demonstrate how LWM-based environments interact with agents, enabling researchers to integrate OccuBench into their own agent harnesses.
>
> We will continue to support evaluation of new models on OccuBench. If you would like a specific model evaluated, or wish to contribute results, please feel free to [open an issue](https://github.com/xxx/occubench/issues) or contact the authors directly.

## Key Features

- **382 evaluation instances** covering professional tasks from emergency triage to nuclear reactor monitoring
- **Language World Model** simulation — any LLM can serve as the environment simulator
- **Fault injection** — evaluate agent robustness under explicit errors (E1), implicit data degradation (E2), and mixed faults (E3)
- **Automated verification** with 3-vote majority rubric-based checking
- **Framework-agnostic** — integrate LWM environments into any agent framework via the OpenAI-compatible API

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Evaluation

```bash
# Basic evaluation (clean environment)
python -m occubench.evaluate \
    --agent-model gpt-4o \
    --world-model gpt-4o \
    --api-key YOUR_API_KEY \
    --env-mode E0 \
    --max-workers 8

# With fault injection (implicit faults)
python -m occubench.evaluate \
    --agent-model gpt-4o \
    --world-model gpt-4o \
    --api-key YOUR_API_KEY \
    --env-mode E2 \
    --fault-count 2 \
    --fault-duration 2

# Single task debug
python -m occubench.evaluate \
    --agent-model gpt-4o \
    --world-model gpt-4o \
    --api-key YOUR_API_KEY \
    --task-ids 11 \
    --max-workers 1
```

### Using a Custom API Endpoint

OccuBench works with any OpenAI-compatible API:

```bash
# DashScope
python -m occubench.evaluate \
    --agent-model qwen-plus \
    --world-model qwen-plus \
    --api-key YOUR_DASHSCOPE_KEY \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1

# Local vLLM server
python -m occubench.evaluate \
    --agent-model meta-llama/Llama-3-70b \
    --world-model meta-llama/Llama-3-70b \
    --base-url http://localhost:8000/v1
```

### Integrating with Your Own Agent Framework

The core LWM environment can be used independently of our evaluation harness. Here is a minimal example:

```python
from occubench.lwm import WorldModelRegistry, LWMEnvironment, create_client

# Load environment
registry = WorldModelRegistry("data/world_model_configs")
config = registry.get("env_153c260a856a")

# Create LWM client
client = create_client(api_key="YOUR_KEY", base_url="https://api.openai.com/v1")
env = LWMEnvironment(config, client, world_model="gpt-4o")

# Get tool schemas (pass these to your agent)
tools = env.get_tool_schemas()

# Simulate a tool call (your agent decides which tool to call)
observation = env.simulate("get_property_metadata", '{"property_id": "OAK-88"}')
print(observation)
```

You can plug this `LWMEnvironment` into any agent framework (LangChain, CrewAI, AutoGen, etc.) by using `env.get_tool_schemas()` for the tool definitions and `env.simulate()` for executing tool calls.

## Project Structure

```
occubench/
├── lwm.py              # Language World Model core
├── agent.py            # Reference agent execution loop
├── verifier.py         # Rubric-based verification (3-vote majority)
├── evaluate.py         # Evaluation harness (CLI)
├── fault_injection.py  # E1/E2/E3 fault templates
└── debug.py            # Debug flag and utilities

data/
├── eval_benchmark_solvable.jsonl       # 382 evaluation tasks
├── task_scenario_all_pool.jsonl        # Scenario metadata
└── world_model_configs/                # Environment configurations
```

## Environment Modes

| Mode | Description | Fault Type |
|------|-------------|------------|
| E0 | Clean environment | None |
| E1 | Explicit faults | HTTP 500, timeouts, connection refused |
| E2 | Implicit faults | Truncated data, missing fields, stale values |
| E3 | Mixed faults | 50% explicit + 50% implicit |

## Debugging

Use the `--debug` flag to enable verbose logging of the agent-LWM interaction:

```bash
python -m occubench.evaluate \
    --agent-model qwen3.5-plus \
    --world-model qwen3.5-plus \
    --api-key YOUR_API_KEY \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --task-ids 11 \
    --max-workers 1 \
    --debug
```

Debug output is printed to terminal and saved to `debug.log` in the results directory. Example output:

```
[AGENT] Step 1, messages=2
[AGENT] finish_reason=tool_calls, has_content=False, tool_calls=2
[AGENT] Tool call: get_current_date({})
[LWM] Simulating: get_current_date, history_len=0
[LWM] Observation: {"current_date": "2023-10-27"}
[AGENT] Tool call: fetch_unit_level_records({"property_id": "OAK-88"})
[LWM] Simulating: fetch_unit_level_records, history_len=1
[LWM] Observation: {"units": [{"unit_id": "A-1", "monthly_base_rent": 2800, ...}]}
[AGENT] Step 2, messages=5
[AGENT] finish_reason=tool_calls, has_content=False, tool_calls=1
[AGENT] Tool call: get_property_metadata({"property_id": "OAK-88"})
[LWM] Simulating: get_property_metadata, history_len=2
[LWM] Observation: {"property_name": "Oakwood Gardens", "total_units": 15, "status": "Active"}
[AGENT] Step 3, messages=7
[AGENT] finish_reason=stop, has_content=True, tool_calls=0
[VERIFIER] Votes: 0 TRUE, 3 FALSE
[VERIFIER] Vote 1: correct=False, feedback=The Agent failed the task based on multiple critical errors...
[VERIFIER] Vote 2: correct=False, feedback=The agent failed to identify the correct unit counts...
[VERIFIER] Vote 3: correct=False, feedback=Hallucination of Current Date: the tool returned '2023-10-27'...
```

## Results

Results are saved to `results/{run_name}/`:

```
results/
└── qwen3.5-plus__qwen3.5-plus__E0/
    ├── args.json           # Run configuration (for reproducibility)
    ├── results.jsonl       # Per-task evaluation results
    └── debug.log           # Debug log (only when --debug is enabled)
```

- **`args.json`**: All command-line arguments used for this run, enabling exact reproducibility.
- **`results.jsonl`**: One JSON object per line, containing the task ID, agent trajectory, verifier feedback, and pass/fail result.
- **`debug.log`**: Step-by-step log of agent tool calls, LWM simulated observations, and verifier votes with feedback.

The evaluation harness supports **resume**: if you re-run the same command, it will skip already-completed tasks and only evaluate pending ones.

Each line in `results.jsonl` contains:

```json
{
    "task_id": 11,
    "task_scenario_name": "Property Valuation Assessment",
    "category": "Business & Enterprise",
    "domain": "Real Estate",
    "difficulty_level": 2,
    "agent_model": "qwen3.5-plus",
    "world_model": "qwen3.5-plus",
    "env_mode": "E0",
    "fault_count": 0,
    "fault_duration": 0,
    "with_plan": false,
    "is_correct": false,
    "feedback": "The Agent failed the task based on ...",
    "step_count": 3,
    "trajectory": "[Agent Action]: Call tool `get_current_date` ..."
}
```

## Citation

```bibtex
@article{hu2026occubench,
  title={OccuBench: Evaluating AI Agents on Real-World Professional Tasks via Language World Models},
  author={Hu, Xiaomeng and Zhang, Yinger and Huang, Fei and Tu, Jianhong and Su, Yang and Deng, Lianghao and Liu, Dayiheng and Ho, Tsung-Yi},
  journal={arXiv preprint arXiv:2506.xxxxx},
  year={2025}
}
```

## License

Apache 2.0
