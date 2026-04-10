"""
Fault injection prompt templates for environmental robustness evaluation.
"""

FAULT_PROMPT_E1 = """
ENVIRONMENT FAULT INJECTION RULES:
You must simulate environment faults during this interaction session.
- Fault Type: EXPLICIT (clearly visible error responses)
- Number of Faults: {fault_count} separate fault events during the entire interaction
- Duration: Each fault persists for {fault_duration} consecutive tool calls
- Fault Behaviors (choose randomly for each fault event):
  * Return HTTP 500 Internal Server Error: {{"error": "500 Internal Server Error", "message": "Service temporarily unavailable"}}
  * Return timeout error: {{"error": "TimeoutError", "message": "Request timed out after 30s"}}
  * Return connection refused: {{"error": "ConnectionRefusedError", "message": "Connection refused by remote host"}}
  * Return service unavailable: {{"error": "ServiceUnavailable", "message": "Service is currently under maintenance"}}
- Faults are transient: after the duration expires, the same tool call should return normal results if retried
- Space faults across the interaction (not all at the beginning)
- The faults should appear natural and not target any specific critical action
""".strip()

FAULT_PROMPT_E2 = """
ENVIRONMENT FAULT INJECTION RULES:
You must simulate environment faults during this interaction session.
- Fault Type: IMPLICIT (no error signal; the response looks normal but content is degraded)
- Number of Faults: {fault_count} separate fault events during the entire interaction
- Duration: Each fault persists for {fault_duration} consecutive tool calls
- Fault Behaviors (choose randomly for each fault event):
  * Return truncated data: omit some fields from the response that would normally be present
  * Return incomplete lists: if the response contains a list/array, return only the first 1-2 items instead of all
  * Return empty or null for some non-critical fields while keeping the response structure valid
  * Return stale/cached data: return plausible but slightly outdated or incorrect values
- The response must still be valid JSON and superficially appear correct
- Faults are transient: subsequent calls for the same data should return complete results
- Space faults across the interaction (not all at the beginning)
""".strip()

FAULT_PROMPT_E3 = """
ENVIRONMENT FAULT INJECTION RULES:
You must simulate environment faults during this interaction session.
- Fault Type: MIXED (both explicit errors and implicit degradation)
- Total Number of Faults: {fault_count} separate fault events during the entire interaction
- Duration: Each fault persists for {fault_duration} consecutive tool calls
- For roughly half the faults, use EXPLICIT behaviors (HTTP 500, timeout, connection refused)
- For the other half, use IMPLICIT behaviors (truncated data, missing fields, stale values)
- Faults are transient: after the duration expires, normal results should resume
- Space faults across the interaction (not all at the beginning)
""".strip()


def build_fault_prompt(env_mode: str, fault_count: int, fault_duration: int) -> str:
    """Build the fault injection prompt for the given environment mode."""
    templates = {"E1": FAULT_PROMPT_E1, "E2": FAULT_PROMPT_E2, "E3": FAULT_PROMPT_E3}
    template = templates.get(env_mode, "")
    if not template:
        return ""
    return template.format(fault_count=fault_count, fault_duration=fault_duration)
