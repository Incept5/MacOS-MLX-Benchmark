"""Tool calling evaluation: tests whether models can correctly invoke tools."""

from __future__ import annotations

import json
import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCallScenario:
    id: str
    category: str  # single-tool, multi-choice, parameter-extraction, multi-step, refusal
    tools: list[dict[str, Any]]
    user_message: str
    expected_calls: list[dict[str, Any]]  # [{function, args}] or [] for refusal


@dataclass
class ToolCallResult:
    scenario_id: str
    category: str
    # Did the model produce a parseable tool call?
    json_valid: bool = False
    # Did it call the right function?
    function_correct: bool = False
    # Did it extract the right parameters? (partial match)
    params_correct: bool = False
    # For refusal scenarios: did it correctly NOT call a tool?
    refusal_correct: bool = False
    # Raw model output for debugging
    raw_output: str = ""
    parsed_function: str = ""
    parsed_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallEvalResults:
    total: int = 0
    json_valid_count: int = 0
    function_correct_count: int = 0
    params_correct_count: int = 0
    refusal_correct_count: int = 0
    refusal_total: int = 0
    results: list[ToolCallResult] = field(default_factory=list)

    @property
    def json_valid_rate(self) -> float:
        non_refusal = self.total - self.refusal_total
        return self.json_valid_count / non_refusal if non_refusal > 0 else 0.0

    @property
    def function_accuracy(self) -> float:
        non_refusal = self.total - self.refusal_total
        return self.function_correct_count / non_refusal if non_refusal > 0 else 0.0

    @property
    def param_accuracy(self) -> float:
        non_refusal = self.total - self.refusal_total
        return self.params_correct_count / non_refusal if non_refusal > 0 else 0.0

    @property
    def refusal_accuracy(self) -> float:
        return self.refusal_correct_count / self.refusal_total if self.refusal_total > 0 else 0.0

    @property
    def overall_accuracy(self) -> float:
        correct = self.function_correct_count + self.refusal_correct_count
        return correct / self.total if self.total > 0 else 0.0


def load_scenarios(path: str | Path) -> list[ToolCallScenario]:
    """Load tool calling scenarios from TOML."""
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)

    scenarios = []
    for s in data.get("scenario", []):
        scenarios.append(ToolCallScenario(
            id=s["id"],
            category=s["category"],
            tools=s.get("tools", []),
            user_message=s["user_message"],
            expected_calls=s.get("expected_calls", []),
        ))
    return scenarios


def eval_tool_calling(
    model: Any,
    tokenizer: Any,
    scenarios_path: str | Path = "evals/tool_calling.toml",
) -> ToolCallEvalResults:
    """Evaluate a model's tool calling ability across all scenarios."""
    from mlx_lm import generate
    from mlx_lm.generate import make_sampler

    scenarios = load_scenarios(scenarios_path)
    results = ToolCallEvalResults(total=len(scenarios))
    sampler = make_sampler(temp=0.0)

    for scenario in scenarios:
        result = _eval_one(model, tokenizer, scenario, generate, sampler)
        results.results.append(result)

        if scenario.category == "refusal":
            results.refusal_total += 1
            if result.refusal_correct:
                results.refusal_correct_count += 1
        else:
            if result.json_valid:
                results.json_valid_count += 1
            if result.function_correct:
                results.function_correct_count += 1
            if result.params_correct:
                results.params_correct_count += 1

    return results


def _eval_one(
    model: Any,
    tokenizer: Any,
    scenario: ToolCallScenario,
    generate_fn: Any,
    sampler: Any,
) -> ToolCallResult:
    """Evaluate a single tool calling scenario."""
    result = ToolCallResult(
        scenario_id=scenario.id,
        category=scenario.category,
    )

    # Build messages with tool definitions
    messages = [{"role": "user", "content": scenario.user_message}]

    # Format tools for the chat template
    tools_for_template = []
    for tool in scenario.tools:
        tools_for_template.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        })

    # Apply chat template with tools
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                messages,
                tools=tools_for_template,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: manual tool prompt
            formatted = _manual_tool_prompt(scenario)
    except Exception as e:
        logger.debug("Chat template with tools failed for %s: %s", scenario.id, e)
        formatted = _manual_tool_prompt(scenario)

    # Generate
    try:
        output = generate_fn(
            model, tokenizer, prompt=formatted,
            max_tokens=512, sampler=sampler, verbose=False,
        )
        result.raw_output = output
    except Exception as e:
        logger.warning("Generation failed for scenario %s: %s", scenario.id, e)
        return result

    # Evaluate
    is_refusal = scenario.category == "refusal"

    if is_refusal:
        # Should NOT contain a tool call
        has_tool_call = _contains_tool_call(output)
        result.refusal_correct = not has_tool_call
        return result

    # Should contain a tool call — parse it
    parsed = _parse_tool_call(output)
    if parsed is None:
        return result

    func_name, func_args = parsed
    result.json_valid = True
    result.parsed_function = func_name
    result.parsed_args = func_args

    # Check function name
    if scenario.expected_calls:
        expected = scenario.expected_calls[0]  # check first expected call
        expected_func = expected["function"]
        expected_args = expected.get("args", {})

        result.function_correct = func_name == expected_func

        # Check parameters (partial match — expected args must be present)
        if result.function_correct and expected_args:
            result.params_correct = _check_params(func_args, expected_args)
        elif result.function_correct:
            # No specific args required, function match is enough
            result.params_correct = True

    return result


def _manual_tool_prompt(scenario: ToolCallScenario) -> str:
    """Fallback prompt when chat template doesn't support tools."""
    tools_desc = json.dumps(
        [{"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
         for t in scenario.tools],
        indent=2,
    )
    return (
        f"You have access to the following tools:\n{tools_desc}\n\n"
        f"To call a tool, respond with a JSON object: "
        f'{{"name": "tool_name", "arguments": {{...}}}}\n\n'
        f"User: {scenario.user_message}\n\nAssistant:"
    )


def _contains_tool_call(text: str) -> bool:
    """Check if text contains anything that looks like a tool call."""
    # Check common tool call patterns
    patterns = [
        r'"name"\s*:\s*"',
        r'"function"\s*:\s*"',
        r'<tool_call>',
        r'\{"name"',
        r'<\|tool_call\|>',
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Parse a tool call from model output. Handles multiple formats."""
    # Try: {"name": "func", "arguments": {...}}
    # Try: {"function": "func", "arguments": {...}}
    # Try: <tool_call>{"name": ...}</tool_call>
    # Try: ```json {...} ```

    # Strip common wrappers
    text = text.strip()

    # Remove <tool_call> tags
    text = re.sub(r'<\|?/?tool_call\|?>', '', text).strip()

    # Remove markdown code blocks
    md_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if md_match:
        text = md_match.group(1)

    # Find JSON objects in the text
    json_objects = _extract_json_objects(text)

    for obj in json_objects:
        # Format 1: {"name": "func", "arguments": {...}}
        if "name" in obj and isinstance(obj.get("name"), str):
            func_name = obj["name"]
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return func_name, args if isinstance(args, dict) else {}

        # Format 2: {"function": {"name": "func", "arguments": {...}}}
        if "function" in obj and isinstance(obj["function"], dict):
            func = obj["function"]
            return func.get("name", ""), func.get("arguments", {})

        # Format 3: {"type": "function", "function": {"name": ...}}
        if obj.get("type") == "function" and "function" in obj:
            func = obj["function"]
            return func.get("name", ""), func.get("arguments", {})

    return None


def _extract_json_objects(text: str) -> list[dict]:
    """Extract all JSON objects from text."""
    objects = []
    # Find all potential JSON object starts
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Try to parse from this position
            depth = 0
            j = i
            while j < len(text):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[i:j+1])
                            if isinstance(obj, dict):
                                objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        break
                j += 1
        i += 1
    return objects


def _check_params(actual: dict, expected: dict) -> bool:
    """Check if actual params contain all expected params (partial match).

    Handles fuzzy matching for strings (case-insensitive, substring).
    Exact match for numbers and booleans.
    """
    for key, expected_val in expected.items():
        if key not in actual:
            return False

        actual_val = actual[key]

        if isinstance(expected_val, str) and isinstance(actual_val, str):
            # Fuzzy string match: case-insensitive containment
            if expected_val.lower() not in actual_val.lower():
                return False
        elif isinstance(expected_val, list) and isinstance(actual_val, list):
            # Check all expected items are present
            actual_lower = [str(v).lower() for v in actual_val]
            for item in expected_val:
                if str(item).lower() not in actual_lower:
                    return False
        elif isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
            if abs(expected_val - actual_val) > 0.01:
                return False
        elif isinstance(expected_val, bool) and isinstance(actual_val, bool):
            if expected_val != actual_val:
                return False
        # For empty expected args, any value is fine

    return True
