# Structured Output for LLM Evaluators

## Problem

`LLMEvaluator.evaluate` currently calls `get_completion` and parses free text:

```python
response = await self.llm.get_completion(messages)
data = parse_json(response.get("text", ""))
if data is not None:
    score = _snap_score(float(data.get("score", 0.5)))
    reasoning = data.get("reasoning", "") or response_text
else:
    numbers = re.findall(r"\b0?\.\d+\b|\b1(?:\.0+)?\b|\b0\b", response_text)
    score = _snap_score(float(numbers[0])) if numbers else 0.5
    reasoning = response_text
```

The regex fallback path exists because the LLM can ignore the system message format
instruction. Structured output eliminates both the fallback and `parse_json`.

## Approach: Tool Calling

Use the `tools` parameter already supported by `SimpleLLMClient.get_completion`.
Force a single tool call with the score schema — the provider enforces the schema,
so the response arguments are always valid JSON.

Preferred over `response_format={"type": "json_schema"}` because tool calling works
across all four providers (OpenAI, Groq, Bedrock, Ollama*).

## Changes

### 1. `llm.py` — add `get_structured_completion` to `SimpleLLMClient`

```python
async def get_structured_completion(
    self,
    messages: List[Dict[str, Any]],
    response_schema: Dict[str, Any],
    tool_name: str = "record_result",
) -> Dict[str, Any]:
    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": "Record the evaluation result",
            "parameters": response_schema,
        },
    }
    response = await self.get_completion(messages, tools=[tool])
    tool_calls = response.get("tool_calls") or []
    if tool_calls:
        data = json.loads(tool_calls[0]["function"]["arguments"])
        return {**response, "structured": data}
    return {**response, "structured": None}
```

Default implementation wraps `get_completion`. Subclasses can override for providers
with native structured output (e.g. OpenAI `response_format`).

### 2. `evaluator.py` — schema constant + updated `LLMEvaluator.evaluate`

Replace the `parse_json` + regex block with a single structured path:

```python
_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "number", "enum": [0.0, 0.5, 1.0]},
    },
    "required": ["reasoning", "score"],
}
```

```python
async def evaluate(self, response_text: str) -> Dict[str, Any]:
    if not response_text.strip():
        return self._empty_result(score=0.0, reasoning="Empty response text provided")

    messages = [
        {"role": "system", "content": "You are an evaluation assistant."},
        {"role": "user", "content": self.build_prompt(response_text)},
    ]

    response = await self.llm.get_structured_completion(messages, _EVAL_SCHEMA)

    if "error" in response.get("metadata", {}):
        raise RuntimeError(f"LLM error: {response['metadata']['error']}")

    data = response.get("structured")
    if data is not None:
        return {
            "score": _snap_score(float(data["score"])),
            "reasoning": data.get("reasoning", ""),
            "elapsed_ms": response.get("elapsed_ms", 0),
            "token_count": response.get("token_count", 0),
        }

    # Fallback: provider/model does not support tool calling
    raw = response.get("text", "")
    parsed = parse_json(raw)
    if parsed is not None:
        return {
            "score": _snap_score(float(parsed.get("score", 0.5))),
            "reasoning": parsed.get("reasoning", "") or raw,
            "elapsed_ms": response.get("elapsed_ms", 0),
            "token_count": response.get("token_count", 0),
        }
    return self._empty_result(score=0.5, reasoning=raw)
```

`build_prompt` no longer needs any JSON format instruction — the schema constraint
is passed out-of-band via the tool definition.

### 3. `build_prompt` — remove system message JSON spec

The system message `'Respond with JSON only: {"score": ...}'` can be dropped.
The format is enforced by the tool schema, not by prompt instruction.

## What stays

- `parse_json` — kept as fallback utility
- `_snap_score` — still applied even on structured responses (defensive against
  providers that return 0.3 despite the enum constraint)
- `build_prompt` interface — no changes to subclasses beyond removing the trailing
  format instruction

## Provider notes

| Provider | Tool calling | Native structured output |
|----------|-------------|--------------------------|
| OpenAI   | Yes         | Yes (`response_format`)  |
| Groq     | Yes         | Partial                  |
| Bedrock  | Yes         | Model-dependent          |
| Ollama   | Model-dependent | No                  |

Ollama models that don't support tool calling will hit the `parse_json` fallback.
