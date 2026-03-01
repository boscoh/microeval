# microeval

A lightweight evaluation framework for LLM testing. Supports local models (Ollama) and cloud providers (OpenAI, AWS Bedrock, Groq). Run evaluations via CLI or web UI, compare models and prompts, and track results.

## Quick Start

### 1. Configure API Keys

Create a `.env` file with your API keys:

```bash
# OpenAI
OPENAI_API_KEY=your-api-key-here

# Groq
GROQ_API_KEY=your-api-key-here

# AWS Bedrock (option 1: use a profile)
AWS_PROFILE=your-profile-name

# AWS Bedrock (option 2: use credentials directly)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=us-east-1
```

For local models, install [Ollama](https://ollama.ai/) and run:
```bash
ollama pull llama3.2
ollama serve
```

### 2. Run a Demo

```bash
uv run microeval demo1
```

This creates a `summary-evals` directory with example evaluations and opens the web UI at http://localhost:8000. Top-level `*-evals` directories are gitignored so your eval data stays local.

---

## Tutorial: Building Your First Evaluation

### Step 1: Create Your Evaluation Directory

```bash
mkdir -p my-evals/{prompts,queries,runs,results}
```

This creates:
```
my-evals/
├── prompts/    # System prompts (instructions for the LLM)
├── queries/    # Test cases (input/output pairs)
├── runs/       # Run configurations (which model, prompt, query to use)
├── results/    # Generated results (created automatically)
├── eval.yaml   # Optional: Global eval service configuration
└── models.yaml # Optional: Override model definitions
```

### Step 2: Write a System Prompt

Create `my-evals/prompts/summarizer.txt`:

```
You are a helpful assistant that summarizes text concisely.

## Instructions
- Summarize the given text in 2-3 sentences
- Capture the key points and main ideas
- Use clear, simple language

## Output Format
Return only the summary, no preamble or explanation.
```

The filename (without extension) becomes the `prompt_ref`.

### Step 3: Create a Query (Test Case)

Create `my-evals/queries/pangram.yaml`:

```yaml
---
input: >-
  The quick brown fox jumps over the lazy dog. This sentence is famous
  because it contains every letter of the English alphabet at least once.
  It has been used for centuries to test typewriters, fonts, and keyboards.
  The phrase was first used in the late 1800s and remains popular today
  for testing purposes.
output: >-
  The sentence "The quick brown fox jumps over the lazy dog" is a pangram
  containing every letter of the alphabet. It has been used since the late
  1800s to test typewriters, fonts, and keyboards.
```

- `input` - The text sent to the LLM (user message)
- `output` - The expected/ideal response (used by evaluators like `equivalence`)

The filename (without extension) becomes the `query_ref`.

### Step 4: Create a Run Configuration

Create `my-evals/runs/summarize-gpt4o.yaml`:

```yaml
---
query_ref: pangram
prompt_ref: summarizer
chat_service: openai
model: gpt-4o
repeat: 3
temperature: 0.5
evaluators:
- word_count
- equivalence
- relevance_llm
- relevance_embedding
```

| Field              | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `query_ref`        | Name of the query file (without `.yaml`)                     |
| `prompt_ref`       | Name of the prompt file (without `.txt`)                     |
| `chat_service`     | LLM provider: `openai`, `bedrock`, `ollama`, or `groq`       |
| `model`            | Model name (e.g., `gpt-4o`, `llama3.2`)                      |
| `repeat`           | Number of times to run the evaluation                        |
| `temperature`      | Sampling temperature (0.0 = deterministic)                   |
| `evaluators`       | List of evaluators to run                                    |
| `eval_chat_service`| Optional: Different service for evaluators (if not set, uses `chat_service`) |
| `eval_chat_model`  | Optional: Different model for evaluators (if not set, uses `model`) |
| `eval_embed_service`| Optional: Service for embedding-based evaluators (if not set, uses `chat_service` or falls back to embedding models from `models.yaml`) |
| `eval_embed_model` | Optional: Model for embedding-based evaluators (if not set, uses default from `models.yaml`) |

### Step 5: Run the Evaluation

**Web UI:**
```bash
uv run microeval ui my-evals
```

Navigate to http://localhost:8000, go to the **Runs** tab, and click the run button.

**CLI:**
```bash
uv run microeval run my-evals
```

### Step 6: View Results

Results are saved to `my-evals/results/` as YAML files:

```yaml
---
texts:
- "The sentence 'The quick brown fox...' is notable for..."
- "The phrase 'The quick brown fox...' contains every letter..."
- "The quick brown fox jumps over the lazy dog is a famous..."
evaluations:
- name: word_count
  values: [1.0, 1.0, 1.0]
  average: 1.0
  standard_deviation: 0.0
- name: equivalence
  values: [0.88, 0.91, 0.85]
  average: 0.88
  standard_deviation: 0.03
- name: relevance_llm
  values: [0.95, 0.92, 0.98]
  average: 0.95
  standard_deviation: 0.03
- name: relevance_embedding
  values: [0.87, 0.89, 0.85]
  average: 0.87
  standard_deviation: 0.02
eval:
  eval_chat_service: openai
  eval_chat_model: gpt-4o-mini
  eval_embed_service: openai
  eval_embed_model: text-embedding-3-small
```

Use the **Graph** tab in the Web UI to visualize and compare results across different runs.

---

## Evaluators

Evaluators score responses on a 0.0-1.0 scale:

| Evaluator            | Description                       | How it Works                               |
|----------------------|-----------------------------------|--------------------------------------------|
| `equivalence`        | Semantic similarity to expected   | LLM compares meaning with query output     |
| `relevance_llm`      | Relevance to the question         | LLM evaluates how relevant the response is to the input question |
| `relevance_embedding` | Relevance using embeddings        | Cosine similarity of embeddings between question and response |
| `word_count`         | Response length validation        | Algorithmic check (no LLM call)            |

### Word Count Configuration

Add these optional fields to your run config:

```yaml
min_words: 50
max_words: 200
target_words: 100
```

### Creating Custom Evaluators

1. Create a class in `microeval/evaluator.py` using the `@register_evaluator` decorator:

```python
@register_evaluator("mycustom")
class MyCustomEvaluator(BaseEvaluator):
    """My custom evaluator with optional parameters."""
    
    async def evaluate(self, response_text: str) -> Dict[str, Any]:
        threshold = self.params.get("threshold", 0.5)
        score = 1.0 if len(response_text) > 100 else 0.5
        return self._empty_result(score=score, reasoning="Custom evaluation")
```

For LLM-based evaluators, extend `LLMEvaluator` instead:

```python
@register_evaluator("custom_llm")
class CustomLLMEvaluator(LLMEvaluator):
    def build_prompt(self, response_text: str) -> str:
        return f"""
            Evaluate the response: {response_text}
            
            Respond with JSON: {{"score": <0.0-1.0>, "reasoning": "<explanation>"}}
        """
```

2. Use in your run config (simple form):
```yaml
evaluators:
- coherence
- mycustom
```

3. Or with parameters:
```yaml
evaluators:
- coherence
- name: word_count
  params:
    min_words: 100
    max_words: 500
- name: mycustom
  params:
    threshold: 0.7
```

---

## Service Configuration

### Basic Configuration

For most use cases, you only need `chat_service` and `model`:

```yaml
chat_service: openai
model: gpt-4o
```

### Advanced: Separate Services for Evaluation

You can use different services/models for running evaluations vs. generating responses:

```yaml
chat_service: bedrock              # Service for generating responses
model: amazon.nova-pro-v1:0
eval_chat_service: openai          # Service for LLM-based evaluators (equivalence, relevance_llm)
eval_chat_model: gpt-4o-mini       # Model for evaluators (cheaper/faster)
```

### Embedding Service Configuration

For embedding-based evaluators (like `relevance_embedding`), you can specify a separate embedding service:

```yaml
chat_service: bedrock
model: amazon.nova-pro-v1:0
eval_embed_service: openai          # Service for embedding-based evaluators
eval_embed_model: text-embedding-3-small
```

If not specified, the system will:
1. Use `eval_embed_service`/`eval_embed_model` if set
2. Check `models.yaml` for embedding models matching your `chat_service` (e.g., `amazon.titan-embed-text-v2:0` for Bedrock)
3. Fall back to OpenAI's `text-embedding-3-small`

**Note:** Bedrock chat models (like `amazon.nova-pro-v1:0`) don't support embeddings, so the system automatically uses Bedrock embedding models from `models.yaml` when Bedrock is your chat service.

---

## Configuration Priority (Hybrid Approach)

Eval service configuration follows this priority order (highest to lowest):

1. **Per-run config** - Explicit settings in individual run YAML files
2. **Environment variables** - Runtime overrides via `EVAL_*` env vars
3. **Global eval.yaml** - Module-level defaults (optional)
4. **Smart defaults** - Automatic fallback logic (see Service Configuration section)

### Option 1: Per-Run Configuration (Most Explicit)

Add eval services directly in each run config:

```yaml
# my-evals/runs/summarize-bedrock.yaml
chat_service: bedrock
model: amazon.nova-pro-v1:0
eval_chat_service: openai          # Override for this run
eval_chat_model: gpt-4o-mini
```

### Option 2: Environment Variables (Runtime Override)

Eval config keys and env vars are aligned:

| Config key (run YAML / eval.yaml) | Environment variable   |
|----------------------------------|------------------------|
| `eval_chat_service`              | `EVAL_CHAT_SERVICE`    |
| `eval_chat_model`                | `EVAL_CHAT_MODEL`      |
| `eval_embed_service`             | `EVAL_EMBED_SERVICE`   |
| `eval_embed_model`               | `EVAL_EMBED_MODEL`     |

```bash
export EVAL_CHAT_SERVICE=openai
export EVAL_CHAT_MODEL=gpt-4o-mini
export EVAL_EMBED_SERVICE=openai
export EVAL_EMBED_MODEL=text-embedding-3-small

microeval run my-evals
```

Environment variables override global config but can be overridden by per-run configs.

### Option 3: Global eval.yaml (Module Defaults)

Create an `eval.yaml` file at the root of your evaluation directory:

```yaml
# Global configuration for all runs
eval_chat_service: openai
eval_chat_model: gpt-4o-mini
eval_embed_service: openai
eval_embed_model: text-embedding-3-small
```

This applies to all runs unless overridden by per-run configs or environment variables.

### Option 4: Smart Defaults (Zero Config)

If no configuration is provided, the system automatically:
- Uses embedding models from `models.yaml` matching your `chat_service`
- Falls back to OpenAI for embeddings if your chat service doesn't support them
- Uses the same service/model for evaluators as for responses

**Example:** If `chat_service: bedrock`, it automatically uses `amazon.titan-embed-text-v2:0` for embeddings.

---

## Comparing Models and Prompts

### Compare Multiple Models

Create multiple run configs with the same query and prompt but different models:

```
my-evals/runs/
├── summarize-gpt4o.yaml      # chat_service: openai, model: gpt-4o
├── summarize-claude.yaml     # chat_service: bedrock, model: amazon.nova-pro-v1:0
├── summarize-llama.yaml      # chat_service: ollama, model: llama3.2
└── summarize-groq.yaml      # chat_service: groq, model: llama-3.3-70b-versatile
```

Run all:
```bash
uv run microeval run my-evals
```

Compare results in the Graph view.

### Compare Multiple Prompts

Create different prompts and run configs:

```
my-evals/prompts/
├── summarizer-basic.txt
├── summarizer-detailed.txt
└── summarizer-expert.txt

my-evals/runs/
├── test-basic.yaml           # prompt_ref: summarizer-basic
├── test-detailed.yaml        # prompt_ref: summarizer-detailed
└── test-expert.yaml          # prompt_ref: summarizer-expert
```

---

## CLI Commands

```bash
microeval                             # Show help
microeval ui BASE_DIR                 # Start web UI for evals directory
microeval run BASE_DIR                # Run all evaluations in directory
microeval demo1                       # Create summary-evals and launch UI
microeval chat SERVICE                # Interactive chat with LLM provider
```

### ui - Web Interface

```bash
microeval ui my-evals                 # Start UI on default port 8000
microeval ui my-evals --port 3000     # Use custom port
microeval ui my-evals --reload        # Enable auto-reload for development
```

### run - CLI Evaluation Runner

```bash
microeval run my-evals                # Run all configs in my-evals/runs/*.yaml
```

Runs all evaluation configs and saves results to `my-evals/results/`.

### demo1 - Quick Start Demo

```bash
microeval demo1                       # Summary evaluation demo
microeval demo1 --base-dir custom     # Use custom directory name
microeval demo1 --port 3000           # Use custom port
```

### chat - Interactive Chat

Test LLM providers directly:
```bash
microeval chat openai
microeval chat ollama
microeval chat bedrock
microeval chat groq
```

---

## Project Structure

```
.
├── .env                             # API keys (see .env.example)
├── microeval/
│   ├── cli.py                       # CLI entry point
│   ├── server.py                    # Web server and API
│   ├── runner.py                    # Evaluation runner
│   ├── evaluator.py                 # Evaluation logic
│   ├── llm.py                       # LLM provider clients
│   ├── chat.py                      # Interactive chat
│   ├── schemas.py                   # Pydantic models
│   ├── logger.py                    # Logging setup
│   ├── index.html                   # Web UI
│   ├── graph.py                     # Metrics visualization
│   ├── utils.py                     # YAML helpers
│   ├── config.py                    # Client configuration
│   ├── models.yaml                  # Default model definitions
│   └── summary-evals/               # Demo template (copied by demo1)
└── *-evals/                         # Eval dirs at repo root (e.g. my-evals, summary-evals); gitignored via /*-evals/
    ├── prompts/                     # System prompts (.txt files)
    ├── queries/                     # Test cases (.yaml files)
    ├── runs/                        # Run configs (.yaml files)
    ├── results/                     # Generated results (.yaml files)
    ├── eval.yaml                    # Optional: Global eval service config
    └── models.yaml                  # Optional: Override model definitions
```

## Services and Models

Default models are configured in `microeval/models.yaml`. You can override them by creating a `models.yaml` file in your evaluation directory.

| Service  | Default Chat Model         | Default Embed Model                    |
|----------|----------------------------|----------------------------------------|
| openai   | gpt-4o                     | text-embedding-3-small                 |
| bedrock  | amazon.nova-pro-v1:0      | amazon.titan-embed-text-v2:0          |
| ollama   | llama3.2                   | nomic-embed-text                       |
| groq     | llama-3.3-70b-versatile    | (no embeddings - falls back to OpenAI) |

**Note:** Groq doesn't support embeddings, so `relevance_embedding` evaluator will automatically use OpenAI's embedding model when Groq is your chat service.

---

## Tips

### Prompt Engineering
- Start with simple prompts and iterate
- Use clear section headers (## Instructions, ## Output Format)
- Specify output format explicitly
- Test with `temperature: 0.0` first for deterministic results

### Evaluation Design
- Use `repeat: 3` or higher to account for model variability
- Include `equivalence` when you have a known-good answer
- Use `relevance_llm` or `relevance_embedding` to measure how well responses address the question
- `relevance_embedding` is faster and cheaper (uses embeddings), while `relevance_llm` provides more nuanced evaluation
- Create multiple query files to test different scenarios

### Eval Service Configuration
- Create `eval.yaml` in your eval directory to set default eval services for all runs
- Use environment variables (`EVAL_CHAT_SERVICE`, etc.) for CI/CD or different environments
- Per-run configs can override global defaults
- Results include `eval_chat_service`, `eval_chat_model`, `eval_embed_service`, and `eval_embed_model` to show which services and models were used

### Comparing Results
- Keep one variable constant when comparing (e.g., same prompt, different models)
- Use the Graph tab to visualize trends
- Check standard deviation to understand consistency
