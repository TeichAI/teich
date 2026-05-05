# Teich - Roadmap

## Current Status: Functional Experimental Baseline
- Docker runtime with Codex, uv, npm
- Configuration system with YAML models and MCP server definitions
- `python -m teich` and CLI init/generate commands
- Raw session extraction from mounted `CODEX_HOME/sessions`
- Auto-generated trace-folder README
- Importable converter for raw Codex traces to training-style messages/tools
- Focused unit coverage for config, CLI, and runner behavior

---

## Phase 1: Testing & Hardening (Next)

### 1.1 Integration Testing
- [x] Test actual Docker image build
- [ ] Validate live `codex exec` end-to-end against installed Codex CLI versions
- [ ] Verify session files are extracted correctly with real sessions
- [ ] Verify trace format matches HF expectations on actual generated traces
- [ ] Test MCP server configuration with real servers
- [ ] Validate LM Studio / Ollama local-provider runs through Codex OSS mode

### 1.2 Error Handling
- [ ] Handle Docker not installed/running
- [ ] Handle invalid OpenAI API key or unavailable local provider
- [x] Handle network timeouts during Codex execution
- [x] Handle session extraction failures
- [ ] Retry logic for failed prompts

### 1.3 Output Format Validation
- [x] Validate trace JSONL structure against example traces
- [ ] Ensure HF trace viewer compatibility
- [x] Generate README for trace upload directories

---

## Phase 2: Training Data Conversion

### 2.1 Converter Module
Implemented `src/teich/converter.py`:

```python
from teich import load_traces

dataset = load_traces("./output")
example = dataset[0]

rendered = tokenizer.apply_chat_template(
    example["messages"],
    tools=example.get("tools") or [],
    tokenize=False,
    add_generation_prompt=False,
)
```

### 2.2 Supported Formats
- [x] **OpenAI-style chat/message format** (primary internal output):
  ```json
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "...", "reasoning_content": "..."},
      {"role": "assistant", "tool_calls": [...]},
      {"role": "tool", "tool_call_id": "...", "content": "..."}
    ]
  }
```
- [ ] **Anthropic Messages API**
- [ ] **Gemini Format**

### 2.3 Field Mapping
From Codex traces to training examples:
- [x] Extract system prompts from session init / developer messages
- [x] Map `message` events → user/assistant messages
- [x] Extract `reasoning_content` from reasoning summary events
- [x] Map `function_call` → assistant message with tool_calls
- [x] Map `function_call_output` → tool message
- [ ] Extract tool schemas when present in raw traces
- [x] Handle multi-turn conversations correctly

---

## Phase 3: Advanced Features

### 3.1 Parallel Execution
- [ ] Run multiple prompts concurrently
- [ ] Configurable max_workers
- [ ] Progress tracking for batch jobs

### 3.2 Session Resumption
- [ ] Save progress checkpoint
- [ ] Resume interrupted runs
- [ ] Skip already-completed prompts

### 3.3 Output Formats
- [ ] Hugging Face datasets integration
- [ ] Parquet output option
- [ ] Train/validation split generation

### 3.4 Quality Filtering
- [ ] Filter empty/short sessions
- [ ] Detect failed/error sessions
- [ ] Workspace artifact validation
- [ ] Configurable quality thresholds

---

## Phase 4: Extended Model Support

### 4.1 OpenRouter/OpenAI-Compatible APIs
Since Codex CLI behavior varies by provider, keep exploring alternatives:
- [ ] Research: Can we use `aider` or similar tools?
- [ ] Custom runner for generic OpenAI-compatible APIs
- [x] OpenRouter/config override path in current Codex runner
- [ ] Harden compatibility for non-OpenAI endpoints under real runs

### 4.2 Multi-Provider Support
- [x] Modular config boundary with `agent.provider`
- [ ] Pi agent runner
- [ ] Anthropic Claude runner
- [ ] Google Gemini runner
- [ ] Ollama/local model runner beyond Codex OSS mode

---

## Phase 5: Production Polish

### 5.1 Documentation
- [ ] Full API documentation
- [ ] Tutorial: Creating your first dataset
- [ ] Tutorial: Fine-tuning with generated data
- [ ] Example configs for common use cases

### 5.2 CLI Improvements
- [ ] `validate` command to check config
- [ ] `preview` command to see what would be generated
- [ ] `status` command to check previous runs
- [ ] Rich progress bars and logging

### 5.3 Testing
- [ ] Integration tests with real API calls (mocked)
- [ ] Docker build tests in CI
- [ ] Format validation tests
- [ ] End-to-end workflow tests

---

## Immediate Next Steps

1. **Validate live Codex execution with the installed CLI**:
   ```bash
   cd v2
   python -m teich generate -c test_run/config.yaml
   ```

2. **Verify trace files** are generated and match the example-style raw session format

3. **Inspect an actual generated trace** and tighten converter field mapping if needed

4. **Prototype Pi-agent trace ingestion** behind the same conversion/export interfaces

---

## Open Questions

1. Which Codex CLI versions should `v2` explicitly support for non-interactive runs?
2. Should LM Studio and Ollama stay routed through Codex OSS mode, or get their own non-Codex runner?
3. What quality metrics should we filter on?
4. How should we handle tool schemas that vary between providers?
