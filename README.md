# multi-turn-persona-drift

This project was completed as a part of BlueDot Impact's Technical AI Safety Sprint.

## Quick setup guide

1. **Install `uv`**: https://docs.astral.sh/uv/getting-started/installation/

2. **Sync dependencies.** 
   ```bash
   uv sync
   ```

3. **Authenticate with Hugging Face.** 
   ```bash
   uv run hf auth login
   ```

4. **Pipeline** 
   ```bash
   cd persona_drift
   uv run python main_pipeline.py --prompts prompts.jsonl --n-turns 10
   ```
   Flags: `--prompts <path>` (default `prompts.jsonl`), `--out <dir>`
   (default `outputs`), `--n-turns <int>`, `--seed <int>`, `--model <str>`, `--cap-activations`.