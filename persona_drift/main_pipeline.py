# -*- coding: utf-8 -*-
"""Multi-turn persona drift: Gemma-2-27B self-play debate pipeline.

Two agents (sharing one model instance) hold a multi-turn discussion from a
jsonl-defined round spec. Each generated turn's layer-22 activations are
projected into the persona PC space built from Lu et al.'s published role
vectors, and one record per turn is streamed to disk.
"""

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from utils import (
    ActivationCache,
    load_model,
    load_persona_space,
    project_topk,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "google/gemma-2-27b-it"
VECTOR_MODEL = "gemma-2-27b"
VECTOR_REPO = "lu-christina/assistant-axis-vectors"
TARGET_LAYER = 22
N_COMPONENTS = 10
N_TURNS = 10
DO_SAMPLE = False
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 500
SEED = 42
PROMPTS_PATH = "prompts.jsonl"
OUTPUT_DIR = "outputs"

REQUIRED_PROMPT_KEYS = ("id", "topic", "shared_system", "support_system", "oppose_system", "opening")


@dataclass
class GenConfig:
    do_sample: bool
    temperature: float
    max_new_tokens: int


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Prompts I/O
# ---------------------------------------------------------------------------

def load_prompts(path):
    rounds = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                spec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
            missing = [k for k in REQUIRED_PROMPT_KEYS if k not in spec]
            if missing:
                raise ValueError(f"{path}:{line_no}: missing required keys {missing}")
            spec.setdefault("meta", {})
            rounds.append(spec)
    return rounds


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DebateAgent:
    def __init__(self, model, tokenizer, name, system_prompt, persona_space,
                 layer_idx, gen_cfg: GenConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.persona_space = persona_space
        self.layer_idx = layer_idx
        self.gen_cfg = gen_cfg
        self.messages = []
        self.last_n_new_tokens = 0

        self.is_gemma = "gemma" in model.config.model_type.lower()
        if self.is_gemma:
            self._pending_system = system_prompt
        else:
            self._pending_system = None
            self.messages.append({"role": "system", "content": system_prompt})

    def get_layer_activations(self, input_ids, layer_idx=None):
        layer_idx = self.layer_idx if layer_idx is None else layer_idx
        cache = ActivationCache()
        handle = self.model.model.layers[layer_idx].register_forward_hook(cache)
        try:
            with torch.no_grad():
                self.model(input_ids.to(self.model.device))
        finally:
            handle.remove()
        return cache.acts[0]  # [seq_len, d_model] on CPU

    def observe(self, text: str):
        if self._pending_system is not None:
            text = self._pending_system + "\n\n" + text
            self._pending_system = None
        self.messages.append({"role": "user", "content": text})

    def respond(self):
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.gen_cfg.max_new_tokens,
            do_sample=self.gen_cfg.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.gen_cfg.do_sample:
            gen_kwargs["temperature"] = self.gen_cfg.temperature

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = out[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": text})
        self.last_n_new_tokens = int(new_tokens.shape[-1])

        full_acts = self.get_layer_activations(out[0].unsqueeze(0))
        mean_act = full_acts[prompt_len:].float().mean(dim=0)  # [d_model], CPU

        del full_acts, out, inputs, new_tokens
        torch.cuda.empty_cache()

        pc_scores = None
        if self.persona_space is not None:
            pca = self.persona_space["pca"]
            scaler = self.persona_space["scaler"]
            expected_dim = np.asarray(scaler.mean).shape[0]
            if mean_act.shape[0] == expected_dim:
                pc_scores = project_topk(mean_act, pca, scaler, N_COMPONENTS)

        return text, pc_scores


# ---------------------------------------------------------------------------
# Debate loop
# ---------------------------------------------------------------------------

def run_round(round_spec, model, tokenizer, persona_space, n_turns, gen_cfg):
    """Yield one record dict per turn."""
    agent_a = DebateAgent(
        model, tokenizer, name="agent_a",
        system_prompt=round_spec["shared_system"] + "\n\n" + round_spec["support_system"],
        persona_space=persona_space, layer_idx=TARGET_LAYER, gen_cfg=gen_cfg,
    )
    agent_b = DebateAgent(
        model, tokenizer, name="agent_b",
        system_prompt=round_spec["shared_system"] + "\n\n" + round_spec["oppose_system"],
        persona_space=persona_space, layer_idx=TARGET_LAYER, gen_cfg=gen_cfg,
    )

    round_id = round_spec["id"]
    topic = round_spec["topic"]
    meta = round_spec.get("meta", {})

    def make_record(turn, speaker, text, pc_scores, n_new_tokens):
        pc_scores = pc_scores or []
        return {
            "round_id": round_id, "topic": topic, "turn": turn,
            "speaker": speaker, "text": text,
            "n_new_tokens": n_new_tokens,
            "degenerate": len(text.strip()) < 20,
            "pc_scores": pc_scores,
            "pc1_score": pc_scores[0] if len(pc_scores) > 0 else None,
            "pc2_score": pc_scores[1] if len(pc_scores) > 1 else None,
            "meta": meta,
        }

    agent_a.observe(round_spec["opening"])

    for turn in range(n_turns):
        text_a, pc_a = agent_a.respond()
        yield make_record(turn, "agent_a", text_a, pc_a, agent_a.last_n_new_tokens)
        agent_b.observe(text_a)

        text_b, pc_b = agent_b.respond()
        yield make_record(turn, "agent_b", text_b, pc_b, agent_b.last_n_new_tokens)
        agent_a.observe(text_b)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default=PROMPTS_PATH)
    parser.add_argument("--out", default=OUTPUT_DIR)
    parser.add_argument("--n-turns", type=int, default=N_TURNS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def write_csv(records_path, csv_path):
    fields = ["round_id", "topic", "turn", "speaker", "pc1_score", "pc2_score",
              "n_new_tokens", "degenerate", "text"]
    with open(records_path, "r", encoding="utf-8") as fin, \
         open(csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            writer.writerow({k: rec.get(k) for k in fields})


def main():
    args = parse_args()
    seed_everything(args.seed)

    prompts = load_prompts(args.prompts)

    persona_space = load_persona_space(VECTOR_MODEL, TARGET_LAYER, VECTOR_REPO)
    model, tokenizer = load_model(MODEL_NAME)

    gen_cfg = GenConfig(do_sample=DO_SAMPLE, temperature=TEMPERATURE,
                         max_new_tokens=MAX_NEW_TOKENS)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_name": MODEL_NAME,
        "vector_model": VECTOR_MODEL,
        "vector_repo": VECTOR_REPO,
        "target_layer": TARGET_LAYER,
        "n_components": N_COMPONENTS,
        "n_turns": args.n_turns,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": args.seed,
        "prompts_path": str(args.prompts),
        "notes": "",
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    records_path = run_dir / "records.jsonl"
    n_records = 0
    n_degenerate = 0

    with open(records_path, "w", encoding="utf-8") as f:
        for round_spec in prompts:
            print(f"=== Round {round_spec['id']}: {round_spec['topic'][:60]}... ===")
            for record in run_round(round_spec, model, tokenizer, persona_space,
                                     args.n_turns, gen_cfg):
                f.write(json.dumps(record) + "\n")
                f.flush()
                n_records += 1
                if record["degenerate"]:
                    n_degenerate += 1

    write_csv(records_path, run_dir / "records.csv")

    print(f"Done. rounds={len(prompts)} records={n_records} degenerate={n_degenerate}")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
