#!/usr/bin/env python3
"""
AIR-BENCH-2024 Runner
- Model responses: vLLM (OpenAI-compatible API)
- Judge: Anthropic Claude (replacing GPT-4o)
"""

import argparse
import json
import os
import random
import sys
import time

from dotenv import load_dotenv
load_dotenv()

import anthropic
import datasets
from openai import OpenAI


def extract_content(tag, text):
    """Extract content after a tag from judge response."""
    start_idx = text.find(tag)
    if start_idx == -1:
        return ""
    start_of_content = start_idx + len(tag)
    if tag == "##the_score: ":
        end_idx = text.find("\n", start_of_content)
    else:
        end_idx = text.find(".\n##", start_of_content)
    if end_idx == -1:
        content = text[start_of_content:].strip()
    else:
        content = text[start_of_content:end_idx].strip()
    return content


def sample_row(raw_data, sample_num):
    """Sample `sample_num` prompts per L2 category (1-16)."""
    sample_row_list = []
    for l2 in range(1, 17):
        l2_specific_rows = []
        for data in raw_data:
            cate_idx, l2_name, l3_name, l4_name, prompt = data.values()
            l2_index = int(cate_idx.split('.')[0])
            if l2_index == l2:
                l2_specific_rows.append((cate_idx, l2_name, l3_name, l4_name, prompt))
        if len(l2_specific_rows) < sample_num:
            print(f"Warning: L2={l2} has only {len(l2_specific_rows)} rows, sampling all")
            sample_row_list.extend(l2_specific_rows)
        else:
            l2_specific_sampled_row = random.sample(l2_specific_rows, sample_num)
            sample_row_list.extend(l2_specific_sampled_row)
    return sample_row_list


def get_model_response(vllm_client, model_name, system_msg, user_msg):
    """Get a response from the vLLM-served model."""
    response = vllm_client.chat.completions.create(
        temperature=0,
        max_tokens=512,
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    msg = response.choices[0].message
    # Some models (e.g. Qwen3.5 with reasoning parser) put output in reasoning field
    content = msg.content
    if content is None:
        reasoning = getattr(msg, 'reasoning', None) or getattr(msg, 'reasoning_content', None)
        if reasoning:
            content = reasoning
    return content or ""


def judge_with_claude(anthropic_client, judge_prompt, question, answer, judge_model="claude-sonnet-4-20250514"):
    """Use Anthropic Claude as the judge instead of GPT-4o."""
    prompt_filled = judge_prompt.replace("{{QUESTION}}", str(question or "")).replace("{{ANSWER}}", str(answer or ""))
    response = anthropic_client.messages.create(
        model=judge_model,
        max_tokens=512,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt_filled}
        ],
    )
    return response.content[0].text


def main():
    parser = argparse.ArgumentParser(description="AIR-BENCH-2024 Runner (vLLM + Anthropic Judge)")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1",
                        help="vLLM OpenAI-compatible API base URL")
    parser.add_argument("--vllm-model", default="Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
                        help="Model name served by vLLM")
    parser.add_argument("--anthropic-api-key", default=os.getenv("ANTHROPIC_API_KEY", ""),
                        help="Anthropic API key for judge")
    parser.add_argument("--judge-model", default="claude-sonnet-4-20250514",
                        help="Anthropic model to use as judge")
    parser.add_argument("--sample-num", type=int, default=5,
                        help="Number of prompts to sample per L2 category")
    parser.add_argument("--region", default="default",
                        choices=["default", "china", "eu_comprehensive", "eu_mandatory", "us"],
                        help="Dataset region")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--system-prompt", default="You are a helpful assistant.",
                        help="System prompt for the model")
    parser.add_argument("--skip-responses", action="store_true",
                        help="Skip response generation, only run judge (requires existing response file)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model_short_name = args.vllm_model.split("/")[-1]
    response_file = os.path.join(args.output_dir, f"step1_{model_short_name}_response.json")
    result_file = os.path.join(args.output_dir, f"step2_{model_short_name}_result.json")

    # ── Step 1: Generate model responses ──────────────────────────────
    if not args.skip_responses:
        print(f"=== Step 1: Generating responses from {args.vllm_model} ===")
        print(f"    vLLM endpoint: {args.vllm_base_url}")

        vllm_client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't need a real key
            base_url=args.vllm_base_url,
        )

        # Verify vLLM is reachable
        try:
            models = vllm_client.models.list()
            available = [m.id for m in models.data]
            print(f"    Available models: {available}")
            if args.vllm_model not in available:
                print(f"    WARNING: {args.vllm_model} not in available models, using first: {available[0]}")
                args.vllm_model = available[0]
        except Exception as e:
            print(f"    ERROR: Cannot reach vLLM at {args.vllm_base_url}: {e}")
            sys.exit(1)

        print(f"    Loading dataset (region={args.region})...")
        test_data = datasets.load_dataset("stanford-crfm/air-bench-2024", args.region, split="test")
        rows = sample_row(test_data, args.sample_num)
        print(f"    Sampled {len(rows)} prompts ({args.sample_num} per L2 category)")

        output_json = []
        for i, (cate_idx, l2_name, l3_name, l4_name, prompt) in enumerate(rows):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"    [{i+1}/{len(rows)}] Querying model...")
            try:
                resp = get_model_response(vllm_client, args.vllm_model, args.system_prompt, prompt)
            except Exception as e:
                print(f"    ERROR at prompt {i+1}: {e}")
                resp = f"[ERROR: {e}]"

            json_entry = {
                "cate_idx": cate_idx,
                "l2_name": l2_name,
                "l3_name": l3_name,
                "l4_name": l4_name,
                "prompt": [{"prompt": prompt}],
                "response": resp,
            }
            output_json.append(json_entry)

        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=4)
        print(f"    Responses saved to {response_file}")
    else:
        print(f"=== Step 1: Skipping response generation, loading {response_file} ===")
        with open(response_file, 'r', encoding='utf-8') as f:
            output_json = json.load(f)
        print(f"    Loaded {len(output_json)} responses")

    # ── Step 2: Judge with Anthropic Claude ───────────────────────────
    print(f"\n=== Step 2: Judging with {args.judge_model} ===")

    if not args.anthropic_api_key:
        print("    ERROR: No Anthropic API key. Set ANTHROPIC_API_KEY or use --anthropic-api-key")
        sys.exit(1)

    anthropic_client = anthropic.Anthropic(api_key=args.anthropic_api_key)

    print("    Loading judge prompts...")
    judge_prompt_list = datasets.load_dataset("stanford-crfm/air-bench-2024", "judge_prompts", split="test")
    judge_prompt_map = {item['cate-idx']: item['judge_prompt'] for item in judge_prompt_list}

    total = len(output_json)
    scores_all = []

    for i, entry in enumerate(output_json):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1}/{total}] Judging...")

        cate_idx = entry['cate_idx']
        question = entry['prompt'][0]['prompt']
        answer = entry['response']

        judge_prompt = judge_prompt_map.get(cate_idx)
        if not judge_prompt:
            # Try matching on L3 level (first two parts of cate_idx)
            l3_key = '.'.join(cate_idx.split('.')[:2])
            judge_prompt = judge_prompt_map.get(l3_key)
        if not judge_prompt:
            print(f"    WARNING: No judge prompt for {cate_idx}, skipping")
            entry['reason'] = "No judge prompt available"
            entry['score'] = -1
            continue

        retries = 3
        for attempt in range(retries):
            try:
                judge_response = judge_with_claude(
                    anthropic_client, judge_prompt, question, answer, args.judge_model
                )
                reason = extract_content("##short_reasoning: ", judge_response)
                score_str = extract_content("##the_score: ", judge_response)
                score = float(score_str) if score_str else -1
                entry['reason'] = reason
                entry['score'] = score
                if score >= 0:
                    scores_all.append(score)
                break
            except Exception as e:
                print(f"    Judge error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    entry['reason'] = f"Judge error: {e}"
                    entry['score'] = -1

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n=== Results Summary ===")
    print(f"    Results saved to {result_file}")
    if scores_all:
        avg_score = sum(scores_all) / len(scores_all)
        safe_count = sum(1 for s in scores_all if s == 1.0)
        ambig_count = sum(1 for s in scores_all if s == 0.5)
        harmful_count = sum(1 for s in scores_all if s == 0.0)
        print(f"    Total judged: {len(scores_all)}")
        print(f"    Average AIR score: {avg_score:.4f}")
        print(f"    Safe (1.0):     {safe_count} ({100*safe_count/len(scores_all):.1f}%)")
        print(f"    Ambiguous (0.5): {ambig_count} ({100*ambig_count/len(scores_all):.1f}%)")
        print(f"    Harmful (0.0):  {harmful_count} ({100*harmful_count/len(scores_all):.1f}%)")

        # Per-category breakdown
        cat_scores = {}
        for entry in output_json:
            l2 = entry['l2_name']
            if entry['score'] >= 0:
                cat_scores.setdefault(l2, []).append(entry['score'])
        print(f"\n    Per-category scores:")
        for cat, scores in sorted(cat_scores.items()):
            cat_avg = sum(scores) / len(scores)
            print(f"      {cat}: {cat_avg:.4f} (n={len(scores)})")
    else:
        print("    No valid scores collected.")


if __name__ == "__main__":
    main()
