#!/usr/bin/env python3
"""
STEP 1 — GPU session.

Runs every arm on the 200 out-of-sample cases and scores them on the PAI and the
sequence-level measures. Closes two gaps at once:

  (A) GENERALISATION  — does the collapse appear on cases the model never trained on?
      arms: base (no adapter), greedy r=8

  (B) CAUSAL TEST for RQ3.2 — is the metric gain separable from the collapse?
      arms: greedy_parallel   (greedy projected ONTO the per-DP subspace)
            greedy_orthogonal (the residual — hypothesised collapse component)
      If greedy_parallel raises the PAI while keeping entropy high, the two
      components are causally separable and a better corrective signal is possible.

Requires: one 80GB GPU (H100 / A100-80G).

    pip install vllm torch transformers safetensors
    python run_oos_gpu.py --cases oos_cases.json --adapters ./adapters -o results_oos/

Outputs one json per arm plus a summary table.
"""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pai_lib import build_messages, score_response, diversity, parse_json_response, summarize

BASE_MODEL = "Qwen/Qwen3-32B"

# arm name -> adapter filename (None = base model, no adapter)
ARMS = {
    "base":              None,
    "greedy_r8":         "adapter_r8.safetensors",
    "greedy_parallel":   "adapter_greedy_parallel.safetensors",
    "greedy_orthogonal": "adapter_greedy_orthogonal.safetensors",
}


def run_arm(llm, sampling, cases, lora_request):
    from vllm import SamplingParams  # noqa
    prompts = [build_messages(c["narrative"]) for c in cases]
    outs = llm.chat(prompts, sampling, lora_request=lora_request)
    results = []
    for case, out in zip(cases, outs):
        rec = {"prompt_id": case["prompt_id"]}
        try:
            resp = parse_json_response(out.outputs[0].text)
            pai, dp_means, act = score_response(resp, case)
            rr, se, mono = diversity(resp)
            rec.update(status="ok", PAI=pai, dp_means=dp_means, activity_scores=act,
                       repetition_rate=rr, selection_entropy=se, fully_monotone=mono,
                       response=resp, raw_text=out.outputs[0].text)
        except Exception as e:
            rec.update(status=f"parse_error: {e}", PAI=None,
                       raw_text=out.outputs[0].text if out.outputs else "")
        results.append(rec)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="oos_cases.json")
    ap.add_argument("--adapters", default="./adapters")
    ap.add_argument("-o", "--outdir", default="results_oos")
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="MUST match the original pipeline (greedy decoding)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    cases = json.load(open(args.cases))
    print(f"cases: {len(cases)}")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=args.max_lora_rank,
              dtype="bfloat16", gpu_memory_utilization=0.92, max_model_len=8192)
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    summaries = {}
    for i, (arm, fname) in enumerate(ARMS.items(), start=1):
        if fname is None:
            lora = None
        else:
            path = os.path.join(args.adapters, fname)
            if not os.path.exists(path):
                print(f"!! skipping {arm}: {path} not found")
                continue
            lora = LoRARequest(arm, i, path)

        print(f"\n=== arm: {arm} ===")
        results = run_arm(llm, sampling, cases, lora)
        summ = summarize(results)
        summaries[arm] = summ
        json.dump({"arm": arm, "base_model": BASE_MODEL, "adapter": fname,
                   "n_cases": len(cases), "summary": summ, "results": results},
                  open(os.path.join(args.outdir, f"oos_{arm}.json"), "w"))
        print(f"  PAI={summ.get('PAI_mean')}  repetition={summ.get('repetition_mean')}  "
              f"entropy={summ.get('entropy_mean')}  monotone={summ.get('pct_fully_monotone')}%")

    print("\n" + "=" * 78)
    print(f"{'arm':<20}{'PAI':>9}{'repetition':>12}{'entropy':>10}{'% monotone':>12}")
    print("-" * 78)
    for arm, s in summaries.items():
        print(f"{arm:<20}{s.get('PAI_mean',0):>9.3f}{s.get('repetition_mean',0):>12.3f}"
              f"{s.get('entropy_mean',0):>10.3f}{s.get('pct_fully_monotone',0):>11.1f}%")
    json.dump(summaries, open(os.path.join(args.outdir, "summary.json"), "w"), indent=2)
    print(f"\nwrote {args.outdir}/  — push summary.json + the four oos_*.json to the repo")


if __name__ == "__main__":
    main()
