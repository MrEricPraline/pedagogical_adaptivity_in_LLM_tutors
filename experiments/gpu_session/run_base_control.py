#!/usr/bin/env python3
"""
STEP 2 — the 8 missing base-proxy control outputs.  NO GPU NEEDED.

Base-proxy outputs already exist for all 30 intervention cases (they sit inside the
589-case training set). The 8 control cases were never queried on the base proxy,
because they were deliberately excluded from training. This script fills that gap:
base Qwen3-32B, no adapter, same prompt, same decoding parameters.

Completing the 38-case base-proxy set is what makes a third (base-proxy) rating
round possible, and it is what lets you rule out "the proxy just writes worse prose"
as an explanation for the fall in expert ratings.

    pip install openai
    export DEEPINFRA_API_KEY=...
    python run_base_control.py --packets case_packets.json -o base_proxy_38.json
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pai_lib import build_messages, score_response, diversity, parse_json_response, summarize, DPS

BASE_MODEL = "Qwen/Qwen3-32B"
BASE_URL = "https://api.deepinfra.com/v1/openai"


def _normalize(resp):
    """Scoring-only view {'activities':[{dp:{'selection':x}}]}, tolerant of both the
    base model's flat schema ({'content_level':'a'}) and the nested one. The original
    `resp` (with descriptions) is kept untouched for the expert-rating text."""
    if isinstance(resp, dict):
        acts = resp.get("activities") or resp.get("sequence") or []
        if not acts and resp and all(str(k).isdigit() for k in resp):
            acts = [resp[k] for k in sorted(resp, key=lambda x: int(x))]
    elif isinstance(resp, list):
        acts = resp
    else:
        acts = []
    out = []
    for a in acts:
        if not isinstance(a, dict):
            continue
        na = {}
        for dp in DPS:
            v = a.get(dp)
            sel = v.get("selection") if isinstance(v, dict) else v
            if sel is None:
                na = None
                break
            na[dp] = {"selection": str(sel).strip().lower()}
        if na:
            out.append(na)
    return {"activities": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", default="case_packets.json")
    ap.add_argument("--scored", default="scored_dataset.json",
                    help="needed for the condition variables used in scoring")
    ap.add_argument("--existing-base", default=None,
                    help="baseline_qwen_train.json — to merge in the 30 intervention cases")
    ap.add_argument("-o", "--out", default="base_proxy_38.json")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"], base_url=BASE_URL)

    packets = json.load(open(args.packets))
    scored = {c["prompt_id"]: c for c in json.load(open(args.scored))}

    # which of the 38 already have a base-proxy output?
    have = {}
    if args.existing_base:
        for r in json.load(open(args.existing_base))["results"]:
            if r.get("post_response"):
                have[r["prompt_id"]] = r["post_response"]
        print(f"reusing {len(have)} existing base-proxy responses")

    results = []
    for c in packets:
        cid = c["case_id"]
        case = scored[cid]  # carries bloom_band / knowledge_state / learning_stage / learning_context / subject_family

        if cid in have:
            resp = have[cid]
            source = "existing"
        else:
            print(f"querying base proxy: {cid} ({c['case_type']})")
            msgs = build_messages(c["narrative"])
            for attempt in range(4):
                try:
                    out = client.chat.completions.create(
                        model=BASE_MODEL, messages=msgs,
                        temperature=args.temperature, max_tokens=args.max_tokens,
                        response_format={"type": "json_object"},
                    )
                    resp = parse_json_response(out.choices[0].message.content)
                    break
                except Exception as e:
                    if attempt == 3:
                        print(f"  FAILED {cid}: {e}")
                        resp = None
                        break
                    time.sleep(2 ** attempt)
            source = "new"
            if resp is None:
                results.append({"case_id": cid, "case_type": c["case_type"],
                                "status": "failed", "PAI": None})
                continue

        scored_view = _normalize(resp)
        try:
            pai, dp_means, act = score_response(scored_view, case)
            rr, se, mono = diversity(scored_view)
        except Exception as e:
            results.append({"case_id": cid, "case_type": c["case_type"], "source": source,
                            "status": f"score_error: {e}", "PAI": None, "response": resp})
            continue
        results.append({
            "case_id": cid, "case_type": c["case_type"], "source": source, "status": "ok",
            "PAI": pai, "dp_means": dp_means, "activity_scores": act,
            "repetition_rate": rr, "selection_entropy": se, "fully_monotone": mono,
            "response": resp,
        })

    for ctype in ("intervention", "control"):
        sub = [r for r in results if r.get("case_type") == ctype and r.get("PAI") is not None]
        s = summarize(sub)
        print(f"\nbase proxy — {ctype} (n={s.get('n')}): PAI={s.get('PAI_mean')} "
              f"repetition={s.get('repetition_mean')} entropy={s.get('entropy_mean')} "
              f"monotone={s.get('pct_fully_monotone')}%")

    json.dump({"base_model": BASE_MODEL, "is_baseline": True, "adapter": None,
               "n_cases": len(results), "results": results}, open(args.out, "w"))
    print(f"\nwrote {args.out}")
    print("→ these 38 base-proxy outputs are the material for a possible third rating round.")


if __name__ == "__main__":
    main()
