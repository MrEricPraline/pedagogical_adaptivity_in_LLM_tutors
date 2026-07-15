#!/usr/bin/env python3
"""
fix_p0959.py — re-run the one pending case (P0959) in base_proxy_38.json.

P0959 came back unscoreable because the base model put ONE decision point per
activity instead of all five in each. This queries base Qwen3-32B once more for
P0959 only, at temperature 0.0 with the identical prompt PLUS a strengthened
format instruction (all five decisions nested in every activity), retrying up to
six times until a well-formed 5x5 response comes back. It scores it with the same
pai_lib.py, patches base_proxy_38.json in place, and leaves a .bak. Everything
else in the file is untouched. No GPU — a single DeepInfra call.

    # from the folder with pai_lib.py, base_proxy_38.json, case_packets.json, scored_dataset.json
    export DEEPINFRA_API_KEY=...
    python fix_p0959.py --mode rerun --infile base_proxy_38.json \
        --packets case_packets.json --scored scored_dataset.json
"""
import argparse, json, os, shutil, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pai_lib import build_messages, score_response, diversity, parse_json_response, DPS

BASE_MODEL = "Qwen/Qwen3-32B"
BASE_URL = "https://api.deepinfra.com/v1/openai"

FORMAT_REMINDER = (
    "\n\nIMPORTANT FORMAT: Return exactly 5 activities. In EVERY activity you MUST "
    "include ALL FIVE decisions as nested objects, with these exact keys: "
    "content_level, student_task, tutor_role, student_engagement, disciplinary_method. "
    'Each decision must be an object {"selection": "a|b|c|d", "description": "..."}. '
    "Do NOT put only one decision per activity."
)


def _normalize(resp):
    """Scoring-only view {'activities':[{dp:{'selection':x}}]}, tolerant of the base
    model's flat schema ({'content_level':'a'}) and the nested one."""
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
            if sel is None or str(sel).strip().lower() not in ("a", "b", "c", "d"):
                na = None
                break
            na[dp] = {"selection": str(sel).strip().lower()}
        if na:
            out.append(na)
    return {"activities": out}


def _well_formed(resp):
    """True iff exactly 5 activities, each with all 5 valid DP selections."""
    return len(_normalize(resp)["activities"]) == 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="rerun", choices=["rerun"])
    ap.add_argument("--infile", default="base_proxy_38.json")
    ap.add_argument("--packets", default="case_packets.json")
    ap.add_argument("--scored", default="scored_dataset.json")
    ap.add_argument("--case-id", default="P0959")
    ap.add_argument("--out", default=None, help="default: overwrite --infile")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-retries", type=int, default=6)
    args = ap.parse_args()
    out = args.out or args.infile
    cid = args.case_id

    data = json.load(open(args.infile))
    packets = {c["case_id"]: c for c in json.load(open(args.packets))}
    scored = {c["prompt_id"]: c for c in json.load(open(args.scored))}
    if cid not in packets:
        sys.exit(f"{cid} not found in {args.packets}")
    pkt = packets[cid]
    case = scored[cid]  # carries the 5 condition variables used for scoring

    before = next((r for r in data["results"] if r.get("case_id") == cid), None)
    print(f"{cid} before: status={before.get('status') if before else 'MISSING'} "
          f"PAI={before.get('PAI') if before else None}")

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPINFRA_API_KEY"], base_url=BASE_URL)

    msgs = build_messages(pkt["narrative"])
    msgs[-1]["content"] += FORMAT_REMINDER  # identical prompt + strengthened format note

    resp = None
    for attempt in range(1, args.max_retries + 1):
        try:
            r = client.chat.completions.create(
                model=BASE_MODEL, messages=msgs,
                temperature=args.temperature, max_tokens=args.max_tokens,
                response_format={"type": "json_object"},
            )
            cand = parse_json_response(r.choices[0].message.content)
            if _well_formed(cand):
                resp = cand
                print(f"  attempt {attempt}/{args.max_retries}: well-formed 5x5 OK")
                break
            print(f"  attempt {attempt}/{args.max_retries}: malformed, retrying...")
        except Exception as e:
            print(f"  attempt {attempt}/{args.max_retries}: error {e}")
            time.sleep(2 ** min(attempt, 4))
    if resp is None:
        sys.exit(f"FAILED: no well-formed 5x5 response for {cid} after {args.max_retries} tries")

    view = _normalize(resp)
    pai, dp_means, act = score_response(view, case)
    rr, se, mono = diversity(view)
    new_rec = {
        "case_id": cid, "case_type": pkt["case_type"], "source": "rerun", "status": "ok",
        "PAI": pai, "dp_means": dp_means, "activity_scores": act,
        "repetition_rate": rr, "selection_entropy": se, "fully_monotone": mono,
        "response": resp,
    }

    shutil.copyfile(args.infile, args.infile + ".bak")  # backup before patching
    replaced = False
    for i, rec in enumerate(data["results"]):
        if rec.get("case_id") == cid:
            data["results"][i] = new_rec
            replaced = True
            break
    if not replaced:
        data["results"].append(new_rec)
    json.dump(data, open(out, "w"))
    print(f"{cid} after:  status=ok PAI={pai}  (backup: {args.infile}.bak)")
    print(f"patched {out} — the other {len(data['results']) - 1} cases untouched")


if __name__ == "__main__":
    main()
