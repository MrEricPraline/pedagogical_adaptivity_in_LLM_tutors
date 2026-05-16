# Pedagogical Adaptivity in LLM Tutors — Research Pipeline

Research pipeline for generating and analysing case narratives that explore how LLM tutors should adapt their pedagogical approach based on learner variables. The pipeline produces a fully-crossed factorial design and then uses xAI/Grok to generate naturalistic learning scenarios for each condition.

## Stages

| Stage | Description | Output |
|-------|-------------|--------|
| **Stage 1** — Factorial sampling | Generates the full-factorial crossing of five experimental variables (Bloom level, knowledge state, learning stage, learning context, subject). | `data/stage1/factorial_sample.csv` (2 160 rows) |
| **Stage 2** — Case narrative generation | For each row, calls Grok via the xAI API to produce a realistic 80–120 word learning scenario where the experimental variables are embedded implicitly. | `data/stage2/case_narratives.jsonl` |
| **Stage 3** — Final case preparation | Deduplication, regeneration and validation pass that produces the curated input set for the target-model query stage. | `data/stage2/cases_final.jsonl` |
| **Stage 4** — Target model querying | Sends each curated case to **Gemini 3.1 Pro Preview** and asks for 5 learning activities, each annotated along 5 pedagogical dimensions (`content_level`, `student_task`, `tutor_role`, `student_engagement`, `disciplinary_method`). Uses structured JSON output. | `data/stage4/gemini_responses.jsonl` |
| **Stage 5** — PAI scoring + corrective LoRA | Applies the Pedagogical Adaptivity Index matrices to every Stage 4 response, builds a corrective LoRA training set from the lowest-PAI cases, fine-tunes Qwen3-32B on Tinker at ranks r∈{1,4,8,16}, re-queries the corrected model, and runs the cross-dimensional interference analysis. | `data/stage5/scored_dataset.json`, `data/stage5/corrective_training_data.json`, `data/stage5/post_intervention_r{rank}.json`, `data/stage5/interference_analysis.json` |

## Project structure

```
.
├── README.md
├── .env.example
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── pipeline/
│   │   ├── cli.py              # CLI entry-point (argparse)
│   │   ├── config.py           # Centralised configuration
│   │   ├── logging_utils.py    # Logging setup
│   │   └── manifests.py        # Manifest generation
│   ├── common/
│   │   ├── io_utils.py         # CSV / JSONL / JSON read/write
│   │   ├── schemas.py          # Variable definitions & data classes
│   │   └── text_utils.py       # Word counting, forbidden-term detection
│   ├── stage1_factorial/
│   │   ├── generator.py        # Full-factorial crossing
│   │   ├── exporter.py         # CSV + JSONL export
│   │   └── pipeline.py         # Stage 1 orchestrator
│   ├── stage2_generation/
│   │   ├── prompt_builder.py   # Variable → natural-language translation
│   │   ├── provider_xai.py     # xAI/Grok client (retry + rate limit)
│   │   ├── validator.py        # Narrative validation checks
│   │   ├── checkpoint.py       # Resume support
│   │   └── pipeline.py         # Stage 2 orchestrator
│   ├── stage4_query/
│   │   ├── prompt_builder.py   # Stage 4 system instruction + JSON schema
│   │   ├── provider_gemini.py  # Gemini 3.1 Pro Preview client
│   │   ├── validator.py        # Local validation of structured responses
│   │   ├── checkpoint.py       # Resume support (skip prompt_ids with status=ok)
│   │   └── pipeline.py         # Stage 4 orchestrator
│   ├── stage5_scoring/
│   │   ├── matrices.py         # 11 PAI sub-matrices (DP1–DP5) + DECISION_POINTS registry
│   │   ├── scorer.py           # score_response(), score_activity(), find_optimal_selections()
│   │   └── pipeline.py         # Stage 5 scoring orchestrator
│   ├── stage5_finetune/
│   │   ├── prompt_builder.py     # Diversified description variants for the corrective targets
│   │   ├── weak_cells.py         # Identify (DP × condition) cells with lowest PAI
│   │   ├── corrective_data.py    # Build corrective sets: global, stratified, per-DP
│   │   ├── eval_split.py         # Build held-out eval set disjoint from the corrective train set
│   │   ├── tinker_train.py       # LoRA fine-tune Qwen3-32B on Tinker (lazy SDK import)
│   │   ├── tinker_query.py       # Query corrected adapter or bare base model + re-score with PAI
│   │   ├── per_dp_train.py       # Train 5 isolated LoRAs (one per DP)
│   │   ├── interference.py       # Three-delta analysis (memorization, generalization, vs Gemini)
│   │   └── causal_interference.py # 5×5 causal matrix from per-DP adapters
│   └── stage6_classroom/
│       ├── case_selection.py     # Select 30 cases for Phase 2 (3 strata)
│       ├── forms.py              # Build per-student blinded evaluation forms
│       └── analysis.py           # Statistical analysis of student ratings
├── data/
│   ├── stage1/                 # Stage 1 outputs
│   ├── stage2/                 # Stage 2 outputs (incl. cases_final.jsonl)
│   ├── stage3/                 # Stage 3 outputs (optional)
│   ├── stage4/                 # Stage 4 outputs (Gemini responses)
│   └── logs/                   # Execution logs
└── tests/
    ├── test_stage1_generator.py
    ├── test_stage2_prompt_builder.py
    ├── test_stage2_validator.py
    └── test_io.py
```

## Installation

Requires **Python 3.10+** (Stage 4 depends on `google-genai`, which does not
ship wheels for Python 3.8/3.9). Recommended: a per-project virtualenv.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Copy the example and fill in your API key:

```bash
cp .env.example .env
```

Required:

| Variable | Description | Default |
|----------|-------------|---------|
| `XAI_API_KEY` | xAI API key | *(required for Stage 2)* |
| `XAI_BASE_URL` | API base URL | `https://api.x.ai/v1` |
| `MODEL` | Model name | `grok-3-fast` |
| `TEMPERATURE` | Sampling temperature | `0.8` |
| `MAX_TOKENS` | Max tokens per completion | `300` |
| `RETRIES` | API retry attempts | `3` |
| `REQUESTS_PER_MINUTE` | Rate limit | `30` |
| `CHECKPOINT_EVERY` | Save checkpoint every N cases | `50` |
| `GEMINI_API_KEY` | Gemini API key | *(required for Stage 4)* |
| `MAX_OUTPUT_TOKENS` | Max output tokens for Stage 4 | `8192` |
| `THINKING_LEVEL` | Gemini thinking level (`low`/`medium`/`high`) | `high` |
| `TINKER_API_KEY` | Tinker API key | *(required for Stage 5 fine-tuning)* |

All variables can also be overridden via CLI flags.

## Usage

### Run Stage 1

```bash
python -m src.pipeline.cli run-stage1
```

Outputs:
- `data/stage1/factorial_sample.csv`
- `data/stage1/factorial_sample.jsonl`
- `data/stage1/manifest.json`

### Run a Stage 2 pilot (first 50 cases)

```bash
python -m src.pipeline.cli run-stage2 --model grok-3-fast --start 0 --end 50
```

### Run Stage 2 in full

```bash
python -m src.pipeline.cli run-stage2 --model grok-3-fast
```

### Resume an interrupted Stage 2 run

```bash
python -m src.pipeline.cli run-stage2 --model grok-3-fast --resume
```

### Run both stages in sequence

```bash
python -m src.pipeline.cli run-all --model grok-3-fast
```

### Run Stage 4 — query Gemini 3.1 Pro Preview

Stage 4 reads the curated narratives (`data/stage3/cases_final.jsonl` if it
exists, otherwise `data/stage2/cases_final.jsonl`) and asks Gemini 3.1 Pro
Preview to design 5 learning activities per case under five pedagogical
dimensions. Output is structured JSON, validated locally, and written
incrementally.

**1. Install / upgrade the SDK:**

```bash
pip install google-genai --upgrade
```

**2. Verify the installed version:**

```bash
python -c "import google.genai; print(google.genai.__version__)"
```

**3. Configure the API key:**

```bash
export GEMINI_API_KEY="your_key_here"
```

> **Important:** Gemini 3.1 Pro Preview requires an active billing account on
> the Gemini API. Free-tier keys will fail.

**4. Run a Stage 4 pilot (first 20 cases):**

```bash
python -m src.pipeline.cli run-stage4 --start 0 --end 20
```

**5. Run Stage 4 in full:**

```bash
python -m src.pipeline.cli run-stage4
```

**6. Resume an interrupted Stage 4 run:**

```bash
python -m src.pipeline.cli run-stage4 --resume
```

Stage-4-specific flags (in addition to the shared ones):

| Flag | Description | Default |
|------|-------------|---------|
| `--input` | Path to `cases_final.jsonl` | `data/stage3/cases_final.jsonl` if present, else `data/stage2/cases_final.jsonl` |
| `--output-dir` | Output directory | `data/stage4` |
| `--model` | Gemini model id | `gemini-3.1-pro-preview` |
| `--temperature` | Sampling temperature | `0.7` |
| `--max-output-tokens` | Max output tokens | `8192` |
| `--thinking-level` | `low` / `medium` / `high` | `high` |
| `--rpm` | Requests per minute | `60` |
| `--retries` | Retry attempts per case | `3` |

Stage 4 outputs:

- `data/stage4/gemini_responses.jsonl` — incremental records (one per case)
- `data/stage4/gemini_responses.json` — consolidated JSON snapshot
- `data/stage4/manifest.json` — run metadata

### Run Stage 5 — PAI scoring + corrective LoRA fine-tuning

Stage 5 implements **Experiment 2, Phase 1** of the study. It is split into
five CLI commands so each step can be re-run independently:

**Experiment 2, Phase 1 — Corrective fine-tuning (technical)**

| Step | Command | Output | Requires |
|------|---------|--------|----------|
| Score every Stage 4 response with the PAI matrices | `run-stage5-score` | `data/stage5/scored_dataset.json` | Stage 4 outputs |
| **Identify weak (DP × condition) cells** | `run-stage5-weak-cells --k 10` | `data/stage5/weak_cells.json` (bottom-K cells by mean PAI) | scored dataset |
| Build a *global* corrective set (N lowest-PAI cases, un-stratified) | `run-stage5-corrective --n 30` | `data/stage5/corrective_training_data.json` | scored dataset |
| **Build a stratified corrective set** (50–100 examples per weak cell, proposal-aligned) | `run-stage5-corrective-stratified --per-cell 75` | `data/stage5/corrective_training_data_stratified.json` | weak_cells.json |
| **Build a per-DP corrective set** (target_dp_only — each case corrects only its weak cell's DP) | `run-stage5-corrective-stratified --per-cell 75 --target-dp-only` | `data/stage5/corrective_training_data_per_dp.json` | weak_cells.json |
| Build the held-out eval set (disjoint from train) | `run-stage5-heldout --k 30` | `data/stage5/eval_heldout_cases.json` | scored dataset + a corrective set |
| LoRA fine-tune Qwen3-32B (unified, all 5 DPs at once) at a rank sweep | `run-stage5-finetune --all-ranks` | `data/stage5/adapters/adapter_r{rank}.json` | Tinker SDK + `TINKER_API_KEY` |
| **Train 5 per-DP isolated LoRAs** (one adapter per DP, at the effective rank) | `run-stage5-per-dp-finetune --rank 8` | `data/stage5/adapters/per_dp/{dp}.json` | per-DP corrective set + Tinker |
| Query the unified adapter on train cases | `run-stage5-query --all-ranks --target train` | `data/stage5/post_intervention_r{rank}.json` | Tinker + adapters |
| Query the unified adapter on held-out cases | `run-stage5-query --all-ranks --target heldout` | `data/stage5/post_intervention_heldout_r{rank}.json` | Tinker + adapters + held-out set |
| Baseline Qwen3-32B (no LoRA) on train and held-out | `run-stage5-baseline --target both` | `data/stage5/baseline_qwen_{train,heldout}.json` | Tinker |
| Aggregate the unified-adapter runs into the three-delta interference report | `run-stage5-interference` | `data/stage5/interference_analysis.json` | post-intervention + baseline files |
| **Build the causal 5×5 interference matrix** from the per-DP adapters | `run-stage5-causal-interference --target heldout` | `data/stage5/causal_interference_matrix.json` | per-DP adapters + baseline |

**Experiment 2, Phase 2 — Human perceptual validation (classroom)**

| Step | Command | Output | Requires |
|------|---------|--------|----------|
| Select ~30 cases for Phase 2 (10 large-improvement / 10 modest / 10 control) | `run-stage6-select-cases --rank 8 --target heldout` | `data/stage6/phase2_cases.json` | a post_intervention file |
| Build per-student blinded evaluation forms (pre+post randomized) | `run-stage6-build-forms --n-students 30 --cases-per-student 5 --raters-per-case 5` | `data/stage6/forms/student_NN.json` + `assignment_key.json` | phase2_cases + pre/post outputs |
| **(off-pipeline)** Collect student ratings into `data/stage6/ratings_raw.json` | — | `data/stage6/ratings_raw.json` | run the sessions |
| Statistical analysis: paired tests, PAI-delta vs rating-delta correlation, ICC | `run-stage6-analyze` | `data/stage6/phase2_analysis.json` | ratings_raw + assignment_key |

**1. Score Stage 4 outputs with the PAI matrices:**

```bash
python -m src.pipeline.cli run-stage5-score
```

The matrices live in `src/stage5_scoring/matrices.py` (11 sub-matrices,
one per DP × condition pair). Each case is scored as the mean over its
5 activities of the mean over its 5 DPs of the mean over the relevant
sub-matrix lookups (∈ [-1, +1]).

**2. Build the corrective training set from the bottom-N cases:**

```bash
python -m src.pipeline.cli run-stage5-corrective --n 30
```

For each of the lowest-N cases by `prompt_PAI`, the optimal selection
per DP is computed from the case's learner conditions and the matrices.
The 5 activities of the assistant target reuse the optimal selection but
rotate through 5 description variants per (DP, selection) cell to avoid
training the model to memorise an identical string.

**3. Install the Tinker SDK and configure the API key:**

```bash
pip install tinker
export TINKER_API_KEY="your_key_here"
```

> Tinker access is gated by a waitlist (https://thinkingmachines.ai/tinker).
> The Stage 5 scoring + corrective-data steps work without Tinker; only
> `run-stage5-finetune` and `run-stage5-query` require the SDK.

**4. LoRA fine-tune Qwen3-32B at the diagnostic rank sweep:**

```bash
python -m src.pipeline.cli run-stage5-finetune --all-ranks --epochs 3 --lr 1e-4
```

Or pin a single rank:

```bash
python -m src.pipeline.cli run-stage5-finetune --rank 8
```

The rank at which the corrective signal first lands is the
representational-dimensionality diagnostic discussed in the study:
low rank (r=1) suggests the pedagogical decision lives in a thin
subspace; high rank (r=16) suggests the encoding is distributed.

**5. Query the corrected model on the low-PAI train cases and compute deltas:**

```bash
python -m src.pipeline.cli run-stage5-query --all-ranks --target train
```

For every rank with a saved adapter, the train cases are re-queried
through the LoRA-augmented model, the response is re-scored with the
PAI matrices, and a `post_intervention_r{rank}.json` file records the
pre/post per-DP deltas where `pre_PAI` is the Gemini 3.1 reference from
Experiment 1 and `post_PAI` is the LoRA-corrected Qwen3-32B output.

**5b. Build a held-out eval set and re-query on it (recommended for clean deltas):**

The train-target query above measures both memorization and generalization
together. To separate them, build a held-out set (cases the LoRA was never
trained on) and re-query the same adapters on it:

```bash
python -m src.pipeline.cli run-stage5-heldout --k 30
python -m src.pipeline.cli run-stage5-query --all-ranks --target heldout
```

**5c. Run the Qwen3-32B baseline (no LoRA) so we can isolate the LoRA effect:**

Without this, `delta_PAI` confounds two effects: the base-model change
(Gemini → Qwen) and the LoRA fine-tune. Run the bare base model on both
case sets:

```bash
python -m src.pipeline.cli run-stage5-baseline --target both
```

Outputs: `baseline_qwen_train.json` and `baseline_qwen_heldout.json`.

**6. Run the cross-dimensional interference analysis:**

```bash
python -m src.pipeline.cli run-stage5-interference
```

This produces `data/stage5/interference_analysis.json` containing:

- **Legacy** — the DP × rank heatmap, per-DP "first positive rank"
  (effective LoRA rank), and the top-tercile interference table from the
  train-target adapter runs (kept for backward compatibility).
- **`deltas_clean`** — the three clean deltas, per rank and per case:
  - `memorization` = Qwen+LoRA(train) − Qwen-base(train). Pure LoRA effect
    on cases the adapter trained on (largely memorization).
  - `generalization` = Qwen+LoRA(heldout) − Qwen-base(heldout). LoRA
    effect on cases the adapter has never seen — this is the real
    generalization measurement.
  - `vs_gemini` = Qwen+LoRA(heldout) − Gemini(heldout). Out-of-train
    comparison vs the Experiment 1 baseline.

### Resume / interruption safety

The expensive Tinker steps (`run-stage5-query`, `run-stage5-baseline`,
`run-stage5-per-dp-finetune`, `run-stage5-causal-interference`) are
**resumable**. Output is written per unit of work, not at the very end:

- `run-stage5-query` writes `post_intervention[_<target>]_r{rank}.json`
  after each rank completes.
- `run-stage5-per-dp-finetune` writes `adapters/per_dp/{dp}.json` after
  each per-DP adapter finishes.
- `run-stage5-causal-interference` writes
  `per_dp_query_{target}_{dp}.json` after each per-DP adapter is queried.
- `run-stage5-baseline` writes `baseline_qwen_{target}.json` on
  completion.

On a re-run, each command checks whether the corresponding output file
already exists **and matches the current run** (same `target` and the
exact same set of `prompt_id`s, or same rank/base-model for adapters).
If it matches, that unit is **skipped** and the existing file is reused
(no Tinker cost). Stale files (different target or case set — e.g. a
30-case file from an earlier design) are detected as non-matching and
redone. So if a long run is interrupted, simply re-issue the **same
command**: finished ranks/adapters are skipped, only the unfinished
work is recomputed (resume granularity is per rank / per adapter, not
per case).

Pass `--force` to ignore existing files and recompute everything from
scratch. Available on `run-stage5-query`, `run-stage5-baseline`,
`run-stage5-per-dp-finetune`, and `run-stage5-causal-interference`.

### All CLI flags

| Flag | Description |
|------|-------------|
| `--model` | Model name (e.g. `grok-3-fast`, `grok-3`, `grok-4-1-fast-non-reasoning`) |
| `--base-url` | API base URL |
| `--temperature` | Sampling temperature |
| `--max-tokens` | Max tokens per completion |
| `--start` | First row index (0-based) |
| `--end` | Last row index (exclusive) |
| `--resume` | Skip already-completed cases |
| `--retries` | Max API retries per case |
| `--requests-per-minute` | Rate limit |
| `--checkpoint-every` | Checkpoint frequency |
| `--input` / `--input-path` | Stage input file (default depends on the stage) |
| `--output-dir` | Output directory for the stage being run |
| `--force` | (Stage 5 Tinker steps) Ignore existing output files and recompute from scratch instead of resuming |
| `--corrective-file` | (Stage 5 finetune/query/causal) Which corrective set defines the train case set (e.g. `corrective_training_data_stratified.json`) |
| `--per-cell` | (`run-stage5-corrective-stratified`) Examples per weak cell (proposal: 50–100, default 75) |
| `--target-dp-only` | (`run-stage5-corrective-stratified`) Correct only the weak cell's DP (input for per-DP isolated LoRAs) |
| `--target` | (Stage 5 query/baseline/causal, Stage 6) `train` or `heldout` case set |
| `--k` | (`run-stage5-weak-cells` / `run-stage5-heldout`) Number of weak cells / held-out cases |
| `--max-output-tokens` | (Stage 4) Max output tokens for Gemini |
| `--thinking-level` | (Stage 4) Gemini thinking level (`low` / `medium` / `high`) |

## Recommended workflow

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your API key:
   ```bash
   cp .env.example .env
   # Edit .env and set XAI_API_KEY
   ```

3. Generate the factorial sample:
   ```bash
   python -m src.pipeline.cli run-stage1
   ```

4. Run a small pilot to verify prompts and outputs:
   ```bash
   python -m src.pipeline.cli run-stage2 --start 0 --end 20
   ```

5. Inspect the pilot results:
   ```bash
   head -5 data/stage2/case_narratives.jsonl | python -m json.tool
   ```

6. Run the full generation (with resume in case of interruption):
   ```bash
   python -m src.pipeline.cli run-stage2 --resume
   ```

## Output formats

### Stage 1 — `factorial_sample.csv`

| Column | Description |
|--------|-------------|
| `prompt_id` | Unique identifier (`P-0001` … `P-2160`) |
| `bloom` | Bloom's taxonomy level |
| `bloom_band` | Lower-order / Middle-order / Higher-order |
| `subject` | Academic subject |
| `subject_family` | STEM / Humanities / Social Sciences |
| `knowledge_state` | novice / informed / misinformed |
| `learning_stage` | conceptual_orientation / skill_building / competency_development / comprehensive_mastery |
| `learning_context` | guided / collaborative / autonomous |

### Stage 2 — `case_narratives.jsonl`

Each JSON line contains all Stage 1 columns plus:

| Field | Description |
|-------|-------------|
| `narrative` | Generated 80–120 word scenario |
| `word_count` | Actual word count |
| `validation` | Dict of check results (`nonempty_ok`, `single_paragraph_ok`, `word_count_ok`, `forbidden_terms_ok`) |
| `validation_clean` | `true` if all checks passed |
| `generator_provider` | `xai` |
| `generator_model` | Model used |
| `generated_at` | ISO 8601 timestamp |
| `generation_status` | `ok`, `validation_failed`, or `error` |
| `error_message` | Error details (empty if successful) |

### Experiment 2 — output files reference

Every file below lives in `data/stage5/` (Phase 1) or `data/stage6/`
(Phase 2). Each data file has a sibling `manifest_*.json` with run
parameters, timestamps and counts (provenance).

#### Phase 1 — corrective fine-tuning

| File | What it contains |
|------|------------------|
| `scored_dataset.json` | Experiment 1 input: 2 000 Gemini-3.1 cases, each with `selections`, `activity_scores`, `dp_means` and `prompt_PAI`, sorted ascending by `prompt_PAI`. |
| `weak_cells.json` | `all_cells` (every DP × condition cell with `mean_pai`, `n_cases`, `case_ids`) and `weak_cells` (the bottom-`k` by mean PAI). 82 cells total. |
| `corrective_training_data_stratified.json` | The proposal-aligned corrective set: ~50–100 cases per weak cell. Each item = chat triple (system + case narrative + pedagogically-optimal assistant response across all 5 DPs) plus `optimal_selections`, `optimal_dp_scores`, `pre_intervention_PAI`, `served_cells`, `target_dps`. |
| `corrective_training_data_per_dp.json` | Same shape but the assistant target corrects **only** the weak cell's DP (other DPs keep the model's current selection). One example per (case, target DP). Input for the per-DP isolated LoRAs. |
| `corrective_training_data.json` | Legacy global (bottom-N) corrective set, kept for reference. |
| `eval_heldout_cases.json` | Held-out evaluation cases disjoint from the corrective train set (methodological add-on, not required by the proposal). |
| `adapters/adapter_r{1,4,8,16}.json` | One per rank in the sweep: Tinker `adapter_uri`, `base_model`, `rank`, `epochs`, `n_examples`, per-epoch `history` (loss), durations, timestamps. The LoRA weights live on Tinker; this is the handle. |
| `adapters/per_dp/{content_level,student_task,tutor_role,student_engagement}.json` | The five (here four) per-DP isolated LoRA adapters. Same schema plus `target_dp`. `disciplinary_method` is absent when no DP5 cell ranks in the weak set (data-driven, expected). |
| `post_intervention_r{1,4,8,16}.json` | Per rank: `summary` (ok/total, mean/min/max `delta_PAI`, per-DP `delta_dp_means`) and `results` — one row per case with `pre_PAI` (Gemini Exp1 reference), `post_PAI` (Qwen+LoRA), `delta_PAI`, `pre_dp_means`, `post_dp_means`, `delta_dp_means`, plus the parsed model output. |
| `interference_analysis.json` | `dp_by_rank_heatmap` (mean per-DP delta at each rank), `effective_rank_by_dp` (lowest rank where each DP's delta turns positive — the representational diagnostic), `interference_top_tercile` (observational), and `deltas_clean` (memorization/generalization/vs-Gemini — populated only if the optional baseline + held-out runs were executed). |
| `per_dp_query_train_{dp}.json` | For each per-DP adapter: the same 589 cases re-queried, scored, with pre/post per-DP deltas. Used to build the causal matrix. |
| `causal_interference_matrix.json` | `matrix` = N×5 (`DP_i` corrected → mean delta on every `DP_j`). `diagonal_within_dp_effect` = direct correction effect; `off_diagonal_top` = strongest cross-DP interference (positive = collateral benefit, negative = collateral damage). `baseline_source` records whether deltas are vs Qwen-base or vs Gemini. |

#### Phase 2 — human perceptual validation (classroom)

| File | What it contains |
|------|------------------|
| `phase2_cases.json` | ~30 cases selected in 3 strata (`large_improvement`, `modest_or_zero`, `control`), each with `pre_PAI`, `post_PAI`, `delta_PAI`, `stratum`. |
| `forms/student_NN.json` | One blinded form per student: shuffled pre+post items, each with the case narrative, the model output, the 6-item rubric (5 DPs + holistic), the 4-point Likert scale, and the justification prompt. Students never see which item is pre/post. |
| `assignment_key.json` | The de-blinding key (item_uid → student, prompt_id, variant pre/post, rank, target). **Not given to students** — used only at analysis time. |
| `ratings_raw.json` | (off-pipeline) The collected student responses, schema: `[{item_uid, ratings:{rubric_id:1..4}, justification}]`. |
| `phase2_analysis.json` | Paired pre/post tests per rubric (Wilcoxon, Cohen's d), Pearson/Spearman correlation of PAI-delta vs rating-delta per rubric and per stratum, ICC inter-rater reliability, and the collected justification texts for thematic analysis. |

## Validation

Each generated narrative is automatically checked for:

- **nonempty_ok** — narrative is not empty
- **single_paragraph_ok** — no internal line breaks
- **word_count_ok** — between 80 and 120 words
- **forbidden_terms_ok** — does not contain pedagogical jargon (Bloom, scaffolding theory, etc.)

If a narrative fails on word count or forbidden terms, it is regenerated once. If it still fails, it is saved with `generation_status: "validation_failed"` and the validation flags preserved for audit.

## Tests

```bash
python -m pytest tests/ -v
```

## What is implemented

- Stage 1: full-factorial sampling (2 160 conditions)
- Stage 2: programmatic narrative generation via xAI/Grok
- Stage 3: dedup + regeneration + final-case audit
- Stage 4: target-model querying with Gemini 3.1 Pro Preview (structured JSON)
- Stage 5 (Experiment 2, Phase 1):
  - PAI scoring of every Stage 4 response with 11 theory-grounded matrices
  - Weak (DP × condition) cell identification from the scored dataset
  - Corrective LoRA training set construction (global, stratified per weak cell, or per-DP only)
  - Held-out eval set construction disjoint from the train set
  - LoRA fine-tuning pipeline for Qwen3-32B on Tinker (lazy SDK import — works without `tinker` installed for the non-training steps)
  - Per-DP isolated LoRAs (five adapters, one per decision point)
  - Post-intervention sampling (train and held-out targets) + Qwen3-32B baseline (no LoRA)
  - Three-delta interference report (memorization, generalization, vs-Gemini)
  - Causal 5×5 interference matrix from per-DP adapters
  - Resume / interruption safety for every Tinker step (per-rank / per-adapter
    granularity, target + case-set matched; `--force` to override)
- Stage 6 (Experiment 2, Phase 2):
  - Phase 2 case selection (3 strata: large improvement, modest, control)
  - Per-student blinded evaluation form generator (rubric + Likert scale + justification)
  - Statistical analysis: paired pre/post tests, PAI-delta vs rating-delta correlation per rubric and per stratum, ICC inter-rater reliability
- Resume support: checkpoint (Stage 2), JSONL-derived resume (Stage 4),
  output-file-matched resume (Stage 5 Tinker steps)
- Automatic validation with one retry (Stage 2) and per-case retries with local
  schema validation (Stage 4)
- Incremental JSONL output and consolidated JSON per stage
- Manifests per stage
- CLI with per-stage and chained execution

## What is not implemented

- The actual classroom sessions (Phase 2 forms + analysis are scaffolded; collecting ratings is off-pipeline)
- Parallel/async API calls
- Web UI or dashboard
