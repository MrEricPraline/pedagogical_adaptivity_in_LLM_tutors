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
│   └── stage5_finetune/
│       ├── prompt_builder.py   # Diversified description variants for the corrective targets
│       ├── corrective_data.py  # Build corrective_training_data.json from lowest-PAI cases
│       ├── tinker_train.py     # LoRA fine-tune Qwen3-32B on Tinker (lazy SDK import)
│       ├── tinker_query.py     # Query corrected adapter + re-score with PAI
│       └── interference.py     # Cross-dimensional interference + effective-rank analysis
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

| Step | Command | Output | Requires |
|------|---------|--------|----------|
| Score every Stage 4 response with the PAI matrices | `run-stage5-score` | `data/stage5/scored_dataset.json` (sorted ascending by `prompt_PAI`) | Stage 4 outputs |
| Build the corrective training set from the *N* lowest-PAI cases | `run-stage5-corrective --n 30` | `data/stage5/corrective_training_data.json` (chat-format triples with the pedagogically optimal selections per DP) | scored dataset + Stage 4 narratives |
| LoRA fine-tune Qwen3-32B on Tinker at one or more ranks | `run-stage5-finetune --all-ranks` (or `--rank 8`) | `data/stage5/adapters/adapter_r{rank}.json` + Tinker checkpoints | Tinker SDK + `TINKER_API_KEY` |
| Query the corrected adapter on the same low-PAI cases and compute pre/post PAI deltas | `run-stage5-query --all-ranks` | `data/stage5/post_intervention_r{rank}.json` (one per rank) | Tinker SDK + trained adapters |
| Aggregate the rank sweep into the effective-rank + interference report | `run-stage5-interference` | `data/stage5/interference_analysis.json` | post-intervention files |

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

**5. Query the corrected model on the low-PAI cases and compute deltas:**

```bash
python -m src.pipeline.cli run-stage5-query --all-ranks
```

For every rank with a saved adapter, the same 30 cases are re-queried
through the LoRA-augmented model, the response is re-scored with the
PAI matrices, and a `post_intervention_r{rank}.json` file records the
pre/post per-DP deltas.

**6. Run the cross-dimensional interference analysis:**

```bash
python -m src.pipeline.cli run-stage5-interference
```

This produces `data/stage5/interference_analysis.json` with: a DP × rank
heatmap of mean delta, the per-DP "first positive rank" (effective LoRA
rank), and a top-tercile interference table that flags DPs whose
already-high pre scores dropped after the unified corrective pass.

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
- Stage 5 (Phase 1):
  - PAI scoring of every Stage 4 response with 11 theory-grounded matrices
  - Corrective LoRA training set construction from the lowest-PAI cases
  - LoRA fine-tuning pipeline for Qwen3-32B on Tinker (lazy SDK import — works without `tinker` installed for the non-training steps)
  - Post-intervention sampling, re-scoring, and pre/post delta computation
  - Cross-dimensional interference + effective-rank analysis across the LoRA rank sweep
- Resume support with checkpoint (Stage 2) and JSONL-derived resume (Stage 4)
- Automatic validation with one retry (Stage 2) and per-case retries with local
  schema validation (Stage 4)
- Incremental JSONL output and consolidated JSON per stage
- Manifests per stage
- CLI with per-stage and chained execution

## What is not implemented

- Stage 5 Phase 2: human-perceptual validation in the classroom (out of scope for the code pipeline; lives in the Stage 2 student-form preparation)
- Per-DP isolated LoRAs for the *causal* interference matrix (current interference analysis is observational; see the future-work notes at the bottom of `src/stage5_finetune/interference.py`)
- Parallel/async API calls
- Web UI or dashboard
