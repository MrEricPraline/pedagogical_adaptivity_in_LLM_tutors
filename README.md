# Pedagogical Adaptivity in LLM Tutors — Research Pipeline

Research pipeline for generating and analysing case narratives that explore how LLM tutors should adapt their pedagogical approach based on learner variables. The pipeline produces a fully-crossed factorial design and then uses xAI/Grok to generate naturalistic learning scenarios for each condition.

## Stages

| Stage | Description | Output |
|-------|-------------|--------|
| **Stage 1** — Factorial sampling | Generates the full-factorial crossing of five experimental variables (Bloom level, knowledge state, learning stage, learning context, subject). | `data/stage1/factorial_sample.csv` (2 160 rows) |
| **Stage 2** — Case narrative generation | For each row, calls Grok via the xAI API to produce a realistic 80–120 word learning scenario where the experimental variables are embedded implicitly. | `data/stage2/case_narratives.jsonl` |

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
│   └── stage2_generation/
│       ├── prompt_builder.py   # Variable → natural-language translation
│       ├── provider_xai.py     # xAI/Grok client (retry + rate limit)
│       ├── validator.py        # Narrative validation checks
│       ├── checkpoint.py       # Resume support
│       └── pipeline.py         # Stage 2 orchestrator
├── data/
│   ├── stage1/                 # Stage 1 outputs
│   ├── stage2/                 # Stage 2 outputs
│   └── logs/                   # Execution logs
└── tests/
    ├── test_stage1_generator.py
    ├── test_stage2_prompt_builder.py
    ├── test_stage2_validator.py
    └── test_io.py
```

## Installation

```bash
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
| `--input-path` | Path to factorial CSV (default: `data/stage1/factorial_sample.csv`) |
| `--output-dir` | Output directory for the stage being run |

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
- Resume support with checkpoint
- Automatic validation with one retry
- Incremental JSONL output
- Manifests per stage
- CLI with per-stage and chained execution

## What is not implemented

- Stages 3+ (rating, analysis, etc.)
- Parallel/async API calls
- Web UI or dashboard
