"""Stage 5 Phase 1 — corrective LoRA fine-tuning on Tinker.

Subpackages:

* :mod:`src.stage5_finetune.corrective_data` — generate the corrective
  training set from the lowest-PAI cases produced by stage5_scoring.
* :mod:`src.stage5_finetune.tinker_train` — LoRA fine-tune Qwen3-32B on
  Tinker at ranks r=1, 4, 8, 16.
* :mod:`src.stage5_finetune.tinker_query` — query the corrected adapter
  on the same low-PAI cases and re-score with the PAI matrices.
* :mod:`src.stage5_finetune.interference` — cross-dimensional
  interference analysis across ranks.
"""
