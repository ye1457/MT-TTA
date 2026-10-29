# Evaluation protocol

[Home](../README.md) · [Runnable reproduction guide](REPRODUCIBILITY.md) · [Reference results](RESULTS.md)

PACS and OfficeHome share evaluation rules with explicit method differences.

## Stream and state

Use seed 0 and every image once in the reported branch, batch size 1, with a pre-update student prediction. Domains run in the declared order; images are sorted by class-directory name and full path. No shuffle, truncation, cycling or external order table is used.

All text, visual, position and scale prompts persist across domains within a branch. Empty AdamW states are recreated at each branch/domain boundary. Each new benchmark or independent run starts fresh. Student and teacher backbones remain frozen.

Text routing previews the first 64 images without ground-truth labels, teacher outputs or updates, then fixes expert IDs/weights for the domain. Formal scoring restarts at image 0 and includes preview images once. Class sorting can concentrate the preview in one class; this is not random/class-balanced previewing. Known domain boundaries and lookahead are part of this protocol.

## Method

- Student OpenAI CLIP ViT-B/16; auxiliary teacher ViT-L/14.
- Text: 4 context tokens initialized from `a photo of a`; 3 shared + 8 private experts; private Top-2; shared mixture 0.5; expert noise std 0.02.
- PACS scores each text expert by mean `p_top1-p_top2` over its 64 previews; OfficeHome uses mean maximum class probability. Scores are quantized to model dtype before expert Top-K and routing softmax.
- Visual: 2 tokens/expert; 2 shared + 4 private; per-image Top-1; `current_text_moe`; shared mixture 0.44; scale initial 0.25; no extra position noise. Routing temperatures 1.
- Pre-update student gate >=0.95 for PACS or >=0.815 for OfficeHome. Each selected image receives 3 updates with fixed pseudo-labels and image route.
- Loss: student hard-pseudo-label CE + 0.08 teacher hard-pseudo-label CE. AdamW: weight decay 0, betas (0.9,0.999), eps 1e-8. LR 7.5e-6, except OfficeHome Clipart uses 2e-6 in both branches.
- No teacher output fusion/calibration, extra random augmentation, replay, EMA, entropy/anchor penalty or KL term in these configurations.
- Deterministic CLIP preprocessing: RGB, bicubic short-side 224, center 224x224 crop, CLIP normalization. Preserve original CLIP CUDA dtype and fp32 external prompts; no extra AMP policy.

## OfficeHome shadow

Initialize primary then text-only shadow before either runs. Their prompts are independent; only frozen backbones/teacher are shared. Execute:

`shadow Art → primary Art → shadow Clipart → primary Clipart → shadow Product → primary Product → shadow Real_World`.

Report primary for Art/Clipart/Product and shadow for Real_World, selected before running. Both branches preserve prompts across domains. The 15,588 reported images require 26,819 branch visits and 448 previews. PACS uses 9,991 primary visits and 256 previews.

## Metrics and records

Primary metric: equal-weight mean of four domain Top-1 accuracies. Also report per-domain correct/total, pooled accuracy, selected images, update counts, branch/preview compute and runtime. Use one configuration throughout the stream, never concatenate per-domain winners.

The evaluator records effective settings, generated input order, source/model/input hashes, pre-update student traces, optimizer creation and prompt-state boundaries. Existing output directories are rejected. Real labels support scoring/audit, not adaptation losses, route scores or gates; class-based input ordering is separately declared above.

After a successful run, `summary.json` contains the per-domain results and reported macro accuracy; `completion.json` certifies `PASS` with bound artifact hashes. `config.json`, `metadata.json`, `input_manifest.json` and `*.predictions.jsonl` record the effective run. Prediction rows include cumulative pre-update step counts; these evaluators do not create a separate `updates.jsonl`. Keep the complete output directory if you need to independently recount that run later.

The evaluator has no midstream prompt/optimizer checkpoint restoration. If interrupted, restart from the beginning in a new output directory. Do not edit runtime source files while an evaluation is running. CUDA/platform or dependency changes may affect finite-precision results; use the reference environment, official weights and original filenames when comparing against the reference.

Historical macro references: PACS 98.997603% (tuned on this same full stream), OfficeHome 86.488202%. These summaries are retained in [RESULTS.md](RESULTS.md); historical raw predictions and per-run output backups were deleted at the owner's request. Refactoring checks and small numerical smoke tests are not new full benchmark results.
