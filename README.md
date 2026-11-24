# Speak When You See It: Proactive Task Assistants Should Wait for Evidence, Not the Clock

[Paper (arXiv:2611.30895)](https://arxiv.org/abs/2611.30895) · [Model weights (Hugging Face collection)](https://huggingface.co/collections/bhc6/speak-when-you-see-it-ect) · [Paper page](https://huggingface.co/papers/2611.30895)

Code and results for the paper by Bohe Chen ([arXiv:2611.30895](https://arxiv.org/abs/2611.30895)). ECT LoRA adapters on Hugging Face: [`bhc6/ECT-Qwen3-VL-2B`](https://huggingface.co/bhc6/ECT-Qwen3-VL-2B) (four seeds and the no-paired-negatives ablation), [`bhc6/ECT-Qwen3-VL-4B`](https://huggingface.co/bhc6/ECT-Qwen3-VL-4B), [`bhc6/ECT-InternVL3.5-2B`](https://huggingface.co/bhc6/ECT-InternVL3.5-2B), [`bhc6/ECT-LLaVA-OneVision-7B`](https://huggingface.co/bhc6/ECT-LLaVA-OneVision-7B). This repository holds the code and results used
to check the paper's analyses; section references ("supplement Sec. B") point to the supplementary material
appended to the arXiv PDF.
The four CPU commands below run from a fresh clone without a dataset download, GPU or model weights.
Other scripts document how the results were obtained but may require inputs not bundled here, including full
annotation files, videos, model weights and complete prediction logs. Selected case material and diagnostic
records are included. Obtain QICD, Ego-MC-Bench and CaptainCook4D separately under their respective licenses.
Machine-specific paths were replaced by placeholders (`DATA_ROOT`, `WORKSPACE`, `PAPER_DIR`).
No dataset annotations are included: bundled records refer to steps by recording ID and step index (0-based, in the
order of the official loader), and the instruction and feedback texts are left out.

## Main results

Official turn-based protocol and evaluation; QICD Advanced Planning test set (36 videos) and Ego-MC-Bench (39 sessions,
no adaptation). IC-Acc: completion accuracy (%). F1: mistake F1. Δ: mistake F1 minus the best F1 a video-blind policy
(the hand-written rules and the learned blind policy on the same inputs) reaches at the same or a higher IC-Acc. ECT rows: mean ± s.d. over four seeds (range of Δ), greedy decoding.

| Model | QICD IC-Acc | QICD F1 | QICD Δ | Ego-MC IC-Acc | Ego-MC F1 | Ego-MC Δ |
|---|---|---|---|---|---|---|
| Video-blind rule, best F1 (alarm at 25 s, confirm at 30 s) | 0.0 | 0.316 | – | 0.0 | 0.373 | – |
| Video-blind rule, best IC-Acc | 46.6 | 0 | – | 56.1 | 0 | – |
| Qwen3-VL-2B zero-shot, memory prompt | 10.5 | 0.146 | -0.207 | 14.4 | 0.215 | -0.172 |
| Qwen3.8-Max zero-shot | 23.9 | 0.189 | -0.143 | 37.9 | 0.324 | +0.025 |
| Qwen3.5-2B fine-tuned on Ego-CoMist+ (published, Ego-MC-Bench v2) | – | – | – | 37.1 | 0.20 | -0.104 |
| **ECT, Qwen3-VL-2B** | **40.8 ± 1.3** | **0.247 ± 0.033** | +0.01 to +0.06 | **59.0 ± 5.7** | **0.246 ± 0.034** | +0.17 to +0.28 |

Per-seed results are on the [model card](https://huggingface.co/bhc6/ECT-Qwen3-VL-2B); all rows are in
`data/RESULTS_AUTHORITATIVE.md`.

## Results and versions

- `data/RESULTS_AUTHORITATIVE.md` indexes the recorded runs behind the paper. The supplementary material distinguishes
  the official scores, the shared validation alarm-cap comparison (Sec. D), and the secondary analyses.
- The video-blind frontier of the paper is the upper hull of the hand-written rules (`analysis/frontier_canon.py`,
  measurements in `data/frontier/`) and of the learned blind policy on the same inputs (`data/blind_strong/summary_strong.json`,
  column "paper's inputs"); `analysis/frontier_v2.py` combines the two. The trees use the official loader's combined QICD
  training and validation data, not the test recordings. Supplement tables report margins against the rules alone
  unless marked.
- The published Ego-MC-Bench comparison uses Tables 2 and 3 of `arXiv:2606.09547v2` (`data/published_frontier_egomc.json`);
  supplement Sec. B compares the v3 scores, which also lie inside the frontier. The published models were not rerun.

## Layout

| Folder | Content |
|---|---|
| `train_eval/` | Construction of the training situations and paired negatives (`sia_data.py`), training (`sia_train.py`), wrappers around the official per-step protocol for each model (`qic_*.py`), the time-matched pair probe (`evidence_pair_probe.py`), video perturbations (`qic_perturb.py`) |
| `analysis/` | Video-blind rules and the frontier (`blind_*.py`, some of them in `train_eval/`, `frontier_canon.py`, `blind_plus/`), decision decomposition (`decision_decomp.py`), pair summaries (`epair_summary.py`), paired tests (`paired_equiv.py`), threshold sweeps (`opcurve.py`), the GuideMe re-read (`guideme_blind/`) |
| `paper/` | Table and figure generators (`make_tables.py`, `make_supp.py`, `make_figs.py`) and the numeric source check (`num_check.py`); these expect the original inputs and paths |
| `data/` | Official scores and margins (`RESULTS_AUTHORITATIVE.md`), decision decomposition, pair accuracies, probe results, paired tests, blind policies and the resampled frontier (`frontier/`), threshold sweeps (`opcurve/`), QICD Main frontier (`frontier_main.json`), coverage (`coverage/`, `reach/`), step-end control (`stepend/`), step-aligned swap (`swapalign/`), shortcut analyses (`shortcuts/`) and supplemental number sources (`supp_numbers.json`) |

## Checking the paper's numbers

From the repository root, with the Python standard library only:

    python3 repro/check_paper_numbers.py

This recomputes every rule margin in `data/RESULTS_AUTHORITATIVE.md`, the ECT rows of Table 1 (mean and standard deviation
over four seeds, range of margins against the paper's frontier) and the numbers quoted from the clock analyses in
supplement B, E, F and H, and prints `ALL MATCH` on success.

## The video-blind frontier

From the repository root, run with the Python standard library only:

    python3 analysis/frontier_v2.py

This combines the rule frontier, replayed by `analysis/frontier_canon.py` from the bundled measurements in `data/frontier/`,
with the measured hull of the learned blind policy, and prints the vertices; it does not rerun evaluation or the learner.
The module exposes `fr(dataset, ic)` and `margin(dataset, ic, f1)`, with `dataset` equal to `qicd`, `mc` or `main`. The relevant upper-frontier vertices are listed in supplement Sec. B.

## Names of the runs

| Name in the data | Name in the paper |
|---|---|
| `sft_greedy`, `s1_greedy`, `s2_greedy`, `s3_greedy` (`T1v4`) | ECT, seeds 0-3, greedy decoding |
| `*_prob` | the same model with decision thresholds |
| `ABL_none` | no paired negatives |
| `ABL_noCL` | + notdone |
| `ABL_noND` | + clean |
| `T1v2_2400` | no negatives, re-balanced sampling |
| `T1_round1`, `T1`, `T1plain` | plain fine-tuning |
| `T1p2400` | cross-recording pairs in the loss (not adopted) |
| `S1v4_sia_retrieve` | auxiliary alignment loss (not adopted) |
| `A1`-`A7`, `*_swap_*` | video perturbations on the validation split (black, static, shuffled, 32 frames, swapped) |
| `S*_stream`, `*_stream` | streaming protocol |
| `API_q35_*`, `API_q38*` | Qwen3.5 / Qwen3.8 models, zero-shot |
| `noCL_s*`, `noND_s*`, `none_s*`, `max` | keys of the decision decomposition: + notdone, + clean, no paired negatives, Qwen3.8-Max |
| `R0`, `R1` | zero-shot Qwen3-VL-2B without / with the memory prompt |
| `C1`, `C4`, `C5` | InternVL3.5-2B, InternVL3.5-8B, LLaVA-OneVision-7B, zero-shot |
| `C2`, `C3`, `C6` | InternVL3.5-2B with ECT, InternVL3.5-2B without paired negatives, LLaVA-OneVision-7B with ECT |
| `B1`, `B2` | clean negatives x0.75, x3 |
| `ECT_unm_s0`, `ECT_unm_s1` (`unm_s*`) | ECT with negatives at random moments of the same steps |
| `MAIN_*`, `*_main` | runs on the Main set of QICD |
| `EXT_proassist*`, `EXT_vlo*` | ProAssist and VideoLLM-online placed in the per-step protocol (validation split) |
| `D25`, `D50` | 48 and 102 training videos |
| `T1v4_4b` | Qwen3-VL-4B |
| `test36`, `mc`, `val` | QICD test set, Ego-MC-Bench, QICD validation split |

## Supplementary analysis sources

| Folder | Content |
|---|---|
| `analysis/blind_strong/` | Stronger learned video-blind policy (`blind_strong.py`), checks against the official evaluation and half-split cross-fitting (`verify_strong.py`), summary for the table of margins against the stronger blind policy (`summarize_strong.py`). `local_eval.py` runs the official `eval.py` on annotations dumped with the official loader (not included: they are part of the dataset). |
| `analysis/budget/` | Exact replay of a thresholded run at any mistake threshold from its probability log (`budget_clean.py`), and the same-validation-alarm-cap comparison with one alarm counter on both sides (`budget_all.py`). |
| `analysis/notdone/` | Step-level analysis of what the notdone negatives change (`notdone_stats.py`, `notdone_boot.py`, `notdone_textcue.py`, `selfcheck.py`, `progress_probe.py`); results in `data/notdone/`. |
| `analysis/protocol_check/` | Annotation structure of steps with mistakes (`annot_structure.py`) and the step at which runs that stop early stop (`stall_steps.py`; supplement A). Check of the step-advance rule of the official per-step protocol on four test runs (`check_advance.py`: in every recording the number of instructions issued equals the number of confirmations plus one, capped at the number of steps); per-run counts of recordings that stop early and of steps reached in `data/protocol_check/advance.json` (supplement A). |
| `analysis/reach/` | Post-hoc official versus reached-only IC from frozen predictions (A; summary in `data/reach/reach.json`). The latter conditions on a model-specific subset. |
| `analysis/egomc_unscored/` | Post-hoc confirmation timing on steps with success after the last mistake (C; summary in `data/egomc_unscored/unscored.json`); not official IC. |
| `analysis/clock_audit/` | Clock audit. `prior_auroc.py`: single-check AUROC of raw elapsed time and of elapsed time relative to the training duration of the instruction (supplement B). `pair_prior.py`: what a blind policy using the instruction's typical duration scores on the time-matched pairs (E). `time_probe.py`: probe that changes only the time stamps written between frames (H; GPU); `tp_analyze.py` and `tp_contrast.py`: its summary, the direction of the effect and the ECT-minus-variant difference with video-bootstrap intervals. `train_pair_prior.py`: how well the instruction's typical duration separates ECT's training pairs, re-simulated from the annotations (H). `replay.py`: offline replay of the per-step protocol from per-check probability logs (steps that outlast the 60 s log are counted as confirmed at the next check and reported as inexact), exchange of the mistake probabilities between two models, per-check alarm rates (F). `zs_probe.py`: why the zero-shot model raises alarms (F). `audit.py` and `runs.py`: false-alarm timing and the partial correlation of confirmation time with the duration prior, from frozen predictions (H). Outputs in `data/clock_audit/`; the probability logs and predictions they read are not included. |
| `analysis/qualitative/` | The fixed rule that selects the success and failure cases in supplement Sec. G (`select_examples.py`) and the frame extraction (`extract_frames.py`). The figure is in the supplementary material; the original video files are not included. |

## Recomputing the paired statistics without the predictions

`repro/` holds the per-video counts for the long-step, same-validation-alarm-cap and Ego-MC-Bench random-moment
comparisons. `export_counts.py` is included for reference and requires the original prediction files.
From the repository root, with the Python standard library only:

    python3 repro/reproduce_tables.py
    python3 repro/replay_random_moment.py

The first script recomputes the long-step and same-validation-alarm-cap tables and compares with `expected_tables.json`.
The second replays the Ego-MC-Bench same-seed random-moment comparisons, including the 2,000 paired video-bootstrap draws
with seed 0 and the mistake-then-success endpoint, from `random_moment_counts.json`. Its four sets of official totals were
verified against the original evaluation logs; rows have a common video order without video identifiers. Both commands print
`ALL MATCH` on success. Neither reruns inference or scores a fresh prediction file.

## License

The code and the bundled results in this repository are released under the Apache License 2.0 (`LICENSE`).
The ECT adapters on Hugging Face are trained on QICD and are for non-commercial research use only, under the QICD Data
License Agreement; see the model cards. QICD, Ego-MC-Bench and CaptainCook4D keep their own licenses.

## Citation

```bibtex
@article{chen2026speak,
  title   = {Speak When You See It: Proactive Task Assistants Should Wait for Evidence, Not the Clock},
  author  = {Chen, Bohe},
  journal = {arXiv preprint arXiv:2611.30895},
  year    = {2026}
}
```
