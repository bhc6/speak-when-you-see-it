"""Replay the registered Ego-MC random-moment control from per-video totals.

Pass --export DATA_ROOT locally to derive those totals from the frozen logs,
predictions and benchmark annotations. Plain replay needs only the stdlib.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import random
import re
import sys
import types


RUNS = {
    "ect0": "machine0/results/final/sft_greedy_mc",
    "unm0": "machine8/results/jobs/ECT_unm_s0_greedy_mc",
    "ect1": "machine4/results/seeds/final/s1_greedy_mc",
    "unm1": "machine8/results/jobs/ECT_unm_s1_greedy_mc",
}
FIELDS = ("ic_hits", "ic_total", "tp", "fp", "tn", "fn", "target", "reached", "on_time")


def target_counts(scorer, dataset, prediction):
    gt = scorer.get_ground_truth_data(dataset, prediction["video_id"])
    segments, times = scorer.segment_pred_feedbacks(prediction["pred_texts"], prediction["pred_timestamps"])
    target = reached = on_time = 0
    for idx, (labels, stamps) in enumerate(zip(gt["texts_types"], gt["texts_timestamps"])):
        mistakes = [stamp for stamp, label in zip(stamps, labels) if "mistake" in label]
        successes = [stamp for stamp, label in zip(stamps, labels) if "success" in label]
        if not (gt["has_mistake"][idx] and mistakes and successes and max(successes) > max(mistakes)):
            continue
        target += 1
        if idx >= len(segments):
            continue
        instruction = segments[idx][0].replace("Instruct:", "").strip()
        if not scorer.check_if_correct_instruction_given(gt["texts"][idx][0], instruction):
            continue
        reached += 1
        confirmation = next((stamp for text, stamp in zip(segments[idx][1:], times[idx][1:]) if "Success:" in text), None)
        on_time += confirmation is not None and abs(confirmation - max(successes)) < 15
    return [target, reached, on_time]


def export_counts(root):
    torch = types.ModuleType("torch")
    torch.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object))
    sys.modules.update({"torch": torch, "torch.utils": torch.utils, "torch.utils.data": torch.utils.data})
    sys.path.insert(0, str(root / "analysis_code/blind_strong"))
    import local_eval

    scorer = local_eval.ev
    dataset = local_eval.load("ego_mc_bench", "test")
    video_ids = sorted(item["video_id"] for item in dataset.annotations)
    assert len(set(video_ids)) == len(video_ids) == 39
    source = Path(scorer.__file__).read_text()
    anchor = '        print("\\n<<<")'
    assert source.count(anchor) == 1
    capture = "        collect(num_detected_instructions, total_instructions_without_mistakes, mistake_tp, mistake_fp, mistake_tn, mistake_fn)\n"
    scorer.tqdm = lambda items: items
    counts = {}
    for name, path in RUNS.items():
        folder = root / path
        predictions = json.loads((folder / "pred.json").read_text())
        assert sorted(item["video_id"] for item in predictions) == video_ids
        predictions.sort(key=lambda item: item["video_id"])
        cumulative = []
        scope = dict(vars(scorer), collect=lambda *values: cumulative.append(values))
        exec(compile(source.replace(anchor, capture + anchor), scorer.__file__, "exec"), scope)
        with contextlib.redirect_stdout(io.StringIO()):
            scope["run"](dataset, predictions)
        log = (folder / "eval.log").read_text()
        ic = re.findall(r"IC-Acc: ([\d.]+)", log)[-1]
        confusion = re.findall(r"mistake_tp:(\d+), mistake_fp:(\d+), mistake_tn:(\d+), mistake_fn:(\d+)", log)[-1]
        assert cumulative[-1][2:] == tuple(map(int, confusion)), name
        assert f"{100 * cumulative[-1][0] / cumulative[-1][1]:.1f}" == ic, name
        previous = (0,) * 6
        rows = []
        for totals, prediction in zip(cumulative, predictions, strict=True):
            rows.append([total - earlier for total, earlier in zip(totals, previous)] + target_counts(scorer, dataset, prediction))
            previous = totals
        counts[name] = rows
    return counts


def metrics(rows):
    hits, total, tp, fp, _tn, fn, target, reached, on_time = map(sum, zip(*rows))
    assert 0 <= on_time <= reached <= target
    return 100 * hits / total, 2 * tp / (2 * tp + fp + fn), 100 * on_time / reached


def summarize(counts):
    assert all(len(rows) == 39 for rows in counts.values())
    assert all(sum(row[6] for row in rows) == 282 for rows in counts.values())
    rng = random.Random(0)
    draws = [[rng.randrange(39) for _ in range(39)] for _ in range(2000)]
    output = {}
    for seed in (0, 1):
        ect, unm = counts[f"ect{seed}"], counts[f"unm{seed}"]
        assert all((left[1], left[6]) == (right[1], right[6]) for left, right in zip(ect, unm))
        reference, control = metrics(ect), metrics(unm)
        samples = [[], [], []]
        for draw in draws:
            left = metrics([ect[idx] for idx in draw])
            right = metrics([unm[idx] for idx in draw])
            for series, base, variant in zip(samples, left, right):
                series.append(variant - base)
        output[str(seed)] = {}
        for name, digits, base, variant, series in zip(("ic", "f1", "on_time"), (1, 3, 1), reference, control, samples):
            series.sort()
            output[str(seed)][name] = dict(ect=round(base, digits), random=round(variant, digits),
                                          delta=round(variant - base, digits),
                                          ci95=[round(series[idx], digits) for idx in (50, 1949)])
        output[str(seed)]["reached"] = [sum(row[7] for row in run) for run in (ect, unm)]
        output[str(seed)]["confusion"] = [list(map(sum, zip(*run)))[2:6] for run in (ect, unm)]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="local dataset and prediction tree")
    parser.add_argument("--counts", type=Path, default=Path(__file__).with_name("random_moment_counts.json"))
    args = parser.parse_args()
    artifact = json.loads(args.counts.read_text())
    assert artifact["fields"] == list(FIELDS)
    counts = export_counts(args.export.resolve()) if args.export else artifact["counts"]
    assert counts == artifact["counts"], "frozen predictions no longer match the exported totals"
    output = summarize(counts)
    assert output == artifact["expected"], "replay does not match the frozen summary"
    print(json.dumps(output, indent=2))
    print("ALL MATCH")


if __name__ == "__main__":
    main()
