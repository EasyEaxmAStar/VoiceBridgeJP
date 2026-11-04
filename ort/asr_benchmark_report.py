"""Create an auditable Markdown comparison from saved ASR benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .asr_metrics import normalized_text
from .asr_benchmark import summarize


def grouped_runs(condition: dict) -> dict[tuple[str, str], list[dict]]:
    grouped = {}
    for row in condition["runs"]:
        grouped.setdefault((row["case_id"], row["language_mode"]), []).append(row)
    return grouped


def comparison(report: dict) -> dict:
    phases = report["conditions"]
    if len(phases) != 3:
        return {"decision": "Baseline only; no candidate comparison"}
    before, candidate, after = map(grouped_runs, phases)
    regressions, changed_hypotheses, ratios = [], [], []
    for key, rows in candidate.items():
        baseline_rows = before[key] + after[key]
        if {normalized_text(row["text"]) for row in rows if "text" in row} != {normalized_text(row["text"]) for row in baseline_rows if "text" in row}:
            changed_hypotheses.append({"case": key[0], "language": key[1]})
        baseline_best = min(row["character_errors"] for row in baseline_rows)
        if max(row["character_errors"] for row in rows) > baseline_best:
            regressions.append({"case": key[0], "language": key[1],
                                "baseline_errors": baseline_best,
                                "candidate_errors": max(row["character_errors"] for row in rows)})
        if rows[0]["punctuation_errors"] is not None:
            if max(row["punctuation_errors"] for row in rows) > min(row["punctuation_errors"] for row in baseline_rows):
                regressions.append({"case": key[0], "language": key[1], "punctuation_regression": True})
        for row in rows:
            iteration = row["repeat"]
            base_times = [value["latency_s"] for value in baseline_rows if value["repeat"] == iteration]
            ratios.append(row["latency_s"] / float(np.mean(base_times)))
    memory_increase = phases[1]["peak_reserved_mib"] - max(phases[0]["peak_reserved_mib"], phases[2]["peak_reserved_mib"])
    bracket = {}
    for language in phases[0]["summary"] if "summary" in phases[0] else []:
        before_p50 = phases[0]["summary"][language]["p50_latency_s"]
        after_p50 = phases[2]["summary"][language]["p50_latency_s"]
        candidate_p50 = phases[1]["summary"][language]["p50_latency_s"]
        bracket[language] = {"baseline_drift_percent": (after_p50 / before_p50 - 1) * 100,
                             "candidate_vs_before_percent": (candidate_p50 / before_p50 - 1) * 100,
                             "candidate_vs_after_percent": (candidate_p50 / after_p50 - 1) * 100,
                             "at_least_5_percent_faster_than_both": candidate_p50 < .95 * min(before_p50, after_p50)}
    return {"accuracy_regressions": regressions,
            "changed_normalized_hypotheses": changed_hypotheses,
            "median_paired_latency_change_percent": (float(np.median(ratios)) - 1) * 100,
            "candidate_reserved_increase_mib": memory_increase,
            "baseline_bracket": bracket,
            "scope": "Candidate settings apply only to the benchmark process; application defaults are unchanged"}


def render(report: dict) -> str:
    if report.get("invalid_cases"):
        raise ValueError("Report contains rejected/unpaired references; rerun with a corrected manifest")
    lines = ["# Qwen3-ASR-0.6B benchmark results", "",
             f"GPU: {report['gpu']}; Torch {report['versions']['torch']}; SDPA; {report['model_dtype']}.", "",
             f"Normalization: {report['normalization']}.", "",
             f"Model initialization: {report['initialization_s']:.3f} s. One model instance reused for all conditions.",
             f"First actual request: {report['first_request']['latency_s']:.3f} s ({report['first_request']['case_id']}, {report['first_request']['language_mode']}).",
             "Warm measurements exclude the initial corpus priming sweep and condition setup call. File decoding, transforms and VAD preparation are recorded separately and excluded from recognition latency.", "",
             "Percentiles use NumPy linear interpolation over the recorded warm observations. Accuracy counts each case once; repeated/derived cases are not independent speech samples.", "",
             "| Condition | Hint | CER | Punctuation errors / scored cases | P50 s | P95 s | P50 RTF | P95 RTF | Peak allocated / reserved MiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for phase in report["conditions"]:
        for language, summary in phase["summary"].items():
            lines.append(f"| {phase['name']} ({phase['cpu_threads']} threads, {phase['max_new_tokens']} token cap) | {language} | {summary['cer']:.2%} ({summary['character_errors']}/{summary['reference_characters']}) | {summary['punctuation_errors']} / {summary['punctuation_scored_cases']} | {summary['p50_latency_s']:.3f} | {summary['p95_latency_s']:.3f} | {summary['p50_rtf']:.3f} | {summary['p95_rtf']:.3f} | {phase['peak_allocated_mib']:.1f} / {phase['peak_reserved_mib']:.1f} |")
    lines += ["", "## Original recordings versus derived stress cases", "",
              "| Condition | Hint | Original recordings CER | Derived stress cases CER |",
              "|---|---|---:|---:|"]
    for phase in report["conditions"]:
        for language in phase["summary"]:
            cohorts = []
            for original in (True, False):
                rows = [row for row in phase["runs"] if row["language_mode"] == language
                        and ("asset" in report["cases"][row["case_id"]]) == original]
                result = summarize(rows) if rows else None
                cohorts.append(f"{result['cer']:.2%} ({result['character_errors']}/{result['reference_characters']}; {result['cases']} cases)" if result else "n/a")
            lines.append(f"| {phase['name']} | {language} | {cohorts[0]} | {cohorts[1]} |")
    decision = comparison(report)
    lines += ["", "## Controlled comparison", "", "```json", json.dumps(decision, ensure_ascii=False, indent=2), "```", "",
              "Any accuracy regression is a reason to reject a candidate. The paired latency change compares each candidate observation with the average of matching before/after baseline observations; it is not a claim of statistical significance.", "",
              "## Baseline cases and language hints", "",
              "| Case | Duration s | Tags | Segments s | zh errors / chars | auto errors / chars | zh P50 / P95 s | auto P50 / P95 s | Clipped samples |",
              "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    first_phase = grouped_runs(report["conditions"][0])
    for case_id, case in report["cases"].items():
        zh, auto = first_phase.get((case_id, "zh"), []), first_phase.get((case_id, "auto"), [])

        def error(rows):
            return f"{min(row['character_errors'] for row in rows)}–{max(row['character_errors'] for row in rows)}/{rows[0]['reference_characters']}" if rows else "n/a"

        def latency(rows):
            return f"{np.percentile([row['latency_s'] for row in rows], 50):.3f} / {np.percentile([row['latency_s'] for row in rows], 95):.3f}" if rows else "n/a"

        lines.append(f"| {case_id} | {case['duration_s']:.3f} | {', '.join(case['tags'])} | {', '.join(f'{x:.2f}' for x in case['segment_seconds'])} | {error(zh)} | {error(auto)} | {latency(zh)} | {latency(auto)} | {case['clipped_sample_fraction']:.2%} |")
    lines += ["", "## Interpretation limits", "",
              "- Punctuation is scored only where the manifest declares a real punctuation reference. It is aligned to content-character boundaries and excluded from CER; unannotated punctuation is unavailable, not zero errors.",
              "- The noise cases use seeded additive white noise at specified SNR. They do not establish accuracy in real meeting-room noise.",
              "- VAD cases use the existing 30 ms energy segmenter with the current 700 ms pause and 8 s cap. Segment hypotheses are concatenated and compared with the original complete transcript; no per-segment ground truth is invented.",
              "- Peaks are PyTorch allocator memory; driver/context and desktop memory are excluded. Model initialization and earlier allocator reservations can exceed a condition's active tensor peak.",
              "- Mandarin-English CER counts Latin characters after case folding and removing spaces. No script, homophone, or numeral equivalence is applied.",
              "- No new production settings or optimization are adopted by this report generator.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.write_text(render(report), encoding="utf-8")
    print(json.dumps(comparison(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
