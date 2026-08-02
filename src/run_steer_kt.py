#!/usr/bin/env python3
"""Run the standard STEER-KT pipeline from retrieval through prediction."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import openai

import dataset
from prediction import (
    build_zero_shot_evidence,
    part_evidence_prompt_3,
    run_prediction_from_cached_evidence,
    save_evidence_outputs,
)
from utils import parse_prediction_text, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = PROJECT_ROOT / "outputs/evidence"
RESULT_ROOT = PROJECT_ROOT / "outputs/results"
CONFIG = {
    "DBE": {"load": "DBE", "slug": "dbe"},
    "NIPS": {"load": "NIPS34", "slug": "nips"},
    "XES": {"load": "XES", "slug": "xes"},
}

# Linux fork workers inherit these read-only objects instead of serializing the
# large train DataFrames for every chunk.
_EVIDENCE_STATE: dict[str, Any] = {}


def csv_items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def complete(path: Path, expected: int) -> bool:
    try:
        return path.exists() and len(read_json(path)) == expected
    except (OSError, json.JSONDecodeError):
        return False


def valid_predictions(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, []
    raw = read_json(path)
    valid: dict[str, Any] = {}
    invalid: list[str] = []
    for key, value in raw.items():
        if parse_prediction_text(str(value)) is None:
            invalid.append(key)
        else:
            valid[key] = value
    return valid, invalid


def make_openai_compatible_client(api_key: str, base_url: str):
    if hasattr(openai, "OpenAI"):
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    # Compatibility with openai 0.28 installations.
    openai.api_key = api_key
    if base_url:
        openai.api_base = base_url
    return SimpleNamespace(chat=SimpleNamespace(completions=openai.ChatCompletion))


def _build_evidence_chunk(positions: list[int]) -> dict[Any, dict[str, Any]]:
    bundle = _EVIDENCE_STATE["bundle"]
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        records, _, _, _ = build_zero_shot_evidence(
            LLM_test=bundle.test.iloc[positions],
            LLM_train=bundle.train,
            context=bundle.context,
            difficulty_dict=bundle.difficulty_dict,
            build_prompt_fn=part_evidence_prompt_3,
            make_peer_seq_fn=bundle.make_peer_seq_fn,
            find_peers_fn=bundle.find_peers_fn,
            find_self_fn=bundle.find_self_fn,
            window=_EVIDENCE_STATE["window"],
            representation=bundle.representation,
            output_prefix=None,
            save_every=0,
        )
    return records


def save_evidence_checkpoint(path: Path, records: dict[Any, Any]) -> None:
    answers = {key: value.get("target_r") for key, value in records.items()}
    prompts = {key: value.get("user_prompt", "") for key, value in records.items()}
    save_evidence_outputs(records, answers, prompts, path)


def build_or_resume_evidence(
    bundle,
    dataset_name: str,
    variant_suffix: str,
    window: int,
    workers: int,
    save_every: int,
) -> dict[str, Any]:
    output = EVIDENCE_ROOT / dataset_name / f"evidence{variant_suffix}.json"
    expected = len(bundle.test)
    if complete(output, expected):
        print(f"[EVIDENCE OK] {dataset_name}: {expected}/{expected} ({output})")
        return read_json(output)

    records = read_json(output) if output.exists() else {}
    completed_ids = {str(key) for key in records}
    pending = [
        position
        for position, row in enumerate(bundle.test.itertuples(index=False))
        if str(row.seq_id) not in completed_ids
    ]
    print(
        f"[EVIDENCE] {dataset_name}: resume={len(records)}/{expected}, "
        f"pending={len(pending)}, workers={workers}"
    )

    if workers == 1:
        records, _, _, _ = build_zero_shot_evidence(
            LLM_test=bundle.test,
            LLM_train=bundle.train,
            context=bundle.context,
            difficulty_dict=bundle.difficulty_dict,
            build_prompt_fn=part_evidence_prompt_3,
            make_peer_seq_fn=bundle.make_peer_seq_fn,
            find_peers_fn=bundle.find_peers_fn,
            find_self_fn=bundle.find_self_fn,
            window=window,
            representation=bundle.representation,
            output_prefix=output,
            save_every=save_every,
            resume_evidences=records,
        )
    elif pending:
        chunk_size = max(1, (len(pending) + workers * 8 - 1) // (workers * 8))
        chunks = [pending[i : i + chunk_size] for i in range(0, len(pending), chunk_size)]
        _EVIDENCE_STATE.clear()
        _EVIDENCE_STATE["bundle"] = bundle
        _EVIDENCE_STATE["window"] = window
        last_saved = len(records)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("fork"),
            ) as executor:
                futures = [executor.submit(_build_evidence_chunk, chunk) for chunk in chunks]
                for future in as_completed(futures):
                    records.update(future.result())
                    print(f"[EVIDENCE] {dataset_name}: {len(records)}/{expected}")
                    if len(records) - last_saved >= save_every:
                        save_evidence_checkpoint(output, records)
                        last_saved = len(records)
        except Exception:
            save_evidence_checkpoint(output, records)
            raise
        finally:
            _EVIDENCE_STATE.clear()
        save_evidence_checkpoint(output, records)

    if len(records) != expected:
        raise RuntimeError(f"Incomplete evidence: {dataset_name} {len(records)}/{expected}")
    print(f"[EVIDENCE DONE] {output}")
    return records


def model_dirname(model: str) -> str:
    return model.replace("/", "_")


def self_variant_suffix(self_topk: int, self_counter: bool) -> str:
    if self_topk == 3 and self_counter:
        return ""
    counter_tag = "counter" if self_counter else "nocounter"
    return f"_self{self_topk}_{counter_tag}"


def experiment_variant_suffix(
    peer_top_per_label: int,
    self_topk: int,
    self_counter: bool,
    window: int,
) -> str:
    suffix = ""
    if peer_top_per_label != 2:
        suffix += f"_peerperlabel{peer_top_per_label}"
    suffix += self_variant_suffix(self_topk, self_counter)
    if window != 5:
        suffix += f"_w{window}"
    return suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=os.getenv("DATASETS", "DBE,NIPS,XES"))
    parser.add_argument("--runs", default=os.getenv("RUNS", "1"))
    parser.add_argument(
        "--modes",
        default=os.getenv("MODES", "all"),
        help="Comma-separated prediction modes: all or self.",
    )
    parser.add_argument("--model", default=os.getenv("MODEL", "gpt-5.2"))
    parser.add_argument(
        "--window",
        type=int,
        default=int(os.getenv("WINDOW", "5")),
        help="Number of recent interactions used for target, peer, and self episodes.",
    )
    parser.add_argument(
        "--peer-top-per-label",
        type=int,
        default=int(os.getenv("PEER_TOP_PER_LABEL", "2")),
        help="Maximum retrieved peer episodes for each target outcome label.",
    )
    parser.add_argument(
        "--self-topk",
        type=int,
        default=int(os.getenv("SELF_TOPK", "3")),
    )
    parser.add_argument(
        "--self-counter",
        type=int,
        choices=(0, 1),
        default=int(os.getenv("SELF_COUNTER", "1")),
        help="Append one opposite-outcome self case when available (1=yes, 0=no).",
    )
    parser.add_argument("--service-tier", default=os.getenv("SERVICE_TIER", "flex"))
    parser.add_argument("--api-key-env", default=os.getenv("API_KEY_ENV", "OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", ""))
    parser.add_argument(
        "--evidence-workers",
        type=int,
        default=int(os.getenv("EVIDENCE_WORKERS", "4")),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("MAX_WORKERS", "8")),
    )
    parser.add_argument("--save-every", type=int, default=int(os.getenv("SAVE_EVERY", "25")))
    parser.add_argument(
        "--invalid-retries",
        type=int,
        default=int(os.getenv("INVALID_RETRIES", "3")),
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        default=os.getenv("RETRIEVE_ONLY", "0") == "1",
    )
    args = parser.parse_args()

    names = [name.upper() for name in csv_items(args.datasets)]
    unknown = set(names) - set(CONFIG)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    runs = [int(value) for value in csv_items(args.runs)]
    modes = [value.lower() for value in csv_items(args.modes)]
    unknown_modes = set(modes) - {"all", "self"}
    if unknown_modes:
        raise ValueError(f"Unknown prediction modes: {sorted(unknown_modes)}; use all or self")
    if args.evidence_workers < 1 or args.max_workers < 1:
        raise ValueError("worker counts must be at least 1")
    if args.self_topk < 1:
        raise ValueError("--self-topk must be at least 1")
    if args.peer_top_per_label < 1:
        raise ValueError("--peer-top-per-label must be at least 1")
    if args.window < 1:
        raise ValueError("--window must be at least 1")
    self_counter = bool(args.self_counter)
    variant_suffix = experiment_variant_suffix(
        args.peer_top_per_label,
        args.self_topk,
        self_counter,
        args.window,
    )

    client = None
    if not args.retrieve_only:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"{args.api_key_env} is required")
        client = make_openai_compatible_client(api_key, args.base_url)
    service_tier = None if args.service_tier.lower() in {"", "none", "null"} else args.service_tier

    try:
        for name in names:
            cfg = CONFIG[name]
            print(f"\n[LOAD] {name}")
            print(
                f"[SETTING] representation=dataset-default, "
                f"window={args.window}, "
                f"peer_top_per_label={args.peer_top_per_label}, "
                f"self_topk={args.self_topk}, "
                f"self_counter={self_counter}"
            )
            bundle = dataset.load_kt_experiment(
                cfg["load"],
                make_evidence_functions=True,
                find_peer_top_per_label=args.peer_top_per_label,
                find_self_topk=args.self_topk,
                find_self_add_counter=self_counter,
            )
            records = build_or_resume_evidence(
                bundle,
                name,
                variant_suffix,
                args.window,
                args.evidence_workers,
                args.save_every,
            )
            if args.retrieve_only:
                continue

            expected = len(bundle.test)
            output_dir = RESULT_ROOT / name / model_dirname(args.model)
            output_dir.mkdir(parents=True, exist_ok=True)
            for run in runs:
                for mode in modes:
                    result_prefix = "ours" if mode == "all" else "self"
                    output = output_dir / f"{result_prefix}{variant_suffix}_run{run}.json"
                    predictions, invalid = valid_predictions(output)
                    if invalid:
                        print(
                            f"[REPAIR] {name} {mode} run{run}: "
                            f"drop {len(invalid)} invalid response(s)"
                        )
                        save_json(predictions, output)
                    if len(predictions) == expected:
                        print(f"[RESULT OK] {name} {mode} run{run}: {expected}/{expected}")
                        continue

                    for attempt in range(args.invalid_retries + 1):
                        print(
                            f"[PREDICT] {name} {mode} run{run}: "
                            f"{len(predictions)}/{expected} | model={args.model} "
                            f"| pass={attempt + 1}/{args.invalid_retries + 1}"
                        )
                        predictions, _, _, _ = run_prediction_from_cached_evidence(
                            LLM_test=bundle.test,
                            evidence_records=records,
                            client=client,
                            context=bundle.context,
                            difficulty_dict=bundle.difficulty_dict,
                            evidence_mode=mode,
                            build_prompt_fn=part_evidence_prompt_3,
                            model=args.model,
                            service_tier=service_tier,
                            window=args.window,
                            representation=bundle.representation,
                            output_prefix=output,
                            save_every=args.save_every,
                            resume_predictions=predictions,
                            max_workers=args.max_workers,
                        )
                        predictions, invalid = valid_predictions(output)
                        if len(predictions) == expected and not invalid:
                            break
                        save_json(predictions, output)

                    if len(predictions) != expected or invalid:
                        raise RuntimeError(
                            f"Incomplete result: {name} {mode} run{run}, "
                            f"valid={len(predictions)}/{expected}, invalid={len(invalid)}"
                        )
                    print(f"[DONE] {output}")
    finally:
        if client is not None and hasattr(client, "close"):
            client.close()


if __name__ == "__main__":
    main()
