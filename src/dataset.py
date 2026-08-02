"""Dataset loading helpers for DBE, XES, NIPS, and custom datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import pickle
import sys

import numpy as np
import pandas as pd

from find_evidence import make_find_peers, make_find_self, make_peer_seq_factory
from utils import (
    KnowledgeContext,
    build_dbe_context,
    build_nips_context,
    build_xes_context,
    calculate_difficulties,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"

DEFAULT_DATASET_PATHS: dict[str, dict[str, str]] = {
    "DBE": {
        "train": str(DATASET_ROOT / "DBE/LLM/DBE_DKT_wo_expand_train.pkl"),
        "test": str(DATASET_ROOT / "DBE/LLM/DBE_LLM_test_500_99.pkl"),
    },
    "XES": {
        "train": str(DATASET_ROOT / "XES/LLM/XES_DKT_wo_expand_train.pkl"),
        "test": str(DATASET_ROOT / "XES/LLM/XES_LLM_test_500_99.pkl"),
    },
    "XES3G5M": {
        "train": str(DATASET_ROOT / "XES/LLM/XES_DKT_wo_expand_train.pkl"),
        "test": str(DATASET_ROOT / "XES/LLM/XES_LLM_test_500_99.pkl"),
    },
    "NIPS": {
        "train": str(DATASET_ROOT / "NIPS/LLM/NIPS34_DKT_wo_expand_train.pkl"),
        "test": str(DATASET_ROOT / "NIPS/LLM/NIPS34_LLM_test_500_99.pkl"),
    },
    "NIPS34": {
        "train": str(DATASET_ROOT / "NIPS/LLM/NIPS34_DKT_wo_expand_train.pkl"),
        "test": str(DATASET_ROOT / "NIPS/LLM/NIPS34_LLM_test_500_99.pkl"),
    },
}

DEFAULT_METADATA_PATHS: dict[str, dict[str, str]] = {
    "DBE": {
        "df_kcs": str(DATASET_ROOT / "DBE/metadata/KCs.csv"),
        "df_q_kc": str(DATASET_ROOT / "DBE/metadata/Question_KC_Relationships.csv"),
        "full_q_df": str(DATASET_ROOT / "DBE/metadata/Questions.csv"),
    },
    "XES": {
        "kc_map": str(DATASET_ROOT / "XES/metadata/kc_routes_map_eng.json"),
        "ques_info": str(DATASET_ROOT / "XES/metadata/questions_eng.json"),
    },
    "XES3G5M": {
        "kc_map": str(DATASET_ROOT / "XES/metadata/kc_routes_map_eng.json"),
        "ques_info": str(DATASET_ROOT / "XES/metadata/questions_eng.json"),
    },
    "NIPS": {
        "question_metadata": str(DATASET_ROOT / "NIPS/metadata/question_metadata_task_3_4.csv"),
        "subject_metadata": str(DATASET_ROOT / "NIPS/metadata/subject_metadata.csv"),
        "question_content": str(DATASET_ROOT / "NIPS/metadata/preprocessed_question_contents.json"),
    },
    "NIPS34": {
        "question_metadata": str(DATASET_ROOT / "NIPS/metadata/question_metadata_task_3_4.csv"),
        "subject_metadata": str(DATASET_ROOT / "NIPS/metadata/subject_metadata.csv"),
        "question_content": str(DATASET_ROOT / "NIPS/metadata/preprocessed_question_contents.json"),
    },
}

REPRESENTATION_BY_DATASET = {
    "DBE": "concept",
    "XES": "content",
    "XES3G5M": "content",
    "NIPS": "content",
    "NIPS34": "content",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str | None = None
    train_path: str | Path | None = None
    test_path: str | Path | None = None
    metadata_paths: dict[str, str | Path] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KTExperimentBundle(DatasetBundle):
    difficulty_dict: dict[Any, dict[str, float | int]] = field(default_factory=dict)
    context: KnowledgeContext | None = None
    representation: str = "content"
    make_peer_seq_fn: Any = None
    find_peers_fn: Any = None
    find_self_fn: Any = None


def install_numpy_pickle_aliases() -> None:
    """Support pickles written with numpy 2 module names on numpy 1 runtimes."""

    try:
        import numpy.core as npcore

        sys.modules.setdefault("numpy._core", npcore)
        for name in ["multiarray", "numeric", "umath", "fromnumeric", "shape_base", "overrides"]:
            try:
                sys.modules.setdefault(
                    f"numpy._core.{name}",
                    __import__(f"numpy.core.{name}", fromlist=["*"]),
                )
            except Exception:
                if hasattr(np.core, name):
                    sys.modules.setdefault(f"numpy._core.{name}", getattr(np.core, name))
    except Exception:
        pass


def _dataset_key(name: str | None) -> str | None:
    return None if name is None else name.upper()


def _resolve_default_path(name: str | None, split: str) -> str | None:
    if name is None:
        return None
    key = _dataset_key(name)
    if key not in DEFAULT_DATASET_PATHS:
        available = ", ".join(sorted(DEFAULT_DATASET_PATHS))
        raise ValueError(f"Unknown dataset '{name}'. Available defaults: {available}")
    return DEFAULT_DATASET_PATHS[key].get(split)


def _resolve_metadata_paths(name: str | None, metadata_paths: dict[str, str | Path] | None) -> dict[str, str | Path]:
    resolved: dict[str, str | Path] = {}
    key = _dataset_key(name)
    if key in DEFAULT_METADATA_PATHS:
        resolved.update(DEFAULT_METADATA_PATHS[key])
    resolved.update(metadata_paths or {})
    return resolved


def load_dataframe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a dataframe from pickle, CSV, JSON, JSONL, or parquet."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        install_numpy_pickle_aliases()
        return pd.read_pickle(path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Unsupported dataframe extension for '{path}'")


def load_object(path: str | Path) -> Any:
    """Load a metadata object from JSON, pickle, CSV, JSONL, or parquet."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        install_numpy_pickle_aliases()
        with path.open("rb") as f:
            return pickle.load(f)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if suffix in {".csv", ".parquet", ".jsonl", ".ndjson"}:
        return load_dataframe(path)
    raise ValueError(f"Unsupported metadata extension for '{path}'")


def load_dataset(
    name: str | None = None,
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    metadata_paths: dict[str, str | Path] | None = None,
    train_kwargs: dict[str, Any] | None = None,
    test_kwargs: dict[str, Any] | None = None,
) -> DatasetBundle:
    """Load train/test dataframes for a named or custom dataset.

    Explicit paths override the defaults in ``DEFAULT_DATASET_PATHS``.
    """

    resolved_train = train_path or _resolve_default_path(name, "train")
    resolved_test = test_path or _resolve_default_path(name, "test")
    if resolved_train is None or resolved_test is None:
        raise ValueError("Both train_path and test_path are required for custom datasets")

    train = load_dataframe(resolved_train, **(train_kwargs or {}))
    test = load_dataframe(resolved_test, **(test_kwargs or {}))
    metadata = {key: load_object(value) for key, value in _resolve_metadata_paths(name, metadata_paths).items()}
    return DatasetBundle(train=train, test=test, name=name, metadata=metadata)


def load_dataset_from_config(config: DatasetConfig) -> DatasetBundle:
    return load_dataset(
        name=config.name,
        train_path=config.train_path,
        test_path=config.test_path,
        metadata_paths=config.metadata_paths,
    )


def build_context_for_dataset(
    name: str,
    train: pd.DataFrame,
    metadata: dict[str, Any],
    min_count: int = 10,
) -> tuple[dict[Any, dict[str, float | int]], KnowledgeContext, str]:
    """Build difficulty dictionary, context, and representation for a dataset."""

    key = _dataset_key(name)
    if key is None:
        raise ValueError("name is required to build a dataset context")

    difficulty_dict = calculate_difficulties(train, min_count=min_count)
    if key in {"XES", "XES3G5M"}:
        context = build_xes_context(
            difficulty_dict=difficulty_dict,
            ques_info=metadata["ques_info"],
            kc_map=metadata["kc_map"],
        )
    elif key in {"NIPS", "NIPS34"}:
        context = build_nips_context(
            difficulty_dict=difficulty_dict,
            question_content=metadata["question_content"],
            question_metadata=metadata["question_metadata"],
            subject_metadata=metadata.get("subject_metadata"),
        )
    elif key == "DBE":
        context = build_dbe_context(
            difficulty_dict=difficulty_dict,
            df_q_kc_matching=metadata["df_q_kc"],
            df_kcs=metadata["df_kcs"],
            full_q_df=metadata.get("full_q_df"),
        )
    else:
        raise ValueError(f"Unsupported dataset for context build: {name}")
    return difficulty_dict, context, REPRESENTATION_BY_DATASET.get(key, "content")


def load_kt_experiment(
    name: str,
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    metadata_paths: dict[str, str | Path] | None = None,
    min_count: int = 10,
    make_evidence_functions: bool = True,
    find_peer_top_per_label: int = 2,
    find_self_topk: int = 1,
    find_self_add_counter: bool = True,
    relaxed_filtering: bool = False,
    min_concept_overlap: float = 0.5,
    overlap_question_map: dict[Any, list[Any]] | None = None,
) -> KTExperimentBundle:
    """Load train/test data and build reusable KT experiment objects.

    This replaces the repeated notebook boilerplate:
    load data -> load metadata -> calculate difficulties -> build context ->
    create peer/self evidence functions.
    """

    bundle = load_dataset(
        name=name,
        train_path=train_path,
        test_path=test_path,
        metadata_paths=metadata_paths,
    )
    difficulty_dict, context, representation = build_context_for_dataset(
        name=name,
        train=bundle.train,
        metadata=bundle.metadata,
        min_count=min_count,
    )

    make_peer_seq_fn = find_peers_fn = find_self_fn = None
    if make_evidence_functions:
        make_peer_seq_fn = make_peer_seq_factory(context, difficulty_dict)
        find_peers_fn = make_find_peers(
            context,
            difficulty_dict,
            representation=representation,
            relaxed_filtering=relaxed_filtering,
            min_concept_overlap=min_concept_overlap,
            overlap_question_map=overlap_question_map,
            top_per_label=find_peer_top_per_label,
        )
        find_self_fn = make_find_self(
            context,
            difficulty_dict,
            representation=representation,
            topk=find_self_topk,
            add_counter=find_self_add_counter,
        )

    return KTExperimentBundle(
        train=bundle.train,
        test=bundle.test,
        name=name,
        metadata=bundle.metadata,
        difficulty_dict=difficulty_dict,
        context=context,
        representation=representation,
        make_peer_seq_fn=make_peer_seq_fn,
        find_peers_fn=find_peers_fn,
        find_self_fn=find_self_fn,
    )
