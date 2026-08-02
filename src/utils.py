"""Reusable adaptive-KT utilities shared by DBE, XES, and NIPS notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import json
import re

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


DIFFICULTY_LABELS = ("Very Hard", "Hard", "Mid", "Easy", "Very Easy")
Representation = Literal["auto", "content", "concept"]


def _qid_key(qid: Any) -> Any:
    try:
        return int(qid)
    except (TypeError, ValueError):
        return str(qid)


def _lookup(mapping: Mapping[Any, Any], key: Any, default: Any = None) -> Any:
    candidates = (key, _qid_key(key), str(key))
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return default


def calculate_difficulties(
    train_df: pd.DataFrame,
    questions_col: str = "questions",
    responses_col: str = "responses",
    min_count: int = 10,
) -> dict[Any, dict[str, float | int]]:
    """Calculate per-question empirical correctness from training sequences."""

    q_stats: dict[Any, dict[str, int]] = {}
    for questions, responses in zip(train_df[questions_col], train_df[responses_col]):
        for q, r in zip(questions, responses):
            qid = _qid_key(q)
            if qid not in q_stats:
                q_stats[qid] = {"cnt": 0, "cor": 0}
            q_stats[qid]["cnt"] += 1
            if int(r) == 1:
                q_stats[qid]["cor"] += 1

    difficulty_dict: dict[Any, dict[str, float | int]] = {}
    for qid, stat in q_stats.items():
        cnt = stat["cnt"]
        if cnt < min_count:
            continue
        ans = min(1.0, float(stat["cor"] / cnt))
        difficulty_dict[qid] = {**stat, "ans": ans}
    return difficulty_dict


def difficulty_thresholds(difficulty_dict: Mapping[Any, Mapping[str, float]]) -> tuple[float, float]:
    difficulties = [1 - float(v["ans"]) for v in difficulty_dict.values()]
    if not difficulties:
        return np.nan, np.nan
    return float(np.quantile(difficulties, 0.25)), float(np.quantile(difficulties, 0.75))


def answer_rate_values(difficulty_dict: Mapping[Any, Mapping[str, float]]) -> np.ndarray:
    return np.sort([float(v["ans"]) for v in difficulty_dict.values()])


def decide_label(
    target_id: Any,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    ans_diffusion: Sequence[float] | None = None,
    unknown_label: str = "unknown",
) -> str:
    """Assign a five-level difficulty label from empirical answer-rate percentiles."""

    stat = _lookup(difficulty_dict, target_id)
    if stat is None:
        return unknown_label
    rates = np.asarray(ans_diffusion if ans_diffusion is not None else answer_rate_values(difficulty_dict))
    if rates.size == 0:
        return unknown_label
    target_per = round(100 - np.mean(rates >= float(stat["ans"])) * 100, 2)
    if target_per < 20:
        return "Very Hard"
    if target_per < 40:
        return "Hard"
    if target_per < 60:
        return "Mid"
    if target_per < 80:
        return "Easy"
    return "Very Easy"


def get_peer_rate(qid: Any, difficulty_dict: Mapping[Any, Mapping[str, float]], default: float | None = None) -> float | None:
    stat = _lookup(difficulty_dict, qid)
    if stat is None or stat.get("ans") is None:
        return default
    return float(stat["ans"])


def format_peer_rate(
    qid: Any,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    digits: int = 4,
    unknown: str = "uncertain",
) -> str:
    rate = get_peer_rate(qid, difficulty_dict)
    return unknown if rate is None else str(round(rate, digits))


def save_json(data: Mapping[Any, Any], path: str | Path) -> None:
    path = Path(path)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    str_key_dict = {str(k): v for k, v in data.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(str_key_dict, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix != ".json":
        path = path.with_suffix(".json")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_json_string(text: str) -> str:
    """Remove invisible/control characters that often break LLM JSON parsing."""

    text = text.replace("\u200b", "").replace("\xa0", " ")
    return re.sub(r"[\x00-\x1f\x7f]", "", text)


def safe_json_loads(text: str) -> Any:
    """Parse the first JSON object from a noisy string."""

    text = sanitize_json_string(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found")
    return json.loads(text[start : end + 1])


def file_save(
    zero_shot_prediction: Mapping[Any, Any],
    answer_list: Mapping[Any, Any],
    task_prompts: Mapping[Any, Any] | None,
    name: str | Path,
    name2: str | Path,
    name3: str | Path | None = None,
) -> None:
    """Save predictions, answers, and optionally prompts as JSON files."""

    if name3 is not None and task_prompts is not None:
        save_json(task_prompts, name3)
    save_json(zero_shot_prediction, name)
    save_json(answer_list, name2)


def file_read(
    name: str | Path,
    name2: str | Path,
    name3: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    zer_v1 = read_json(name)
    ans_v1 = read_json(name2)
    task_v1 = None if name3 in {None, "None"} else read_json(name3)
    return zer_v1, ans_v1, task_v1


def parse_prediction_text(text: str) -> int | None:
    """Parse Correct/Incorrect style LLM output into 1/0."""

    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = safe_json_loads(text)
        except Exception:
            parsed = None
    if isinstance(parsed, Mapping):
        prediction = parsed.get("prediction")
        if isinstance(prediction, str):
            lowered_prediction = prediction.strip().lower()
            if lowered_prediction == "correct":
                return 1
            if lowered_prediction == "incorrect":
                return 0

    marker_idx = 0
    lowered = text.lower()
    for marker in ("final prediction:", '"prediction":', "prediction:"):
        idx = lowered.find(marker)
        if idx >= 0:
            marker_idx = idx + len(marker)
            break
    tail_lower = text[marker_idx:].strip().lower()
    if tail_lower.startswith(("<incorrect>", "incorrect", '"incorrect"', "z9")):
        return 0
    if tail_lower.startswith(("<correct>", "correct", '"correct"', "q7")):
        return 1
    if "incorrect" in tail_lower:
        return 0
    if "correct" in tail_lower:
        return 1
    return None
def zero_answer(
    zero_shot_tracing: Mapping[Any, str],
    answer_list: Mapping[Any, Any],
    seq_ids: Iterable[Any],
    show_progress: bool = True,
) -> tuple[list[int], list[int], list[Any]]:
    """Convert raw LLM predictions to binary predictions and aligned answers."""

    answers: list[int] = []
    zero: list[int] = []
    rest: list[Any] = []
    iterator = tqdm.tqdm(seq_ids) if show_progress else seq_ids
    for u in iterator:
        lookup_id = str(u) if u not in zero_shot_tracing and str(u) in zero_shot_tracing else u
        if lookup_id not in zero_shot_tracing:
            rest.append(u)
            continue
        pred = parse_prediction_text(str(zero_shot_tracing[lookup_id]))
        if pred is None:
            rest.append(u)
            continue
        zero.append(pred)
        answers.append(int(answer_list[lookup_id] if lookup_id in answer_list else answer_list[str(lookup_id)]))
    return zero, answers, rest


def zero_answer_read(
    zero_shot_tracing: Mapping[Any, str],
    answer_list: Mapping[Any, Any],
    seq_ids: Iterable[Any],
    show_progress: bool = True,
) -> tuple[list[int], list[int], list[Any]]:
    return zero_answer(zero_shot_tracing, answer_list, (str(u) for u in seq_ids), show_progress)


def evaluation(zero: Sequence[int], answers: Sequence[int], print_result: bool = True) -> dict[str, Any]:
    """Return and optionally print accuracy, macro-F1, per-class F1/recall/precision."""

    metrics = {
        "accuracy": accuracy_score(answers, zero),
        "macro_f1": f1_score(answers, zero, average="macro"),
        "f1": f1_score(answers, zero, average=None),
        "recall": recall_score(answers, zero, average=None),
        "precision": precision_score(answers, zero, average=None),
    }
    if print_result:
        print("Accuracy:", metrics["accuracy"])
        print("macro f1: ", metrics["macro_f1"])
        print(metrics["f1"])
        print(metrics["recall"])
        print(metrics["precision"])
    return metrics


def majority_voting(
    zz1: Sequence[int],
    zz2: Sequence[int],
    zz3: Sequence[int],
    answers: Sequence[int] | None = None,
) -> np.ndarray:
    final_pred = (np.sum(np.vstack([np.array(zz1), np.array(zz2), np.array(zz3)]), axis=0) >= 2).astype(int)
    if answers is not None:
        evaluation(final_pred, answers)
    return final_pred


@dataclass
class KnowledgeContext:
    """Metadata hooks used by history/question formatting functions."""

    difficulty_dict: Mapping[Any, Mapping[str, float]]
    question_to_concepts: Callable[[Any], Sequence[Any]] | None = None
    concept_name: Callable[[Any], str] | None = None
    concept_description: Callable[[Any], str] | None = None
    question_text: Callable[[Any], str] | None = None
    concept_mask: Callable[[Any], str] | None = None
    unknown_label: str = "unknown"
    default_representation: Literal["content", "concept"] = "content"

    def label(self, qid: Any) -> str:
        return decide_label(qid, self.difficulty_dict, unknown_label=self.unknown_label)

    def peer_rate(self, qid: Any) -> str:
        return format_peer_rate(qid, self.difficulty_dict)

    def concepts(self, qid: Any) -> list[Any]:
        if self.question_to_concepts is None:
            return []
        return list(self.question_to_concepts(qid))

    def concept_names(self, qid: Any) -> list[str]:
        concepts = self.concepts(qid)
        if self.concept_name is None:
            return [str(c) for c in concepts]
        return [self.concept_name(c) for c in concepts]

    def concept_masks_with_names(self, qid: Any) -> list[str]:
        values = []
        for concept in self.concepts(qid):
            name = self.concept_name(concept) if self.concept_name else str(concept)
            mask = self.concept_mask(concept) if self.concept_mask else str(concept)
            values.append(f"{mask} ({name})")
        return values

    def content(self, qid: Any, unknown: str = "") -> str:
        if self.question_text is None:
            return unknown
        return str(self.question_text(qid))


def concept_overlap_ratio(
    target_concepts: Iterable[Any],
    candidate_concepts: Iterable[Any],
    denominator: Literal["target", "candidate", "min", "union"] = "target",
    exclude_concepts: Iterable[Any] | None = None,
) -> float:
    """Return concept-overlap ratio between two questions.

    The default denominator is the target question's concept count. This matches
    cold-question peer filtering: a candidate question is usable when it covers
    enough of the target question's concept set.
    """

    excluded = {str(concept) for concept in (exclude_concepts or [])}
    target_set = {str(concept) for concept in target_concepts if str(concept) != "" and str(concept) not in excluded}
    candidate_set = {str(concept) for concept in candidate_concepts if str(concept) != "" and str(concept) not in excluded}
    if not target_set or not candidate_set:
        return 0.0

    intersection = target_set.intersection(candidate_set)
    if denominator == "target":
        base = len(target_set)
    elif denominator == "candidate":
        base = len(candidate_set)
    elif denominator == "min":
        base = min(len(target_set), len(candidate_set))
    elif denominator == "union":
        base = len(target_set.union(candidate_set))
    else:
        raise ValueError(f"Unsupported denominator: {denominator}")
    return len(intersection) / base if base > 0 else 0.0


def build_concept_overlap_question_map(
    context: KnowledgeContext | None,
    qids: Iterable[Any] | None = None,
    overlap: float = 0.5,
    denominator: Literal["target", "candidate", "min", "union"] = "target",
    include_self: bool = False,
    return_scores: bool = False,
    show_progress: bool = False,
    exclude_concepts: Iterable[Any] | None = None,
    question_to_concepts: Mapping[Any, Sequence[Any]] | Callable[[Any], Sequence[Any]] | None = None,
) -> dict[Any, list[Any] | list[tuple[Any, float]]]:
    """Map each question to questions whose concept sets overlap enough.

    Parameters
    ----------
    context:
        Dataset-specific ``KnowledgeContext`` for XES, NIPS, or DBE.
    qids:
        Question ids to match. If omitted, keys from ``context.difficulty_dict``
        are used, which usually means train-seen questions.
    overlap:
        Minimum overlap ratio. With the default denominator, ``0.5`` means the
        candidate question contains at least half of the target question's
        concepts.
    denominator:
        Ratio denominator. ``"target"`` is the intended cold-question setting.
    include_self:
        Whether a question can match itself.
    return_scores:
        If true, each value is ``[(matched_qid, overlap_ratio), ...]``.
        Otherwise each value is ``[matched_qid, ...]``.
    show_progress:
        If true, show a tqdm progress bar over target questions.
    exclude_concepts:
        Concept ids to ignore before computing overlap, e.g. ``["865"]`` for
        XES or ``[3]`` for NIPS root concepts.
    question_to_concepts:
        Optional precomputed question-to-concepts mapping. Use this for
        leaf-only matching or other custom concept spaces.
    """

    if not 0 <= overlap <= 1:
        raise ValueError("overlap must be between 0 and 1")

    if qids is None:
        if question_to_concepts is not None and not callable(question_to_concepts):
            question_ids = list(question_to_concepts.keys())
        elif context is not None:
            question_ids = list(context.difficulty_dict.keys())
        else:
            raise ValueError("qids are required when context is None and question_to_concepts is callable")
    else:
        question_ids = list(qids)

    excluded = {str(concept) for concept in (exclude_concepts or [])}

    def get_concepts(qid: Any) -> list[str]:
        if question_to_concepts is None:
            if context is None:
                raise ValueError("context is required when question_to_concepts is not provided")
            values = context.concepts(qid)
        elif callable(question_to_concepts):
            values = question_to_concepts(qid)
        else:
            values = _lookup(question_to_concepts, qid, [])
        seen = set()
        concepts = []
        for concept in values or []:
            key = str(concept)
            if key and key not in excluded and key not in seen:
                seen.add(key)
                concepts.append(key)
        return concepts

    concept_by_qid = {qid: get_concepts(qid) for qid in question_ids}
    concept_index: dict[str, list[Any]] = {}
    for qid, concepts in concept_by_qid.items():
        for concept in concepts:
            concept_index.setdefault(concept, []).append(qid)

    matching: dict[Any, list[Any] | list[tuple[Any, float]]] = {}

    iterator = tqdm.tqdm(question_ids, desc="Matching concept-overlap questions") if show_progress else question_ids
    for target_qid in iterator:
        matches: list[Any] | list[tuple[Any, float]] = []
        target_concepts = concept_by_qid.get(target_qid, [])
        candidate_counts: dict[Any, int] = {}
        for concept in target_concepts:
            for candidate_qid in concept_index.get(concept, []):
                candidate_counts[candidate_qid] = candidate_counts.get(candidate_qid, 0) + 1

        for candidate_qid, intersection_size in candidate_counts.items():
            if not include_self and str(candidate_qid) == str(target_qid):
                continue
            candidate_concepts = concept_by_qid.get(candidate_qid, [])
            if denominator == "target":
                base = len(target_concepts)
            elif denominator == "candidate":
                base = len(candidate_concepts)
            elif denominator == "min":
                base = min(len(target_concepts), len(candidate_concepts))
            elif denominator == "union":
                base = len(set(target_concepts).union(candidate_concepts))
            else:
                raise ValueError(f"Unsupported denominator: {denominator}")
            score = intersection_size / base if base > 0 else 0.0
            if score >= overlap:
                if return_scores:
                    matches.append((candidate_qid, score))  # type: ignore[arg-type]
                else:
                    matches.append(candidate_qid)  # type: ignore[arg-type]

        if return_scores:
            matches = sorted(matches, key=lambda item: (-item[1], str(item[0])))  # type: ignore[index]
        else:
            matches = sorted(matches, key=lambda item: str(item))
        matching[target_qid] = matches
    return matching


def build_leaf_question_concepts(
    dataset: str,
    metadata: Mapping[str, Any],
    qids: Iterable[Any] | None = None,
    xes_root_concept_id: str = "865",
    nips_root_subject_id: int = 3,
    route_separator: str = "----",
) -> dict[Any, list[Any]]:
    """Build question -> leaf concept ids for XES, NIPS, or DBE metadata."""

    dataset_key = dataset.lower()

    def add_unique(values: list[Any], value: Any) -> None:
        if value is None:
            return
        if value not in values:
            values.append(value)

    if dataset_key in {"xes", "xes3g5m"}:
        ques_info = metadata["ques_info"]
        question_ids = list(qids if qids is not None else ques_info.keys())
        out: dict[Any, list[Any]] = {}
        for qid in question_ids:
            item = _lookup(ques_info, qid, {}) or {}
            leaves: list[Any] = []
            for route in item.get("kc_routes", []) or []:
                nodes = [node.strip() for node in str(route).split(route_separator) if node.strip()]
                nodes = [node for node in nodes if node != str(xes_root_concept_id)]
                if nodes:
                    add_unique(leaves, nodes[-1])
            out[qid] = leaves
        return out

    if dataset_key in {"nips", "nips34"}:
        question_metadata = metadata["question_metadata"]
        subject_metadata = metadata.get("subject_metadata")
        question_id_col = "QuestionId"
        subject_id_col = "SubjectId"
        question_ids = list(qids if qids is not None else question_metadata[question_id_col].unique())
        rows = question_metadata.set_index(question_id_col).to_dict("index")
        parent: dict[str, str] = {}
        if subject_metadata is not None and "ParentId" in subject_metadata.columns:
            for _, subject_row in subject_metadata.iterrows():
                subject = str(subject_row[subject_id_col])
                subject = subject[:-2] if subject.endswith(".0") else subject
                raw_parent = subject_row["ParentId"]
                if pd.isna(raw_parent):
                    continue
                parent_id = str(raw_parent)
                parent_id = parent_id[:-2] if parent_id.endswith(".0") else parent_id
                if parent_id and parent_id.lower() not in {"nan", "none", "null"}:
                    parent[subject] = parent_id

        def ancestors(subject: str) -> set[str]:
            out: set[str] = set()
            current = subject
            while current in parent and parent[current] and parent[current] != current:
                current = parent[current]
                out.add(current)
            return out

        out = {}
        for qid in question_ids:
            row = _lookup(rows, qid, {})
            raw_subjects = row.get(subject_id_col, []) if row else []
            if isinstance(raw_subjects, str):
                subjects = [x.strip() for x in raw_subjects.strip("[]").split(",") if x.strip()]
            elif isinstance(raw_subjects, Iterable):
                subjects = list(raw_subjects)
            else:
                subjects = []
            unique_subjects: list[str] = []
            for raw_subject in subjects:
                try:
                    subject = str(int(raw_subject))
                except (TypeError, ValueError):
                    continue
                if subject == str(nips_root_subject_id) or subject in unique_subjects:
                    continue
                unique_subjects.append(subject)
            ancestor_subjects: set[str] = set()
            for subject in unique_subjects:
                ancestor_subjects.update(ancestors(subject))
            deepest = [subject for subject in unique_subjects if subject not in ancestor_subjects]
            out[qid] = [int(subject) for subject in (deepest or unique_subjects)]
        return out

    if dataset_key in {"dbe", "dbe-kt22", "dbe_kt22"}:
        df_q_kc = metadata["df_q_kc"]
        question_ids = list(qids if qids is not None else df_q_kc["question_id"].unique())
        out = {qid: [] for qid in question_ids}
        for _, row in df_q_kc.iterrows():
            qid = row["question_id"]
            if str(qid) not in {str(x) for x in question_ids}:
                continue
            raw_kcs = row["knowledgecomponent_id"]
            if isinstance(raw_kcs, (list, tuple, set, np.ndarray, pd.Series)):
                kcs = list(raw_kcs)
            else:
                kcs = [raw_kcs]
            key = next((x for x in question_ids if str(x) == str(qid)), qid)
            out.setdefault(key, [])
            for kc in kcs:
                add_unique(out[key], kc)
        return out

    raise ValueError(f"Unsupported dataset for leaf concept extraction: {dataset}")


def extract_content(qid: Any, context: KnowledgeContext, unknown: str = "") -> str:
    """Dataset-neutral content lookup for XES/NIPS-style prompts."""

    return context.content(qid, unknown=unknown)


def extract_concepts(
    qid: Any,
    context: KnowledgeContext | Any | None = None,
    dataset: str | None = None,
    as_string: bool = True,
    separator: str = ", ",
    **kwargs: Any,
) -> str | list[str]:
    """Dataset-neutral concept lookup for DBE/NIPS/XES-style prompts.

    Preferred usage is ``extract_concepts(qid, context)`` with a
    ``KnowledgeContext``. Older dataset-specific metadata can be passed with
    ``dataset="nips"``, ``dataset="xes"``, or ``dataset="dbe"``.
    """

    def flatten(values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, (list, tuple, set)):
            flattened: list[str] = []
            for value in values:
                flattened.extend(flatten(value))
            return flattened
        return [str(values)]

    # ``importlib.reload(utils)`` creates a new KnowledgeContext class object.
    # Context instances built before the reload are still valid, but a strict
    # isinstance check against the newly created class rejects them.  Accept
    # the context protocol as well so notebook reloads do not break retrieval.
    if isinstance(context, KnowledgeContext) or callable(getattr(context, "concept_names", None)):
        names = flatten(context.concept_names(qid))
    else:
        if dataset is None:
            if "df_question_metadata" in kwargs or "question_metadata" in kwargs:
                dataset = "nips"
            elif "qid_to_routes" in kwargs:
                dataset = "xes"
            elif "df_q_kc_matching" in kwargs:
                dataset = "dbe"
            elif context is not None:
                raise TypeError(
                    "dataset must be provided when context is not a KnowledgeContext"
                )
            else:
                names = []

        if dataset is not None:
            dataset_key = dataset.lower()
            if dataset_key in {"nips", "nips34"}:
                names = _extract_concept_list_nips(
                    qid,
                    df_question_metadata=kwargs.pop(
                        "df_question_metadata", kwargs.pop("question_metadata", context)
                    ),
                    subject_metadata=kwargs.pop("subject_metadata", None),
                    question_id_col=kwargs.pop("question_id_col", "QuestionId"),
                    subject_id_col=kwargs.pop("subject_id_col", "SubjectId"),
                    subject_name_col=kwargs.pop("subject_name_col", "Name"),
                    root_subject_id=kwargs.pop("root_subject_id", 3),
                )
            elif dataset_key in {"xes", "xes3g5m"}:
                names = _extract_concept_list_xes(
                    qid,
                    qid_to_routes=kwargs.pop("qid_to_routes", context),
                    route_separator=kwargs.pop("route_separator", "----"),
                )
            elif dataset_key in {"dbe", "dbe-kt22", "dbe_kt22"}:
                names = _extract_concept_list_dbe(
                    qid,
                    df_q_kc_matching=kwargs.pop("df_q_kc_matching", context),
                    df_kcs=kwargs.pop("df_kcs", None),
                    question_id_col=kwargs.pop("question_id_col", "question_id"),
                    kc_id_col=kwargs.pop("kc_id_col", "knowledgecomponent_id"),
                    kc_name_col=kwargs.pop("kc_name_col", "name"),
                    use_masked=kwargs.pop("use_masked", False),
                    masked_col=kwargs.pop("masked_col", "masked_concept"),
                )
            else:
                raise ValueError(f"Unsupported dataset for concept extraction: {dataset}")

    return separator.join(names) if as_string else names


def _extract_concept_list_nips(
    qid: Any,
    df_question_metadata: pd.DataFrame,
    subject_metadata: pd.DataFrame | None = None,
    question_id_col: str = "QuestionId",
    subject_id_col: str = "SubjectId",
    subject_name_col: str = "Name",
    root_subject_id: int = 3,
) -> list[str]:
    if df_question_metadata is None:
        raise ValueError("df_question_metadata is required for dataset='nips'")

    row = df_question_metadata[df_question_metadata[question_id_col] == int(qid)]
    if len(row) == 0:
        return []

    raw_subjects = row[subject_id_col].iloc[0]
    if isinstance(raw_subjects, str):
        subject_ids = [
            int(x.strip())
            for x in raw_subjects.strip("[]").split(",")
            if x.strip()
        ]
    else:
        subject_ids = [int(x) for x in raw_subjects]

    subject_ids = [sid for sid in subject_ids if sid != root_subject_id]
    if subject_metadata is None:
        return [str(sid) for sid in subject_ids]

    names = []
    for sid in subject_ids:
        matched = subject_metadata[subject_metadata[subject_id_col] == sid]
        if len(matched) == 0:
            names.append(str(sid))
        else:
            names.append(str(matched[subject_name_col].iloc[0]))
    return names


def _extract_concept_list_xes(
    qid: Any,
    qid_to_routes: Mapping[Any, Any],
    route_separator: str = "----",
) -> list[str]:
    if qid_to_routes is None:
        raise ValueError("qid_to_routes is required for dataset='xes'")

    routes = qid_to_routes.get(str(qid))
    if routes is None:
        try:
            routes = qid_to_routes.get(int(qid), [])
        except (TypeError, ValueError):
            routes = []
    if routes is None:
        return []
    if isinstance(routes, str):
        routes = [routes]

    merged = []
    for route in routes:
        path = str(route).split(route_separator)
        for node in path:
            node = node.strip()
            if node and node not in merged:
                merged.append(node)
    return merged


def _extract_concept_list_dbe(
    qid: Any,
    df_q_kc_matching: pd.DataFrame,
    df_kcs: pd.DataFrame | None = None,
    question_id_col: str = "question_id",
    kc_id_col: str = "knowledgecomponent_id",
    kc_name_col: str = "name",
    use_masked: bool = False,
    masked_col: str = "masked_concept",
) -> list[str]:
    if df_q_kc_matching is None:
        raise ValueError("df_q_kc_matching is required for dataset='dbe'")

    row = df_q_kc_matching[
        df_q_kc_matching[question_id_col].astype(str) == str(qid)
    ]
    if len(row) == 0:
        return []

    kcs = row[kc_id_col].iloc[0]
    if not isinstance(kcs, (list, tuple, set)):
        kcs = [kcs]
    if df_kcs is None:
        return [str(kc) for kc in kcs]

    concepts = []
    for kc in kcs:
        matched = df_kcs[df_kcs["id"] == kc]
        if len(matched) == 0:
            concepts.append(str(kc))
            continue

        name = str(matched[kc_name_col].iloc[0])
        if use_masked and masked_col in matched.columns:
            masked = str(matched[masked_col].iloc[0])
            concepts.append(f"{masked} ({name})")
        else:
            concepts.append(name)
    return concepts


def extract_concepts_nips(qid: Any, *args: Any, **kwargs: Any) -> str:
    return extract_concepts(qid, *args, dataset="nips", **kwargs)


def extract_concepts_xes(qid: Any, *args: Any, **kwargs: Any) -> str:
    return extract_concepts(qid, *args, dataset="xes", **kwargs)


def extract_concepts_dbe(qid: Any, *args: Any, **kwargs: Any) -> str:
    return extract_concepts(qid, *args, dataset="dbe", **kwargs)


def build_content_context(
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    question_content: Mapping[Any, Any],
    question_to_concepts: Mapping[Any, Sequence[Any]] | Callable[[Any], Sequence[Any]] | None = None,
    concept_names: Mapping[Any, Any] | Callable[[Any], str] | None = None,
    concept_descriptions: Mapping[Any, Any] | Callable[[Any], str] | None = None,
    unknown_label: str = "unknown",
) -> KnowledgeContext:
    """Create a content-first context for XES/NIPS-style datasets."""

    def question_text(qid: Any) -> str:
        return str(_lookup(question_content, qid, ""))

    def q_to_concepts(qid: Any) -> Sequence[Any]:
        if question_to_concepts is None:
            return []
        if callable(question_to_concepts):
            return question_to_concepts(qid)
        return _lookup(question_to_concepts, qid, [])

    def concept_name(concept: Any) -> str:
        if concept_names is None:
            return str(concept)
        if callable(concept_names):
            return concept_names(concept)
        return str(_lookup(concept_names, concept, concept))

    def concept_description(concept: Any) -> str:
        if concept_descriptions is None:
            return ""
        if callable(concept_descriptions):
            return concept_descriptions(concept)
        return str(_lookup(concept_descriptions, concept, "")).strip()

    return KnowledgeContext(
        difficulty_dict=difficulty_dict,
        question_to_concepts=q_to_concepts if question_to_concepts is not None else None,
        concept_name=concept_name if concept_names is not None else None,
        concept_description=concept_description if concept_descriptions is not None else None,
        question_text=question_text,
        unknown_label=unknown_label,
        default_representation="content",
    )


def build_xes_context(
    difficulty_dict,
    ques_info,
    kc_map,
    root_concept_id="865",
    route_separator="----",
    ):
    def normalize_qid(qid):
      try:
          return str(int(qid))
      except (TypeError, ValueError):
          return str(qid)
    
    def merge_kc_routes(routes):
      merged = []
      for route in routes or []:
          for node in str(route).split(route_separator):
              node = node.strip()
              if node and node != root_concept_id and node not in merged:
                  merged.append(node)
      return merged
    
    def question_text(qid):
      item = ques_info.get(normalize_qid(qid), {})
      if not item:
          return ""
      text = str(item.get("content", ""))
      options = item.get("options", {}) or {}
      if options:
          text += "; Select One. "
          for opt, value in options.items():
              text += f"{opt}:{value} "
      return text
    
    def question_to_concepts(qid):
      item = ques_info.get(normalize_qid(qid), {})
      return merge_kc_routes(item.get("kc_routes", [])) if item else []
    
    def concept_name(concept):
      return str(kc_map.get(str(concept), concept))
    
    return KnowledgeContext(
      difficulty_dict=difficulty_dict,
      question_to_concepts=question_to_concepts,
      concept_name=concept_name,
      question_text=question_text,
      unknown_label="unknown",
      default_representation="content",
    )


def build_nips_context(
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    question_content: Mapping[Any, Any],
    question_metadata: pd.DataFrame,
    subject_metadata: pd.DataFrame | None = None,
    question_id_col: str = "QuestionId",
    subject_id_col: str = "SubjectId",
    subject_name_col: str = "Name",
    root_subject_id: int = 3,
) -> KnowledgeContext:
    """Create a NIPS context from question/subject metadata."""

    metadata = question_metadata.set_index(question_id_col).to_dict("index")
    subject_names = {}
    if subject_metadata is not None:
        subject_names = subject_metadata.set_index(subject_id_col)[subject_name_col].to_dict()

    def parse_subjects(raw: Any) -> list[int]:
        if isinstance(raw, str):
            values = [x.strip() for x in raw.strip("[]").split(",") if x.strip()]
        elif isinstance(raw, Iterable):
            values = list(raw)
        else:
            return []
        subjects = []
        for value in values:
            try:
                subject = int(value)
            except (TypeError, ValueError):
                continue
            if subject != root_subject_id:
                subjects.append(subject)
        return subjects

    def question_to_concepts(qid: Any) -> Sequence[Any]:
        row = _lookup(metadata, qid)
        return [] if row is None else parse_subjects(row.get(subject_id_col, []))

    def concept_name(concept: Any) -> str:
        return str(_lookup(subject_names, concept, concept))

    return build_content_context(
        difficulty_dict=difficulty_dict,
        question_content=question_content,
        question_to_concepts=question_to_concepts,
        concept_names=concept_name,
    )


def build_dbe_context(
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    df_q_kc_matching: pd.DataFrame,
    df_kcs: pd.DataFrame,
    full_q_df: pd.DataFrame | None = None,
    ) -> KnowledgeContext:
    """Create a concept-first KnowledgeContext from DBE metadata dataframes."""
    
    def normalize_kcs(value: Any) -> list[Any]:
      if value is None or (isinstance(value, float) and pd.isna(value)):
          return []
      if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
          return list(value)
      return [value]
    
    q_to_kcs: dict[str, list[Any]] = {}
    for _, row in df_q_kc_matching.iterrows():
      qid = str(row["question_id"])
      q_to_kcs.setdefault(qid, [])
      for kc in normalize_kcs(row["knowledgecomponent_id"]):
          if kc not in q_to_kcs[qid]:
              q_to_kcs[qid].append(kc)
    
    kc_by_id = df_kcs.set_index("id").to_dict("index")
    q_text = full_q_df.set_index("id")["question_text"].to_dict() if full_q_df is not None else {}
    
    def question_to_concepts(qid: Any) -> Sequence[Any]:
      return _lookup(q_to_kcs, qid, [])
    
    def concept_name(kc: Any) -> str:
      row = _lookup(kc_by_id, kc)
      if row is None:
          return str(kc)
      return str(row.get("name", kc))
    
    def concept_description(kc: Any) -> str:
      row = _lookup(kc_by_id, kc)
      if row is None:
          return ""
      return str(row.get("description", "")).strip()
    
    def question_text(qid: Any) -> str:
      return str(_lookup(q_text, qid, ""))
    
    def concept_mask(kc: Any) -> str:
      row = _lookup(kc_by_id, kc)
      if row is None:
          return str(kc)
      return str(row.get("masked_concept", kc))
    
    return KnowledgeContext(
      difficulty_dict=difficulty_dict,
      question_to_concepts=question_to_concepts,
      concept_name=concept_name,
      concept_description=concept_description,
      question_text=question_text,
      concept_mask=concept_mask,
      unknown_label="Uncertain",
      default_representation="concept",
    )


def _resolve_representation(context: KnowledgeContext, representation: Representation) -> Literal["content", "concept"]:
    if representation == "auto":
        return context.default_representation
    return representation


def _question_descriptor(qid: Any, context: KnowledgeContext, representation: Representation = "auto") -> str:
    mode = _resolve_representation(context, representation)
    if mode == "content":
        content = extract_content(qid, context, unknown="")
        if content:
            return f"Content: {content}"
        concepts = extract_concepts(qid, context)
        return f"Concept(s): {concepts}" if concepts else ""

    concepts = extract_concepts(qid, context)
    return f"Concept(s): {concepts}" if concepts else ""


def all_qids_to_string(
    qids: Iterable[Any],
    questions: Sequence[Any],
    responses: Sequence[Any] | None,
    context: KnowledgeContext,
    concepts: Sequence[Any] | None = None,
    representation: Representation = "auto",
    return_concept_descriptions: bool | None = None,
) -> str | tuple[str, str]:
    """Format the question database block for XES/NIPS/DBE prompts.

    XES/NIPS generally use ``representation="content"`` and return one string.
    DBE uses ``representation="concept"`` and returns ``(questions, concepts)``.
    The ``concepts`` argument is accepted for compatibility with older XES/NIPS
    notebook signatures, but the context is the source of metadata.
    """

    _ = concepts, responses
    all_qids = sorted(set([str(q) for q in qids] + [str(q) for q in questions]))
    unique_kc: list[Any] = []
    questions_data = "[Questions Database]\n"
    for qq in all_qids:
        kcs = context.concepts(qq)
        unique_kc += kcs
        top_percent = f"({context.label(qq)})" if get_peer_rate(qq, context.difficulty_dict) is not None else ""
        peer_rate = f"Peer answer rate: {context.peer_rate(qq)} "
        questions_data += f"{_question_descriptor(qq, context, representation)} | {peer_rate} {top_percent}\n"

    if return_concept_descriptions is None:
        return_concept_descriptions = _resolve_representation(context, representation) == "concept"
    if not return_concept_descriptions:
        return questions_data

    concept_data = "[Concept Descriptions]\n"
    seen = set()
    for kc in unique_kc:
        if kc in seen:
            continue
        seen.add(kc)
        name = context.concept_name(kc) if context.concept_name else str(kc)
        desc = context.concept_description(kc) if context.concept_description else ""
        concept_data += f"- {name}: {desc}\n"
    return questions_data, concept_data


def history_to_string(
    questions: Sequence[Any],
    responses: Sequence[Any],
    context: KnowledgeContext,
    include_question_id: bool = True,
    include_concepts: bool = True,
    representation: Representation = "auto",
    group_by_correctness: bool = False,
) -> str:
    """Format student history for prompts."""

    parts = []
    for idx, (key, value) in enumerate(zip(questions, responses)):
        is_correct = "Correct" if int(value) == 1 else "Incorrect"
        peer_rate = f"Peer answer rate: {context.peer_rate(key)} "
        top_percent = f"({context.label(key)})" if get_peer_rate(key, context.difficulty_dict) is not None else ""
        fields = []
        mode = _resolve_representation(context, representation)
        descriptor = _question_descriptor(key, context, representation)
        if mode == "content":
            fields.append(descriptor)
        elif include_concepts:
            fields.append(descriptor)
        # if include_concepts and mode == "concept":
        #     concept_text = ", ".join(context.concept_names(key))
        #     if concept_text:
        #         fields.append(f"Concept(s): {concept_text}")
        fields.extend([peer_rate + top_percent, f"Correctness: {is_correct}"])
        parts.append((is_correct, f"[#{idx + 1}] " + " | ".join(fields)))

    if group_by_correctness:
        correct = [text for label, text in parts if label == "Correct"]
        incorrect = [text for label, text in parts if label == "Incorrect"]
        return "Correct Attempts:\n" + "\n".join(correct) + "\n\nIncorrect Attempts:\n" + "\n".join(incorrect)
    return "\n".join(text for _, text in parts)

def history_to_string_no_concept(
    questions: Sequence[Any],
    responses: Sequence[Any],
    context: KnowledgeContext,
    representation: Representation = "auto",
    group_by_correctness=False,
    ):
  parts = []

  for idx, (qid, response) in enumerate(zip(questions, responses)):
      is_correct = "Correct" if int(response) == 1 else "Incorrect"
      peer_rate = f"Peer answer rate: {context.peer_rate(qid)} "
      top_percent = f"({context.label(qid)})" if get_peer_rate(qid, context.difficulty_dict) is not None else ""

      text = (
          f"[#{idx + 1}] "
          f"{peer_rate}{top_percent} | "
          f"Correctness: {is_correct}"
      )
      parts.append((is_correct, text))

  if group_by_correctness:
      correct = [text for label, text in parts if label == "Correct"]
      incorrect = [text for label, text in parts if label == "Incorrect"]
      return "Correct Attempts:\n" + "\n".join(correct) + "\n\nIncorrect Attempts:\n" + "\n".join(incorrect)

  return "\n".join(text for _, text in parts)



def history_to_string_no_ar(
    questions: Sequence[Any],
    responses: Sequence[Any],
    context: KnowledgeContext,
    include_question_id: bool = True,
    include_concepts: bool = True,
    representation: Representation = "auto",
    group_by_correctness: bool = False,
) -> str:
    """Format student history for prompts."""

    parts = []
    for idx, (key, value) in enumerate(zip(questions, responses)):
        is_correct = "Correct" if int(value) == 1 else "Incorrect"
        fields = []
        mode = _resolve_representation(context, representation)
        descriptor = _question_descriptor(key, context, representation)
        if mode == "content" or not include_question_id:
            fields.append(descriptor)
        elif include_concepts:
            fields.append(descriptor)
            
        fields.extend([f"Correctness: {is_correct}"])
        parts.append((is_correct, f"[#{idx + 1}] " + " | ".join(fields)))

    if group_by_correctness:
        correct = [text for label, text in parts if label == "Correct"]
        incorrect = [text for label, text in parts if label == "Incorrect"]
        return "Correct Attempts:\n" + "\n".join(correct) + "\n\nIncorrect Attempts:\n" + "\n".join(incorrect)
    return "\n".join(text for _, text in parts)

def pattern_to_string(
      questions: Sequence[Any],
      responses: Sequence[Any],
      context: KnowledgeContext,
      difficulty_dict: Mapping[Any, Mapping[str, float]] | None = None,
  ) -> str:
      """Compact chronological performance/difficulty pattern summary."""

      if difficulty_dict is None:
          difficulty_dict = context.difficulty_dict

      performance = []
      difficulties = []

      for qid, response in zip(questions, responses):
          is_correct = "Correct" if int(response) == 1 else "Incorrect"
          performance.append(is_correct)

          rate = get_peer_rate(qid, difficulty_dict)
          peer_rate = "unknown" if rate is None else f"{rate:.3f}"
          label = decide_label(qid, difficulty_dict) if rate is not None else "unknown"
          difficulties.append(f"{peer_rate} ({label})")

      text = "- Performance History (oldest -> most recent):\n"
      text += ", ".join(performance)
      text += "\n- Question Difficulty History (peer answer rate; difficulty label):\n"
      text += ", ".join(difficulties)

      return text


def read_prob(target_id: Any, context: KnowledgeContext) -> str:
    if context.question_text is None:
        raise ValueError("KnowledgeContext.question_text is not configured")
    return context.question_text(target_id)


def build_recency_weighted_vector(state_seq: Sequence[Any], alpha: float = 0.9) -> dict[Any, float]:
    """Build a recency-weighted count vector from an oldest-to-newest state sequence."""

    vec: dict[Any, float] = {}
    length = len(state_seq)
    for idx, state in enumerate(state_seq):
        weight = alpha ** (length - 1 - idx)
        vec[state] = vec.get(state, 0.0) + weight
    return vec


def build_recency_weighted_vector_incor(
    state_seq: Sequence[Any],
    alpha: float = 0.9,
    incorrect_weight: float = 1.5,
) -> dict[Any, float]:
    """Build a recency-weighted vector with optional extra weight for incorrect states."""

    vec: dict[Any, float] = {}
    length = len(state_seq)
    for idx, state in enumerate(state_seq):
        weight = alpha ** (length - 1 - idx)
        correctness = state[-2] if isinstance(state, tuple) and len(state) >= 2 else None
        if correctness in {"0", 0, "Incorrect"}:
            weight *= incorrect_weight
        vec[state] = vec.get(state, 0.0) + weight
    return vec
