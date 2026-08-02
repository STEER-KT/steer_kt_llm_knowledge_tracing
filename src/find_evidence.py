"""Peer and self evidence retrieval for adaptive-KT prompting.

Use factory helpers to bind dataset metadata once, then pass the returned functions
into run_zero_shot_prediction(...):

    from find_evidence import make_peer_seq_factory, make_find_peers, make_find_self

    make_peer_seq_fn = make_peer_seq_factory(context, difficulty_dict)
    find_peers_fn = make_find_peers(context, difficulty_dict)
    find_self_fn = make_find_self(context, difficulty_dict)

    run_zero_shot_prediction(
        ...,
        make_peer_seq_fn=make_peer_seq_fn,
        find_peers_fn=find_peers_fn,
        find_self_fn=find_self_fn,
        LLM_train=LLM_train,
    )
"""

from __future__ import annotations

from itertools import chain
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from utils import *


DIFF_RANK = {
    "veryhard": 0,
    "hard": 1,
    "mid": 2,
    "easy": 3,
    "veryeasy": 4,
}


def _qid(q: Any) -> int:
    return int(q)


def _norm_label(label: Any) -> str:
    return str(label).replace(" ", "").replace("_", "").lower()


def _rate(q: Any, difficulty_dict: Mapping[Any, Mapping[str, float]]) -> float | None:
    return get_peer_rate(q, difficulty_dict)


def _rate_to_str(rate: float | None) -> str:
    return "unknown" if rate is None else f"{rate:.4f}"


def _diff_label(q: Any, difficulty_dict: Mapping[Any, Mapping[str, float]]) -> str:
    try:
        return decide_label(_qid(q), difficulty_dict)
    except Exception:
        return "unknown"


def get_position_weights(length: int) -> list[float]:
    base = [1.0, 1.0, 1.2, 1.5, 2.0]
    if length <= len(base):
        return base[-length:]
    return [1.0] * (length - len(base)) + base


def weighted_available_average(
    components: Sequence[tuple[float | None, float]],
    expected_total_weight: float | None = None,
) -> tuple[float | None, float]:
    used_weight = 0.0
    score_sum = 0.0
    if expected_total_weight is None:
        expected_total_weight = sum(weight for _, weight in components)

    for value, weight in components:
        if value is None:
            continue
        used_weight += weight
        score_sum += value * weight

    if used_weight <= 0:
        return None, 0.0
    score = score_sum / used_weight
    coverage = used_weight / expected_total_weight if expected_total_weight > 0 else 0.0
    return score, coverage


def question_concepts(
    qid: Any,
    context: KnowledgeContext,
    concept_route_fn: Callable[[Any], Sequence[Any]] | None = None,
) -> list[Any]:
    values = concept_route_fn(qid) if concept_route_fn is not None else context.concepts(qid)
    values = list(values or [])
    return values if values else [qid]


def question_concept_set(
    qid: Any,
    context: KnowledgeContext,
    concept_route_fn: Callable[[Any], Sequence[Any]] | None = None,
    ) -> set[str]:
    return {str(concept) for concept in question_concepts(qid, context, concept_route_fn)}


def has_min_concept_overlap(
    qid: Any,
    target_qid: Any,
    target_concepts: set[str],
    context: KnowledgeContext,
    concept_route_fn: Callable[[Any], Sequence[Any]] | None = None,
    min_overlap_ratio: float = 0.5,
    ) -> bool:
    try:
        if _qid(qid) == _qid(target_qid):
            return False
    except (TypeError, ValueError):
        if str(qid) == str(target_qid):
            return False
    
    if not target_concepts:
        return False
    
    candidate_concepts = question_concept_set(qid, context, concept_route_fn)
    if not candidate_concepts:
        return False
    
    overlap_ratio = len(target_concepts.intersection(candidate_concepts)) / len(target_concepts)
    return overlap_ratio >= min_overlap_ratio


def make_peer_seq(
    questions: Sequence[Any],
    responses: Sequence[Any],
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    concept_route_fn: Callable[[Any], Sequence[Any]] | None = None,
) -> set[tuple[Any, str, str, int]]:
    target_seq: set[tuple[Any, str, str, int]] = set()
    counts: dict[tuple[Any, str, Any], int] = {}

    for q, r in zip(questions, responses):
        q_int = _qid(q)
        resp = str(int(r))
        diff = _diff_label(q_int, difficulty_dict)
        for concept_id in question_concepts(q_int, context, concept_route_fn):
            key = (concept_id, diff, r)
            counts[key] = counts.get(key, 0) + 1
            target_seq.add((concept_id, diff, resp, counts[key]))
    return target_seq


def get_peer_sequence_jaccard(
    row: pd.Series,
    target_qid: Any,
    window_size: int,
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    concept_route_fn: Callable[[Any], Sequence[Any]] | None = None,
    relaxed_filtering: bool = False,
    min_concept_overlap: float = 0.5,
    overlap_question_map: Mapping[Any, Sequence[Any]] | None = None,
) -> pd.Series:
    try:
        target_qid = _qid(target_qid)
        
        if relaxed_filtering and overlap_question_map is not None:
            matched_qids = overlap_question_map.get(target_qid)
            if matched_qids is None:
                matched_qids = overlap_question_map.get(str(target_qid), [])
            matched_qids = {str(q) for q in matched_qids}
            all_indices = [i for i, q in enumerate(row["questions"]) if str(q) in matched_qids]
        elif relaxed_filtering:
            target_concepts = question_concept_set(target_qid, context, concept_route_fn)
            all_indices = [i
              for i, q in enumerate(row["questions"])
              if has_min_concept_overlap(
                  q,
                  target_qid,
                  target_concepts,
                  context,
                  concept_route_fn,
                  min_concept_overlap,
              )
            ]
        else:
            all_indices = [i for i, q in enumerate(row["questions"]) if _qid(q) == target_qid]
        target_idx = all_indices[-1]
        matched_question_id = row["questions"][target_idx]
        if target_idx == 0:
            return pd.Series([None, None, None, None, None, None, None])

        start_idx = max(0, target_idx - window_size)
        sub_questions = row["questions"][start_idx:target_idx]
        sub_corrects = row["responses"][start_idx:target_idx]
        sub_answer_rate = round(float(row["responses"][:target_idx].count(1)) / target_idx, 4)
        sequence_text = make_peer_seq(sub_questions, sub_corrects, context, difficulty_dict, concept_route_fn)
        target_correct = int(row["responses"][target_idx])
        return pd.Series([
            sequence_text,
            target_correct,
            sub_questions,
            sub_corrects,
            sub_answer_rate,
            matched_question_id,
            target_idx,
        ])
    except (ValueError, IndexError, KeyError, TypeError):
        return pd.Series([None, None, None, None, None, None, None])


def search_jaccard_refined(
    df: pd.DataFrame,
    target_qid: Any,
    window: int,
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    concept_route_fn: Callable[[Any], Sequence[Any]] | None = None,
    relaxed_filtering = False,
    min_concept_overlap: float = 0.5,
    overlap_question_map: Mapping[Any, Sequence[Any]] | None = None,
    
) -> pd.DataFrame:
    df = df.copy()
    df[[
        "peer_seq_set",
        "target_result",
        "sub_questions",
        "sub_corrects",
        "sub_answer_rate",
        "matched_question_id",
        "matched_timestamp",
    ]] = df.apply(
        lambda row: get_peer_sequence_jaccard(
            row,
            target_qid,
            window,
            context,
            difficulty_dict,
            concept_route_fn,
            relaxed_filtering,
            min_concept_overlap,
            overlap_question_map,
        ),
        axis=1,
    )
    return df.dropna(subset=["peer_seq_set"])


def _weighted_jaccard_with_incorrect(target_set: set[Any], peer_set: set[Any]) -> tuple[float, float]:
    intersection = target_set.intersection(peer_set)
    union = target_set.union(peer_set)
    score = len(intersection) / len(union) if union else 0.0
    intersection_score = 0.0
    union_score = 0.0
    for state in union:
        correctness = state[-2]
        weight = 2.0 if correctness in {"0", 0, "Incorrect"} else 1.0
        union_score += weight
        if state in intersection:
            intersection_score += weight
    score_inc = intersection_score / union_score if union_score > 0 else 0.0
    return score, score_inc


def get_top_k_jaccard(
    target_set: set[Any],
    candidates_df: pd.DataFrame,
    threshold: float | None,
) -> list[tuple[Any, float, float, int, list[Any], list[Any], set[Any], Any]]:
    scores = []
    for row in candidates_df.itertuples():
        score, score_inc = _weighted_jaccard_with_incorrect(target_set, row.peer_seq_set)
        if threshold is not None and score < threshold:
            continue
        uid = getattr(row, "uid", getattr(row, "user_id", getattr(row, "student_id", None)))
        scores.append((
            uid,
            score,
            score_inc,
            row.target_result,
            row.sub_questions,
            row.sub_corrects,
            row.peer_seq_set,
            row.matched_question_id,
            row.Index,
            int(row.matched_timestamp),
        ))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def remove_conflicting_histories(df_scores: pd.DataFrame) -> pd.DataFrame:
    grouped: dict[tuple[tuple[Any, ...], tuple[Any, ...]], list[Any]] = {}
    for row in df_scores.itertuples():
        key = (tuple(row.questions), tuple(row.responses))
        grouped.setdefault(key, []).append(row)

    filtered_rows = []
    for rows in grouped.values():
        labels = {row.target_result for row in rows}
        if len(labels) > 1:
            continue
        filtered_rows.append(sorted(rows, key=lambda x: x.score, reverse=True)[0])
    return pd.DataFrame(filtered_rows)


def RAG_jaccard_peers(
    target_seq: set[Any],
    peer_df: pd.DataFrame,
    threshold: float | None = None,
    top_per_label: int = 1,
) -> tuple[set[Any], list[list[Any]], pd.DataFrame]:
    scores = get_top_k_jaccard(target_seq, peer_df, threshold)
    if not scores:
        return set(), [], pd.DataFrame()

    df_scores = pd.DataFrame(
        scores,
        columns=[
            "uid",
            "score",
            "score_inc",
            "target_result",
            "questions",
            "responses",
            "peer_set",
            "matched_question_id",
            "train_row_index",
            "matched_timestamp",
        ],
    )
    df_scores = remove_conflicting_histories(df_scores)
    if df_scores.empty:
        return set(), [], df_scores
    df_scores = df_scores.sort_values("score", ascending=False)
    top_pool = df_scores.head(30).copy()
    correct_peers = top_pool[top_pool["target_result"] == 1].head(top_per_label)
    incorrect_peers = top_pool[top_pool["target_result"] == 0].head(top_per_label)
    selected = pd.concat([correct_peers, incorrect_peers]).sort_values("score_inc", ascending=False)

    qids = []
    others = []
    for uid, score, score_inc, target_result, questions, responses, peer_set, matched_question_id in selected[
        ["uid", "score", "score_inc", "target_result", "questions", "responses", "peer_set", "matched_question_id"]
    ].itertuples(index=False, name=None):
        qids.append(questions)
        qids.append([matched_question_id])
        others.append([
            str(uid),
            questions,
            responses,
            int(target_result),
            float(score),
            float(score_inc),
            matched_question_id,
        ])
    return set(chain.from_iterable(qids)), others, selected


def render_matched_question_info(
    qid: Any,
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
) -> str:
    # Values returned through the pandas apply/concat path can be coerced to
    # floats (for example, DBE question 219 becomes 219.0).  DBE's
    # question-to-concept mapping uses canonical integer/string IDs, so the
    # float form can still find integer-keyed question text while failing the
    # string-keyed concept lookup.  Normalize once before every metadata lookup.
    try:
        qid = _qid(qid)
    except (TypeError, ValueError):
        pass

    try:
        content = read_prob(qid, context) if context.question_text is not None else ""
    except Exception:
        content = ""
    concepts = extract_concepts(qid, context)
    peer_rate = get_peer_rate(qid, difficulty_dict)
    peer_rate_text = "unknown" if peer_rate is None else f"{peer_rate:.4f}"
    difficulty = _diff_label(qid, difficulty_dict)

    lines = [
        ">>>> Matched peer question",
    ]
    if content:
        lines.append(f"- Content: {content}")
    if concepts:
        lines.append(f"- Concept: {concepts}")
    lines.append(f"- Peer Answer Rate: {peer_rate_text} ({difficulty})")
    return "\n".join(lines)


def render_peer_evidence(
    peers: Sequence[Sequence[Any]],
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    representation: str = "auto",
    relaxed_filtering: bool = False,
) -> str:
    if not peers:
        return "No peers who have similar behavioral patterns with the TARGET STUDENT."
    text = ""
    for peer in peers:
        uid, questions, responses, target_result, score, score_inc = peer[:6]
        matched_question_id = peer[6] if len(peer) > 6 else None
        text += f"**Student_ID {uid}**\n"
        text += history_to_string(questions, responses, context, representation=representation)
        # text += f"\nSimilarity: {score:.3f}; incorrect-weighted similarity: {score_inc:.3f}\n"
        if relaxed_filtering and matched_question_id is not None:
            text += f"\n\n{render_matched_question_info(matched_question_id, context, difficulty_dict)}"
        outcome_label = "Outcome on the matched peer question" if relaxed_filtering else "Answer for the TARGET QUESTION"
        text += f"\n\n>>>> {outcome_label}: {'Correct' if target_result == 1 else 'Incorrect'}\n\n"
    return text


def find_peers(
    target_seq: set[Any],
    df_train: pd.DataFrame,
    target_id: Any,
    alpha: float,
    window: int,
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    representation: str = "auto",
    concept_route_fn: Callable[[Any], Sequence[Any]] | None = None,
    relaxed_filtering = False,
    min_concept_overlap: float = 0.5,
    overlap_question_map: Mapping[Any, Sequence[Any]] | None = None,
    top_per_label: int = 1,
) -> tuple[str, pd.DataFrame]:
    _ = alpha
    target_int = _qid(target_id)
    if relaxed_filtering and overlap_question_map is not None:
        matched_qids = overlap_question_map.get(target_int)
        if matched_qids is None:
            matched_qids = overlap_question_map.get(str(target_int), [])
        matched_qids = {str(q) for q in matched_qids}
        filtered = df_train[
            df_train["questions"].apply(lambda qs: any(str(q) in matched_qids for q in qs))
        ].copy()
    elif relaxed_filtering:
        target_concepts = question_concept_set(target_int, context, concept_route_fn)
        filtered = df_train[
            df_train["questions"].apply(
                lambda qs: any(
                    has_min_concept_overlap(
                        q,
                        target_int,
                        target_concepts,
                        context,
                        concept_route_fn,
                        min_concept_overlap,
                    )
                    for q in qs
                )
            )
        ].copy()
    else:
        filtered = df_train[df_train["questions"].apply(lambda x: target_int in [_qid(q) for q in x])].copy()
    
    if len(filtered) == 0:
        if relaxed_filtering:
            return "No peers who solved concept-matched questions for the TARGET QUESTION.", pd.DataFrame()
        return "No peers who solved the TARGET QUESTION.", pd.DataFrame()

    filtered = search_jaccard_refined(
        filtered,
        target_id,
        window,
        context,
        difficulty_dict,
        concept_route_fn,
        relaxed_filtering,
        min_concept_overlap,
        overlap_question_map,
    )
    qids, peers, selected = RAG_jaccard_peers(
        target_seq,
        filtered,
        threshold=None,
        top_per_label=top_per_label,
    )
    return render_peer_evidence(
        peers,
        context,
        difficulty_dict,
        representation=representation,
        relaxed_filtering=relaxed_filtering,
    ), qids


def make_numeric_state_seq(
    questions: Sequence[Any],
    responses: Sequence[Any],
    difficulty_dict: Mapping[Any, Mapping[str, float]],
) -> list[dict[str, Any]]:
    seq = []
    for q, r in zip(questions, responses):
        q = _qid(q)
        r = int(r)
        peer_rate = _rate(q, difficulty_dict)
        residual = None if peer_rate is None else r - peer_rate
        seq.append({"q": q, "peer_rate": peer_rate, "resp": r, "residual": residual})
    return seq


def numerical_position_sim(
    x: Mapping[str, Any],
    y: Mapping[str, Any],
    resp_weight: float = 0.30,
    diff_weight: float = 0.30,
    residual_weight: float = 0.40,
) -> tuple[float | None, float]:
    resp_sim = 1.0 if x["resp"] == y["resp"] else 0.0
    diff_sim = None
    if x["peer_rate"] is not None and y["peer_rate"] is not None:
        diff_sim = max(0.0, 1.0 - abs(x["peer_rate"] - y["peer_rate"]))

    residual_sim = None
    if x["residual"] is not None and y["residual"] is not None:
        residual_sim = max(0.0, 1.0 - abs(x["residual"] - y["residual"]) / 2.0)

    return weighted_available_average(
        [(resp_sim, resp_weight), (diff_sim, diff_weight), (residual_sim, residual_weight)],
        expected_total_weight=resp_weight + diff_weight + residual_weight,
    )


def numerical_state_seq_sim(
    current_seq: Sequence[Mapping[str, Any]],
    previous_seq: Sequence[Mapping[str, Any]],
    coverage_penalty: float = 0.25,
) -> tuple[float, float, float]:
    assert len(current_seq) == len(previous_seq)
    weights = get_position_weights(len(current_seq))
    score_sum = 0.0
    coverage_sum = 0.0
    total_weight = 0.0

    for weight, x, y in zip(weights, current_seq, previous_seq):
        pos_score, pos_coverage = numerical_position_sim(x, y)
        if pos_score is None:
            continue
        score_sum += weight * pos_score
        coverage_sum += weight * pos_coverage
        total_weight += weight

    if total_weight <= 0:
        return 0.0, 0.0, 0.0
    raw_score = score_sum / total_weight
    coverage = coverage_sum / total_weight
    adjusted_score = raw_score * ((1.0 - coverage_penalty) + coverage_penalty * coverage)
    return adjusted_score, raw_score, coverage


def numerical_question_difficulty_sim(q1: Any, q2: Any, difficulty_dict: Mapping[Any, Mapping[str, float]]) -> float | None:
    r1 = _rate(q1, difficulty_dict)
    r2 = _rate(q2, difficulty_dict)
    if r1 is None or r2 is None:
        return None
    return max(0.0, 1.0 - abs(r1 - r2))


def find_timestamp(
    tar: Any,
    questions: Sequence[Any],
    responses: Sequence[Any],
    qlist: Sequence[Any],
    rlist: Sequence[Any],
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    window: int = 5,
    topk: int = 3,
    state_weight: float = 0.65,
    target_diff_weight: float = 0.30,
    recency_weight: float = 0.05,
    coverage_penalty: float = 0.25,
    max_t: int | None = None,
) -> list[dict[str, Any]]:
    questions = list(questions)
    responses = list(responses)
    qlist = list(qlist)
    rlist = list(rlist)

    if max_t is None:
        max_t = len(questions)
    max_t = min(max_t, len(questions), len(responses))
    cur_len = min(window, len(qlist), len(rlist))
    if cur_len == 0:
        return []

    current_q = qlist[-cur_len:]
    current_r = rlist[-cur_len:]
    current_seq = make_numeric_state_seq(current_q, current_r, difficulty_dict)
    candidates = []
    expected_final_weight = state_weight + target_diff_weight + recency_weight

    for t in range(cur_len, max_t):
        pre_q = questions[t - cur_len:t]
        pre_r = responses[t - cur_len:t]
        if len(pre_q) != cur_len or len(pre_r) != cur_len:
            continue

        previous_seq = make_numeric_state_seq(pre_q, pre_r, difficulty_dict)
        state_sim, raw_state_sim, state_coverage = numerical_state_seq_sim(current_seq, previous_seq, coverage_penalty)
        target_diff_sim = numerical_question_difficulty_sim(questions[t], tar, difficulty_dict)
        denom = max(1, max_t - cur_len - 1)
        recency = (t - cur_len) / denom
        raw_score, final_coverage = weighted_available_average(
            [(state_sim, state_weight), (target_diff_sim, target_diff_weight), (recency, recency_weight)],
            expected_total_weight=expected_final_weight,
        )
        if raw_score is None:
            continue
        score = raw_score * ((1.0 - coverage_penalty) + coverage_penalty * final_coverage)
        candidates.append({
            "t": t,
            "qlist": pre_q,
            "rlist": pre_r,
            "tq": questions[t],
            "tr": responses[t],
            "score": score,
            "state_sim": state_sim,
            "raw_state_sim": raw_state_sim,
            "state_coverage": state_coverage,
            "target_diff_sim": target_diff_sim,
            "recency": recency,
            "final_coverage": final_coverage,
            "pseudo_target_peer_rate": _rate(questions[t], difficulty_dict),
            "current_residuals": [state.get("residual") for state in current_seq],
            "pre_residuals": [state.get("residual") for state in previous_seq],
            "current_responses": current_r,
            "pre_responses": pre_r,
        })
    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:topk]


def build_similarity_breakdown(case: Mapping[str, Any]) -> str:
    """Build the self-evidence similarity explanation used in prompts."""
    
    state_sim = case.get("state_sim")
    diff_sim = case.get("target_diff_sim")
    cur_res = case.get("current_residuals", [])
    pre_res = case.get("pre_residuals", [])
    
    valid_pairs = []
    for x, y in zip(cur_res, pre_res):
      if x is None or y is None:
          continue
      valid_pairs.append(1.0 - abs(x - y) / 2.0)
    
    residual_sim = sum(valid_pairs) / len(valid_pairs) if valid_pairs else None
    
    lines = ["Similarity breakdown:"]
    if state_sim is not None:
      lines.append(f"- State similarity: {state_sim:.3f}")
    if residual_sim is not None:
      lines.append(f"- Residual similarity: {residual_sim:.3f}")
    if diff_sim is not None:
      lines.append(
          f"- Difficulty similarity: {diff_sim:.3f}"
      )
    
    return "\n".join(lines)


def render_self_evidence(
    selected: Sequence[Mapping[str, Any]],
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    representation: str = "auto",
) -> str:
    if not selected:
        return "No previous self-history cases are available."

    text = ""
    for idx, case in enumerate(selected):
        text += f"**Prior similar state {idx + 1}**\n"
        text += build_similarity_breakdown(case) + "\n"
        text += history_to_string_no_concept(case["qlist"], case["rlist"], context, representation=representation)
        rate = _rate_to_str(case.get("pseudo_target_peer_rate"))
        diff = _diff_label(case["tq"], difficulty_dict)
        concepts = extract_concepts(case["tq"], context)
        outcome = "Correct" if int(case["tr"]) == 1 else "Incorrect"
        text += "\n\n>>>> Next question after this pattern:\n"
        text += f"- Concept(s): {concepts}\n"
        text += f"- Peer Answer Rate: {rate} ({diff})\n"
        text += f"- Outcome: {outcome}\n\n"
    return text


def find_self(
    questions: Sequence[Any],
    responses: Sequence[Any],
    qlist: Sequence[Any],
    rlist: Sequence[Any],
    tq: Any,
    window: int,
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    representation: str = "auto",
    topk: int = 3,
    add_counter: bool = True,
    max_next_idx: int | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    if max_next_idx is None:
        max_next_idx = len(questions)
    max_next_idx = min(max_next_idx, len(questions), len(responses))

    candidate_limit = topk * 10

    candidates = find_timestamp(
        tar=tq,
        questions=questions,
        responses=responses,
        qlist=qlist,
        rlist=rlist,
        difficulty_dict=difficulty_dict,
        window=window,
        topk=candidate_limit,
        max_t=max_next_idx,
    )
    if not candidates:
        return "No previous self-history cases are available.", []

    selected = candidates[:topk]
    remaining = candidates[topk:]

    if add_counter and selected:
        outcomes = [int(case["tr"]) for case in selected]
        if len(set(outcomes)) == 1:
            opposite_label = 1 - outcomes[0]
            for case in remaining:
                if int(case["tr"]) == opposite_label:
                    selected.append(case)
                    break

    return render_self_evidence(selected, context, difficulty_dict, representation=representation), selected


def make_peer_seq_factory(
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    concept_route_fn: Callable[[Any], Sequence[Any]] | None = None,
) -> Callable[[Sequence[Any], Sequence[Any]], set[Any]]:
    def _make_peer_seq(questions: Sequence[Any], responses: Sequence[Any]) -> set[Any]:
        return make_peer_seq(questions, responses, context, difficulty_dict, concept_route_fn)
    return _make_peer_seq


def make_find_peers(
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    representation: str = "auto",
    concept_route_fn: Callable[[Any], Sequence[Any]] | None = None,
    relaxed_filtering = False,
    min_concept_overlap: float = 0.5,
    overlap_question_map: Mapping[Any, Sequence[Any]] | None = None,
    top_per_label: int = 2,
) -> Callable[[set[Any], pd.DataFrame, Any, float, int], tuple[str, pd.DataFrame]]:
    def _find_peers(target_seq: set[Any], df_train: pd.DataFrame, target_id: Any, alpha: float, window: int):
        return find_peers(
            target_seq,
            df_train,
            target_id,
            alpha,
            window,
            context,
            difficulty_dict,
            representation,
            concept_route_fn,
            relaxed_filtering,
            min_concept_overlap,
            overlap_question_map,
            top_per_label,
        )
    _find_peers.relaxed_filtering = relaxed_filtering
    _find_peers.top_per_label = top_per_label
    return _find_peers


def make_find_self(
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    representation: str = "auto",
    topk: int = 3,
    add_counter = True,
) -> Callable[..., tuple[str, list[dict[str, Any]]]]:
    def _find_self(
        questions: Sequence[Any],
        responses: Sequence[Any],
        qlist: Sequence[Any],
        rlist: Sequence[Any],
        tq: Any,
        window: int,
    ):
        return find_self(
            questions,
            responses,
            qlist,
            rlist,
            tq,
            window,
            context,
            difficulty_dict,
            representation,
            topk,
            add_counter,
            None,
        )
    return _find_self

