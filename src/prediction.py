"""Zero-shot prediction runner for NIPS, XES, and DBE adaptive-KT datasets.

This file is intended to be used from a notebook after you have already loaded:
- LLM_test
- difficulty_dict
- context from build_content_context/build_nips_context/build_dbe_context
- an OpenAI client

The main loop intentionally starts with:
for row in tqdm.tqdm(LLM_test[:].itertuples(index=False), total = len(LLM_test)):
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping

import json
import time

import tqdm

from utils import *
from find_evidence import *

ZERO_SHOT_SYSTEM_PROMPT = """You are an expert student-performance prediction assistant.
Predict whether the TARGET STUDENT will answer the TARGET QUESTION correctly.
Return valid JSON only."""


ZERO_SHOT_OUTPUT_SCHEMA = """Return exactly this JSON object:
{
  "prediction": "Correct" or "Incorrect",
  "analysis": "Your evidence-grounded reasoning in 3-6 sentences."
}"""


CUSTOM_SYSTEM_PROMPT = """You are a professional Knowledge Tracing Assistant.
Predict whether the TARGET STUDENT will answer the TARGET QUESTION correctly using the provided evidence. Base your prediction on the observed evidence rather than assumptions about underlying mastery. Return valid JSON only."""
# You are a professional Knowledge Tracing Assistant that predicts whether a student will answer the target question correctly.
# Analyze the student's mastery level and behavioral patterns using all provided evidence, and provide a highly accurate prediction.

NORMAL_DESCRIPTION = """[Evidence Sources]

[TARGET STUDENT Recent History]
- The following is the recent history of the TARGET STUDENT.
** NOTE: The history may contain only correct attempts; consider both performance consistency and the types of problems attempted.

[TARGET STUDENT's Past Similar-State Cases]:
- These cases are past similar-state cases from the same TARGET STUDENT.
- Each case reflects a previously observed behavioral state with a similar recent performance pattern relative to question difficulty.
- Similarity is based on recent correctness patterns, question difficulty, and residual behavior (performance relative to expected difficulty).
- Use these cases as evidence of the student's state-dependent behavior rather than direct concept overlap.
- Do not simply majority-vote the outcomes; weigh each case by its behavioral similarity and next-question difficulty similarity.

For each case:
- Prior similar state: a past attempt pattern behaviorally similar to the student's current recent state.
- State similarity: overall similarity between the prior state and the TARGET STUDENT's current recent state.
- Difficulty similarity (next question): similarity between the difficulty of the prior case's next question and the TARGET QUESTION.
- Residual similarity: similarity in performance relative to question difficulty.

- Next question: the question attempted after the prior-state pattern.
- Outcome: whether the student answered that next question correctly or incorrectly.

[Similar Students' Recent Histories]:
- These cases from other students who attempted the TARGET QUESTION and share similar learning trajectories.
- Similarity is based on recent correctness patterns, question difficulty trajectories, and concept/context similarity.
- When referencing similar students, do not rely only on overlap in correct attempts.
- When deciding which peer cases are most relevant, compare the provided students relative to one another and explain why certain students are more behaviorally similar to the TARGET STUDENT than others.
- IMPORTANT: Do not use the number of positive vs negative cases as a signal. This is a retrieval formatting constraint, not statistical information. Only use similarity-based reasoning.

----------------------------

[Task]

The TARGET STUDENT will attempt the TARGET QUESTION.
Predict whether the student will answer the TARGET QUESTION correctly.

"""

CUMTOM_DESCRIPTION = """[Evidence Sources]

[TARGET STUDENT Recent History]
- The following is the recent history of the TARGET STUDENT.
** NOTE: The history may contain only correct attempts; consider both performance consistency and the types of problems attempted.

[TARGET STUDENT's Past Similar-State Cases]:
- These cases are past similar-state cases from the same TARGET STUDENT.
- Each case reflects a previously observed behavioral state with a similar recent performance pattern relative to question difficulty.
- Similarity is based on recent correctness patterns, question difficulty, and residual behavior (performance relative to expected difficulty).
- Use these cases as evidence of the student's state-dependent behavior rather than direct concept overlap.
- Do not simply majority-vote the outcomes; weigh each case by its behavioral similarity and next-question difficulty similarity.

For each case:
- Prior similar state: a past attempt pattern behaviorally similar to the student's current recent state.
- State similarity: overall similarity between the prior state and the TARGET STUDENT's current recent state.
- Difficulty similarity (next question): similarity between the difficulty of the prior case's next question and the TARGET QUESTION.
- Residual similarity: similarity in performance relative to question difficulty.

- Next question: the question attempted after the prior-state pattern.
- Outcome: whether the student answered that next question correctly or incorrectly.

[Similar Students' Recent Histories]:
- These cases from other students who attempted the TARGET QUESTION and share similar learning trajectories.
- Similarity is based on recent correctness patterns, question difficulty trajectories, and concept/context similarity.
- When referencing similar students, do not rely only on overlap in correct attempts.
- When deciding which peer cases are most relevant, compare the provided students relative to one another and explain why certain students are more behaviorally similar to the TARGET STUDENT than others.
- IMPORTANT: Do not use the number of positive vs negative cases as a signal. This is a retrieval formatting constraint, not statistical information. Only use similarity-based reasoning.

----------------------------

[Task]

The TARGET STUDENT will attempt the TARGET QUESTION.
Predict whether the student will answer the TARGET QUESTION correctly.
** NOTE: You are an assistant who evaluates all provided evidence. You must base your prediction only on evidence that is relevant and informative for the TARGET QUESTION. It is NOT required to use all evidence sources. You may ignore any evidence that does not meaningfully improve prediction quality. Do not force integration of weak or irrelevant evidence.
"""

CUMTOM_NONE = """[Evidence Sources]

[TARGET STUDENT Recent History]
- The following is the recent history of the TARGET STUDENT.
** NOTE: The history may contain only correct attempts; consider both performance consistency and the types of problems attempted.

----------------------------

[Task]

The TARGET STUDENT will attempt the TARGET QUESTION.
Predict whether the student will answer the TARGET QUESTION correctly.
"""
# ** NOTE: You are an assistant who evaluates all provided evidence. You must base your prediction only on evidence that is relevant and informative for the TARGET QUESTION. It is NOT required to use all evidence sources. You may ignore any evidence that does not meaningfully improve prediction quality. Do not force integration of weak or irrelevant evidence.
# """

CUMTOM_PEER = """[Evidence Sources]

[TARGET STUDENT Recent History]
- The following is the recent history of the TARGET STUDENT.
** NOTE: The history may contain only correct attempts; consider both performance consistency and the types of problems attempted.

[Similar Students' Recent Histories]:
- These cases from other students who attempted the TARGET QUESTION and share similar learning trajectories.
- Similarity is based on recent correctness patterns, question difficulty trajectories, and concept/context similarity.
- When referencing similar students, do not rely only on overlap in correct attempts.
- When deciding which peer cases are most relevant, compare the provided students relative to one another and explain why certain students are more behaviorally similar to the TARGET STUDENT than others.
- IMPORTANT: Do not use the number of positive vs negative cases as a signal. This is a retrieval formatting constraint, not statistical information. Only use similarity-based reasoning.

----------------------------

[Task]

The TARGET STUDENT will attempt the TARGET QUESTION.
Predict whether the student will answer the TARGET QUESTION correctly.
"""
# ** NOTE: You are an assistant who evaluates all provided evidence. You must base your prediction only on evidence that is relevant and informative for the TARGET QUESTION. It is NOT required to use all evidence sources. You may ignore any evidence that does not meaningfully improve prediction quality. Do not force integration of weak or irrelevant evidence.
# """

CUMTOM_SELF = """[Evidence Sources]

[TARGET STUDENT Recent History]
- The following is the recent history of the TARGET STUDENT.
** NOTE: The history may contain only correct attempts; consider both performance consistency and the types of problems attempted.

[TARGET STUDENT's Past Similar-State Cases]:
- These cases are past similar-state cases from the same TARGET STUDENT.
- Each case reflects a previously observed behavioral state with a similar recent performance pattern relative to question difficulty.
- Similarity is based on recent correctness patterns, question difficulty, and residual behavior (performance relative to expected difficulty).
- Use these cases as evidence of the student's state-dependent behavior rather than direct concept overlap.
- Do not simply majority-vote the outcomes; weigh each case by its behavioral similarity and next-question difficulty similarity.

For each case:
- Prior similar state: a past attempt pattern behaviorally similar to the student's current recent state.
- State similarity: overall similarity between the prior state and the TARGET STUDENT's current recent state.
- Difficulty similarity (next question): similarity between the difficulty of the prior case's next question and the TARGET QUESTION.
- Residual similarity: similarity in performance relative to question difficulty.

- Next question: the question attempted after the prior-state pattern.
- Outcome: whether the student answered that next question correctly or incorrectly.

----------------------------

[Task]

The TARGET STUDENT will attempt the TARGET QUESTION.
Predict whether the student will answer the TARGET QUESTION correctly.
"""
# ** NOTE: You are an assistant who evaluates all provided evidence. You must base your prediction only on evidence that is relevant and informative for the TARGET QUESTION. It is NOT required to use all evidence sources. You may ignore any evidence that does not meaningfully improve prediction quality. Do not force integration of weak or irrelevant evidence.
# """

STRICT_PEER_DESCRIPTION_LINE = (
    "- These cases from other students who attempted the TARGET QUESTION and share similar learning trajectories."
)
RELAXED_PEER_DESCRIPTION_LINE = (
    "- These cases are from other students who attempted questions whose concepts match or sufficiently overlap "
    "with the TARGET QUESTION, and who share similar learning trajectories."
)


def _peer_description(description: str, relaxed_peer_evidence: bool = False) -> str:
    if not relaxed_peer_evidence:
        return description
    return description.replace(STRICT_PEER_DESCRIPTION_LINE, RELAXED_PEER_DESCRIPTION_LINE)

CUSTOM_OUTPUT_SCHEMA_PARTIAL = """>> [Output Format]
Return valid JSON only. Do not include markdown or extra text.
** IMPORTANT: Only include "evidence_tension" if there is meaningful disagreement between evidence sources. If evidence is consistent, omit this field entirely.

{
  "prediction": "Correct" or "Incorrect",

  "used_evidence": [...],

  "ignored_evidence": [...],
  
  "evidence_tension": {
  "status": "none / mild / strong",

  "between": [...],

  "reason": "brief explanation of why signals agree or disagree"},

  "analysis": "3~6 sentences grounded reasoning"
}
"""

CUSTOM_OUTPUT_SCHEMA = """>> [Output Format]
Return valid JSON only. Do not include markdown or extra text.

{
  "prediction": "Correct" or "Incorrect",

  "used_evidence": [
    "Recent History",
    "Self Evidence (Case #n)",
    "Peer Evidence (Student #n)"
  ],

  "ignored_evidence": [
    "Self Evidence (Case #n - ex. irrelevant pattern)",
    "Peer Evidence (Student #n - ex.weak alignment)"
  ],
  
  "evidence_tension": {
  "status": "none / mild / strong",

  "between": [
    ex. "Recent History vs Self Evidence",
    ex. "Self Evidence vs Peer Evidence"
  ],

  "reason": "brief explanation of why signals agree or disagree"},

  "analysis": "3~6 sentences grounded reasoning"
}
"""

def _row_value(row: Any, name: str, default: Any = None) -> Any:
    return getattr(row, name, default)


def _safe_lookup(mapping: Mapping[Any, Any], key: Any, default: Any = None) -> Any:
    for candidate in (key, str(key)):
        if candidate in mapping:
            return mapping[candidate]
    try:
        int_key = int(key)
    except (TypeError, ValueError):
        int_key = None
    if int_key is not None and int_key in mapping:
        return mapping[int_key]
    return default


def part_evidence_prompt(
    row: Any,
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    window: int = 5,
    representation: Representation = "auto",
    peer_evidence: Any = None,
    self_evidence: Any = None,
    evidence_conflict: Any = None,
    additional_context: str = "",
    relaxed_peer_evidence: bool = False,
    **_: Any,
    ) -> tuple[str, str]:
    target_id = _row_value(row, "target_q")
    questions = list(_row_value(row, "questions", []))
    responses = list(_row_value(row, "responses", []))
    
    target_content = read_prob(target_id, context) if context.question_text is not None else ""
    target_concepts = extract_concepts(target_id, context)
    target_rate = get_peer_rate(target_id, difficulty_dict)
    target_ar = "unknown" if target_rate is None else f"{target_rate:.4f}"
    target_diff = decide_label(target_id, difficulty_dict)
    
    user_prompt = """You are a knowledge tracing assistant.
Your goal is to predict whether the TARGET STUDENT will answer the TARGET QUESTION correctly.

----------------------------
"""
    
    if representation == "auto":
      representation = getattr(context, "default_representation", "content")
    elif representation == "concept" and additional_context != "":
      user_prompt += f"""{additional_context}
----------------------------
"""
    
    target_history = history_to_string(
      questions[-window:],
      responses[-window:],
      context,
      representation=representation,
    )
    
    target_pattern = history_to_string_no_concept(
      questions[-window:],
      responses[-window:],
      context,
    )
    
    has_self = self_evidence is not None
    has_peer = peer_evidence is not None
    
    if has_self and has_peer:
      evidence_description = NORMAL_DESCRIPTION # CUMTOM_DESCRIPTION
    elif has_self:
      evidence_description = CUMTOM_SELF
    elif has_peer:
      evidence_description = CUMTOM_PEER
    else:
      evidence_description = CUMTOM_NONE
 
    target_q_info = f""">> [TARGET QUESTION Info]
- Content: {target_content}
- Concept: {target_concepts}
- Peer Answer Rate: {target_ar} ({target_diff})

----------------------------
"""

    if evidence_conflict:
        evidence_description = NORMAL_DESCRIPTION
        target_q_info = f""">> [TARGET QUESTION Info]
- Content: {target_content}
- Concept: {target_concepts}
- Peer Answer Rate: {target_ar} ({target_diff})

----------------------------
"""

    evidence_description = _peer_description(evidence_description, relaxed_peer_evidence and has_peer)
    
    evidence_lines = []
    
    if has_self:
      evidence_lines.append(
          ">> [Target Student's chronological recent pattern summary]\n"
          + str(target_pattern)
      )
      evidence_lines.append(
          ">> [TARGET STUDENT's Past Similar-State Cases]\n"
          + str(self_evidence)
      )
    
    if has_peer:
      evidence_lines.append(
          ">> [Similar Students' Recent Histories]\n"
          + str(peer_evidence)
      )
    
    evidence_block = "\n\n".join(evidence_lines)
    
    user_prompt += f"""
{evidence_description}
{target_q_info}
[Evidences]
>> [Definitions & Context]
- Student History: A chronological sequence of attempts, indexed from #1 (earliest) to #N (most recent).
- Peer Answer Rate is the empirical correctness rate among other students. Lower values usually indicate harder questions. The difficulty label is precomputed from the dataset distribution; use both the numeric rate and the
label.
- Difficulty Scale: The 'Peer Answer Rate' reflects the relative difficulty compared to all questions; Very Easy: Top 20% (highest correct rates), Easy: Top 20% - 40%, Mid: Top 40% - 60%, Hard: Top 60% - 80%, Very Hard: Top
80% - 100% (lowest correct rates)
- All attempts are indexed chronologically, from #1 (earliest) to #N (most recent).
{"* (Please refer to the [Concept Descriptions] above for the specific descriptions of each concept.)" if representation == "concept" else ""}

>> [TARGET STUDENT Recent History]
{target_history}

{evidence_block}

----------------------------

""" # {"* (Please refer to the [Concept Descriptions] above for the specific descriptions of each concept.)" if representation == "concept" else ""}
    if evidence_conflict: user_prompt += f"{ZERO_SHOT_OUTPUT_SCHEMA}"
    else: user_prompt += f"{ZERO_SHOT_OUTPUT_SCHEMA}"

    return ZERO_SHOT_SYSTEM_PROMPT, user_prompt


def part_evidence_prompt_3(**kwargs: Any) -> tuple[str, str, str]:
  _, user_prompt = part_evidence_prompt(
      **kwargs,
      evidence_conflict=True,
  )
  return ZERO_SHOT_SYSTEM_PROMPT, user_prompt, ""

def all_evidence_prompt(
    row: Any,
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    window: int = 5,
    representation: Representation = "auto",
    peer_evidence: Any = None,
    self_evidence: Any = None,
    additional_context: str = "",
    relaxed_peer_evidence: bool = False,
    **_: Any,
) -> tuple[str, str]:
    seq_id = _row_value(row, "seq_id")
    target_id = _row_value(row, "target_q")
    questions = list(_row_value(row, "questions", []))
    responses = list(_row_value(row, "responses", []))

    target_content = read_prob(target_id, context) if context.question_text is not None else ""
    target_concepts = extract_concepts(target_id, context)
    target_rate = get_peer_rate(target_id, difficulty_dict)
    target_ar = "unknown" if target_rate is None else f"{target_rate:.4f}"
    target_diff = decide_label(target_id, difficulty_dict)
    
    user_prompt = """You are a knowledge tracing assistant.
Your goal is to predict whether the TARGET STUDENT will answer the TARGET QUESTION correctly.

----------------------------
"""
    if representation == "auto":
        representation = getattr(context, "default_representation", "content")
    elif representation == "concept" and additional_context != "":
        user_prompt += f"""{additional_context}\n----------------------------"""

    target_history = history_to_string(
        questions[-window:],
        responses[-window:],
        context,
        representation=representation,
    )

    target_pattern = history_to_string_no_concept(
      questions[-window:],
      responses[-window:],
      context,
    )

    TARGET_Q_INFO = f""">> [TARGET QUESTION Info]
- Content: {target_content}
- Concept: {target_concepts}
- Peer Answer Rate: {target_ar} ({target_diff})

----------------------------

"""

    question_lines = []
    if target_concepts:
        question_lines.append(f"- Concept(s): {target_concepts}")
    question_lines.append(f"- Peer Answer Rate: {target_ar} ({target_diff})")

    evidence_lines = []
    if self_evidence is not None:
        evidence_lines.append(">> [TARGET STUDENT's Past Similar-State Cases]\n" + str(self_evidence))
    if peer_evidence is not None:
        evidence_lines.append(">> [Similar Students' Recent Histories]\n" + str(peer_evidence))
    evidence_block = "\n\n".join(evidence_lines)
    evidence_description = _peer_description(NORMAL_DESCRIPTION, relaxed_peer_evidence and peer_evidence is not None) # CUMTOM_DESCRIPTION

    user_prompt += f"""
{evidence_description}
{TARGET_Q_INFO}
[Evidences]
>> [Definitions & Context]
- Student History: A chronological sequence of attempts, indexed from #1 (earliest) to #N (most recent).
- Peer Answer Rate is the empirical correctness rate among other students. Lower values usually indicate harder questions. The difficulty label is precomputed from the dataset distribution; use both the numeric rate and the label.
- Difficulty Scale: The 'Peer Answer Rate' reflects the relative difficulty compared to all questions; Very Easy: Top 20% (highest correct rates), Easy: Top 20% - 40%, Mid: Top 40% - 60%, Hard: Top 60% - 80%, Very Hard: Top 80% - 100% (lowest correct rates)
- All attempts are indexed chronologically, from #1 (earliest) to #N (most recent).
{"* (Please refer to the [Concept Descriptions] above for the specific descriptions of each concept.)" if representation == "concept" else ""}

>> [TARGET STUDENT Recent History]
{target_history}

>> [Target Student's chronological recent pattern summary]
{target_pattern}

{evidence_block}

----------------------------

{ZERO_SHOT_OUTPUT_SCHEMA}
""" # 
    evidence_prompt = f"""{TARGET_Q_INFO}
[Evidences]
>> [Definitions & Context]
- Student History: A chronological sequence of attempts, indexed from #1 (earliest) to #N (most recent).
- Peer Answer Rate is the empirical correctness rate among other students. Lower values usually indicate harder questions. The difficulty label is precomputed from the dataset distribution; use both the numeric rate and the
label.
- Difficulty Scale: The 'Peer Answer Rate' reflects the relative difficulty compared to all questions; Very Easy: Top 20% (highest correct rates), Easy: Top 20% - 40%, Mid: Top 40% - 60%, Hard: Top 60% - 80%, Very Hard: Top
80% - 100% (lowest correct rates)
- All attempts are indexed chronologically, from #1 (earliest) to #N (most recent).

>> [TARGET STUDENT Recent History]
{target_history}

>> [Target Student's chronological recent pattern summary]
{target_pattern}

{evidence_block}""" # {"* (Please refer to the [Concept Descriptions] above for the specific descriptions of each concept.)" if representation == "concept" else ""}

    return ZERO_SHOT_SYSTEM_PROMPT, user_prompt, evidence_prompt


def save_outputs(
    zero_shot_prediction: Mapping[Any, Any],
    answer_list: Mapping[Any, Any],
    message_task: Mapping[Any, Any],
    output_prefix: str | Path,
) -> None:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    file_save(
        zero_shot_prediction,
        answer_list,
        message_task,
        output_prefix,
        output_prefix.with_name(f"ans_{output_prefix.name}"),
        output_prefix.with_name(f"ex_{output_prefix.name}"),
    )


def _prediction_messages(system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

def _unpack_prompt_output(prompt_output: tuple[Any, ...]) -> tuple[str, str, str]:
    if len(prompt_output) == 3:
        return prompt_output
    if len(prompt_output) == 2:
        system_prompt, user_prompt = prompt_output
        return system_prompt, user_prompt, user_prompt
    raise ValueError("Prompt builder must return (system_prompt, user_prompt) or (system_prompt, user_prompt, evidence).")

    
def save_evidence_outputs(
    evidence_records: Mapping[Any, Any],
    answer_list: Mapping[Any, Any],
    message_task: Mapping[Any, Any],
    output_prefix: str | Path,
) -> None:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_json(evidence_records, output_prefix)
    save_json(answer_list, output_prefix.with_name(f"ans_{output_prefix.name}"))
    save_json(message_task, output_prefix.with_name(f"ex_{output_prefix.name}"))


def build_zero_shot_evidence(
    LLM_test,
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    build_prompt_fn: Callable[..., tuple[str, str]] = all_evidence_prompt,
    find_peers_fn: Callable[..., Any] | None = None,
    find_self_fn: Callable[..., Any] | None = None,
    make_peer_seq_fn: Callable[..., Any] | None = None,
    LLM_train=None,
    df_train=None,
    alpha: float = 1.0,
    window: int = 5,
    representation: Representation = "auto",
    output_prefix: str | Path | None = None,
    save_every: int = 25,
    resume_evidences: Mapping[Any, Any] | None = None,
    skip_seq_ids: set[Any] | None = None,
    relaxed_peer_evidence: bool | None = None,
) -> tuple[dict[Any, dict[str, Any]], dict[Any, Any], dict[Any, str], dict[Any, str]]:
    """Build and optionally save evidence/prompt records without calling the LLM."""
    evidence_records: dict[Any, dict[str, Any]] = dict(resume_evidences or {})
    answer_list: dict[Any, Any] = {}
    message_task: dict[Any, str] = {}
    evidences: dict[Any, str] = {}
    skipped = {str(x) for x in (skip_seq_ids or set())}
    if relaxed_peer_evidence is None:
        relaxed_peer_evidence = bool(getattr(find_peers_fn, "relaxed_filtering", False))

    for row in tqdm.tqdm(LLM_test.itertuples(index=False), total=len(LLM_test)):
        seq_id = row.seq_id
        if seq_id in evidence_records or str(seq_id) in evidence_records or str(seq_id) in skipped:
            continue

        answer_list[seq_id] = row.target_r
        questions = list(_row_value(row, "questions", []))
        responses = list(_row_value(row, "responses", []))
        target_id = _row_value(row, "target_q")
        train_df = LLM_train if LLM_train is not None else df_train
        qids = {target_id}

        target_seq = None
        if make_peer_seq_fn is not None:
            if getattr(make_peer_seq_fn, "use_full_history", False):
                target_seq = make_peer_seq_fn(questions, responses)
            else:
                target_seq = make_peer_seq_fn(questions[-window:], responses[-window:])

        peer_evidence = None
        if find_peers_fn is not None:
            if train_df is None:
                raise ValueError("Pass LLM_train or df_train when find_peers_fn is provided.")
            if target_seq is None:
                raise ValueError("Pass make_peer_seq_fn when find_peers_fn requires target_seq.")
            peer_evidence, qids = find_peers_fn(target_seq, train_df, target_id, alpha, window)

        self_evidence = None
        if find_self_fn is not None:
            self_evidence, _ = find_self_fn(
                questions,
                responses,
                questions[-window:],
                responses[-window:],
                target_id,
                window,
            )

        if find_peers_fn is None:
            qids = {target_id}

        qid_context = all_qids_to_string(
            set(qids) | {target_id},
            questions[-window:],
            responses[-window:],
            context,
        )
        concept_description = qid_context[1] if isinstance(qid_context, tuple) else ""

        system_prompt, user_prompt, evidence = _unpack_prompt_output(build_prompt_fn(
            row=row,
            context=context,
            difficulty_dict=difficulty_dict,
            window=window,
            representation=representation,
            peer_evidence=peer_evidence,
            self_evidence=self_evidence,
            additional_context=concept_description,
            relaxed_peer_evidence=relaxed_peer_evidence,
        ))
        
        message_task[seq_id] = user_prompt
        evidences[seq_id] = evidence
        evidence_records[seq_id] = {
            "seq_id": seq_id,
            "target_q": target_id,
            "target_r": row.target_r,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "evidence": evidence,
            "peer_evidence": peer_evidence,
            "self_evidence": self_evidence,
            "relaxed_peer_evidence": relaxed_peer_evidence,
            "qids": list(qids),
            "concept_description": concept_description,
        }

        if output_prefix is not None and save_every > 0 and len(evidence_records) % save_every == 0:
            save_evidence_outputs(evidence_records, answer_list, message_task, output_prefix)

    if output_prefix is not None:
        save_evidence_outputs(evidence_records, answer_list, message_task, output_prefix)

    return evidence_records, answer_list, message_task, evidences


def run_prediction_from_evidence(
    evidence_records: Mapping[Any, Any],
    client,
    model: str = "gpt-5.2",
    service_tier: str | None = "flex",
    output_prefix: str | Path | None = None,
    save_every: int = 25,
    sleep_seconds: float = 0.0,
    resume_predictions: dict[Any, Any] | None = None,
    max_workers: int = 1,
) -> tuple[dict[Any, str], dict[Any, Any], dict[Any, str]]:
    """Run LLM prediction from previously built evidence/prompt records.

    API calls can run concurrently with ``max_workers > 1``. Only the main
    thread mutates and checkpoints the result dictionaries, so partial-result
    resume files remain valid even when responses finish out of order.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    zero_shot_prediction = dict(resume_predictions or {})
    answer_list: dict[Any, Any] = {}
    message_task: dict[Any, str] = {}

    pending = []
    for seq_id, record in evidence_records.items():
        # Keep answer/prompt sidecars complete even when predictions resume
        # from a partial file. Previously these dictionaries contained only
        # newly requested rows, which produced truncated ans_*.json files.
        answer_list[seq_id] = record.get("target_r")
        message_task[seq_id] = record["user_prompt"]
        if seq_id in zero_shot_prediction or str(seq_id) in zero_shot_prediction:
            continue
        pending.append((seq_id, record))

    def request_one(seq_id, record):
        system_prompt = record["system_prompt"]
        user_prompt = record["user_prompt"]
        kwargs = {
            "model": model,
            "messages": _prediction_messages(system_prompt, user_prompt),
        }
        # GPT-5 mini/nano accept only their default temperature value. Omitting
        # the parameter preserves that default while keeping deterministic-
        # style temperature=0 calls for models that support it.
        if not str(model).startswith(("gpt-5-mini", "gpt-5-nano")):
            kwargs["temperature"] = 0.0
        if service_tier is not None:
            kwargs["service_tier"] = service_tier

        resp = client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content.strip()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return seq_id, content

    def checkpoint() -> None:
        if output_prefix is not None:
            save_outputs(zero_shot_prediction, answer_list, message_task, output_prefix)

    completed_since_start = 0

    def accept_result(seq_id, content) -> None:
        nonlocal completed_since_start
        zero_shot_prediction[seq_id] = content
        completed_since_start += 1
        if output_prefix is not None and save_every > 0 and completed_since_start % save_every == 0:
            checkpoint()

    try:
        if max_workers == 1:
            for seq_id, record in tqdm.tqdm(pending, total=len(pending)):
                result_seq_id, content = request_one(seq_id, record)
                accept_result(result_seq_id, content)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(request_one, seq_id, record): seq_id
                    for seq_id, record in pending
                }
                errors = []
                for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                    try:
                        result_seq_id, content = future.result()
                    except Exception as exc:
                        errors.append((futures[future], exc))
                    else:
                        accept_result(result_seq_id, content)
                if errors:
                    failed_seq_id, first_error = errors[0]
                    raise RuntimeError(
                        f"{len(errors)} concurrent API request(s) failed; "
                        f"first failed seq_id={failed_seq_id!r}"
                    ) from first_error
    except Exception:
        # Preserve every response completed before a rate-limit/network/API
        # failure so rerunning can resume rather than repeat paid calls.
        checkpoint()
        raise

    checkpoint()

    return zero_shot_prediction, answer_list, message_task

def _normalize_evidence_mode(evidence_mode: str) -> str:
    mode = str(evidence_mode).strip().lower()
    aliases = {
        "none": "none",
        "no": "none",
        "self": "self",
        "peer": "peer",
        "all": "all",
        "all_evidence": "all",
        "self_peer": "all",
        "both": "all",
    }
    if mode not in aliases:
        raise ValueError("evidence_mode must be one of: none, self, peer, all")
    return aliases[mode]


def _lookup_record(records: Mapping[Any, Any], seq_id: Any) -> Any:
    for key in (seq_id, str(seq_id)):
        if key in records:
            return records[key]
    try:
        int_key = int(seq_id)
    except (TypeError, ValueError):
        int_key = None
    if int_key is not None and int_key in records:
        return records[int_key]
    return None


def _select_cached_evidence(record: Mapping[str, Any], evidence_mode: str) -> tuple[Any, Any]:
    mode = _normalize_evidence_mode(evidence_mode)
    peer_evidence = record.get("peer_evidence") if mode in {"peer", "all"} else None
    self_evidence = record.get("self_evidence") if mode in {"self", "all"} else None
    return peer_evidence, self_evidence


def rebuild_zero_shot_prompts_from_evidence(
    LLM_test,
    evidence_records: Mapping[Any, Any],
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    evidence_mode: str = "all",
    build_prompt_fn: Callable[..., tuple[str, str]] | None = None,
    window: int = 5,
    representation: Representation = "auto",
    relaxed_peer_evidence: bool | None = None,
) -> tuple[dict[Any, dict[str, Any]], dict[Any, Any], dict[Any, str], dict[Any, str]]:
    """Rebuild prediction prompts using cached peer/self evidence only.

    This does not call find_peers_fn or find_self_fn. Use it after one full
    build_zero_shot_evidence(...) run that stored peer_evidence and self_evidence.
    """
    mode = _normalize_evidence_mode(evidence_mode)
    if build_prompt_fn is None:
        build_prompt_fn = all_evidence_prompt if mode == "all" else part_evidence_prompt

    prompt_records: dict[Any, dict[str, Any]] = {}
    answer_list: dict[Any, Any] = {}
    message_task: dict[Any, str] = {}
    evidences: dict[Any, str] = {}

    for row in tqdm.tqdm(LLM_test.itertuples(index=False), total=len(LLM_test)):
        seq_id = row.seq_id
        record = _lookup_record(evidence_records, seq_id)
        if record is None:
            continue
        record_relaxed_peer_evidence = (
            record.get("relaxed_peer_evidence", False)
            if relaxed_peer_evidence is None
            else relaxed_peer_evidence
        )

        target_id = _row_value(row, "target_q")
        answer_list[seq_id] = _row_value(row, "target_r", record.get("target_r"))
        peer_evidence, self_evidence = _select_cached_evidence(record, mode)

        concept_description = record.get("concept_description", "")
        if not concept_description:
            questions = list(_row_value(row, "questions", []))
            responses = list(_row_value(row, "responses", []))
            cached_qids = set(record.get("qids", []) or [])
            qid_context = all_qids_to_string(
                cached_qids | {target_id},
                questions[-window:],
                responses[-window:],
                context,
            )
            concept_description = qid_context[1] if isinstance(qid_context, tuple) else ""

        system_prompt, user_prompt, evidence = _unpack_prompt_output(build_prompt_fn(
            row=row,
            context=context,
            difficulty_dict=difficulty_dict,
            window=window,
            representation=representation,
            peer_evidence=peer_evidence,
            self_evidence=self_evidence,
            additional_context=concept_description,
            relaxed_peer_evidence=record_relaxed_peer_evidence,
        ))

        message_task[seq_id] = user_prompt
        evidences[seq_id] = evidence
        prompt_records[seq_id] = {
            "seq_id": seq_id,
            "target_q": target_id,
            "target_r": answer_list[seq_id],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "evidence": evidence,
            "peer_evidence": peer_evidence,
            "self_evidence": self_evidence,
            "relaxed_peer_evidence": record_relaxed_peer_evidence,
            "evidence_mode": mode,
        }

    return prompt_records, answer_list, message_task, evidences


def run_prediction_from_cached_evidence(
    LLM_test,
    evidence_records: Mapping[Any, Any],
    client,
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    evidence_mode: str = "all",
    build_prompt_fn: Callable[..., tuple[str, str]] | None = None,
    model: str = "gpt-5.2",
    service_tier: str | None = "flex",
    window: int = 5,
    representation: Representation = "auto",
    relaxed_peer_evidence: bool | None = None,
    output_prefix: str | Path | None = None,
    save_every: int = 25,
    sleep_seconds: float = 0.0,
    resume_predictions: dict[Any, Any] | None = None,
    max_workers: int = 1,
) -> tuple[dict[Any, str], dict[Any, Any], dict[Any, str], dict[Any, str]]:
    """Run only prediction for none/self/peer/all using cached evidence."""
    prompt_records, answer_list, message_task, evidences = rebuild_zero_shot_prompts_from_evidence(
        LLM_test=LLM_test,
        evidence_records=evidence_records,
        context=context,
        difficulty_dict=difficulty_dict,
        evidence_mode=evidence_mode,
        build_prompt_fn=build_prompt_fn,
        window=window,
        representation=representation,
        relaxed_peer_evidence=relaxed_peer_evidence,
    )
    zero_shot_prediction, pred_answers, pred_messages = run_prediction_from_evidence(
        evidence_records=prompt_records,
        client=client,
        model=model,
        service_tier=service_tier,
        output_prefix=output_prefix,
        save_every=save_every,
        sleep_seconds=sleep_seconds,
        resume_predictions=resume_predictions,
        max_workers=max_workers,
    )
    answer_list.update(pred_answers)
    message_task.update(pred_messages)
    return zero_shot_prediction, answer_list, message_task, evidences
    
def run_zero_shot_prediction(
    LLM_test,
    client,
    context: KnowledgeContext,
    difficulty_dict: Mapping[Any, Mapping[str, float]],
    build_prompt_fn: Callable[..., tuple[str, str]] = all_evidence_prompt,
    find_peers_fn: Callable[..., Any] | None = None,
    find_self_fn: Callable[..., Any] | None = None,
    make_peer_seq_fn: Callable[..., Any] | None = None,
    LLM_train=None,
    df_train=None,
    alpha: float = 1.0,
    model: str = "gpt-5.2",
    service_tier: str | None = "flex",
    window: int = 5,
    representation: Representation = "auto",
    output_prefix: str | Path | None = None,
    save_every: int = 25,
    sleep_seconds: float = 0.0,
    resume_predictions: dict[Any, Any] | None = None,
) -> tuple[dict[Any, str], dict[Any, Any], dict[Any, str], dict[Any, str]]:
    evidence_records, answer_list, message_task, evidences = build_zero_shot_evidence(
        LLM_test=LLM_test,
        context=context,
        difficulty_dict=difficulty_dict,
        build_prompt_fn=build_prompt_fn,
        find_peers_fn=find_peers_fn,
        find_self_fn=find_self_fn,
        make_peer_seq_fn=make_peer_seq_fn,
        LLM_train=LLM_train,
        df_train=df_train,
        alpha=alpha,
        window=window,
        representation=representation,
        output_prefix=None,
        save_every=save_every,
        skip_seq_ids=set((resume_predictions or {}).keys()),
    )
    zero_shot_prediction, pred_answers, pred_messages = run_prediction_from_evidence(
        evidence_records=evidence_records,
        client=client,
        model=model,
        service_tier=service_tier,
        output_prefix=output_prefix,
        save_every=save_every,
        sleep_seconds=sleep_seconds,
        resume_predictions=resume_predictions,
    )
    answer_list.update(pred_answers)
    message_task.update(pred_messages)
    return zero_shot_prediction, answer_list, message_task, evidences


if __name__ == "__main__":
    raise SystemExit(
        "Import this file from a notebook and call run_zero_shot_prediction(...). "
        "It expects LLM_test, client, context, and difficulty_dict to be prepared first."
    )
