# STEER-KT

This directory is a portable, standard-setting implementation of STEER-KT.
It contains the DBE, NIPS, and XES data used by the method.

## Dataset Availability

This repository does not include the preprocessed dataset files because they exceed GitHub's maximum file size limit. All datasets used in this work are publicly available. We will release the preprocessed datasets through a Google Drive link after publication. 


## Repository layout

```text
STEER_KT/
├── dataset/                 # DKT splits, LLM splits, and metadata
├── src/                     # retrieval, prompt, and prediction code
│   ├── dataset.py           # portable dataset/context loader
│   ├── find_evidence.py     # peer and self evidence retrieval
│   ├── prediction.py        # prompt construction and LLM calls
│   ├── utils.py             # metadata/context utilities
│   ├── run_steer_kt.py      # end-to-end runner
│   ├── run_steer_kt.sh      # shell entry point
│   └── requirements.txt
└── outputs/                 # generated automatically; not included
```

## Setup

```bash
cd STEER_KT
python3 -m pip install -r src/requirements.txt
export OPENAI_API_KEY="your-key"
```

## Run

Run retrieval and one prediction pass on all datasets:

```bash
./src/run_steer_kt.sh
```

Run selected datasets or repeated prediction passes:

```bash
DATASETS=NIPS,XES,DBE RUNS=1,2,3 ./src/run_steer_kt.sh
```

Build evidence without making any LLM API requests:

```bash
RETRIEVE_ONLY=1 ./src/run_steer_kt.sh
```

Useful environment variables are `MODEL`, `SERVICE_TIER`,
`EVIDENCE_WORKERS`, `MAX_WORKERS`, `SAVE_EVERY`, `API_KEY_ENV`, and
`API_BASE_URL`. Evidence retrieval and prediction are checkpointed, so the
same command resumes completed work.

## Main setting

- Retrieval window: the five interactions immediately preceding the target.
- Peer evidence: the two highest-scoring correct and two highest-scoring
  incorrect peer episodes.
- Self evidence: the three highest-scoring prior self episodes, plus an
  opposite-outcome counterexample when available.
- Representation: concept for DBE and question content for NIPS/XES.
- Prediction mode: peer and self evidence together (`all`).

Generated evidence is stored under `outputs/evidence/`; model predictions are
stored under `outputs/results/`.
