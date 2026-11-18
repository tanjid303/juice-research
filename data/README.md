# Data acquisition and JSONL format

This project uses **filtered JSONL** files under `data/`. Scripts do **not** auto-download; convert upstream sources locally once, then point `--data` at the file.

## ConflictQA (primary — mini-pilot & Fix 1)

| Field | Value |
|-------|--------|
| **Source** | [osunlp/ConflictQA](https://huggingface.co/datasets/osunlp/ConflictQA) |
| **GitHub** | [OSU-NLP-Group/LLM-Knowledge-Conflict](https://github.com/OSU-NLP-Group/LLM-Knowledge-Conflict) |
| **Paper** | Xie et al., ICLR 2024 — *Adaptive Chameleon or Stubborn Sloth* |
| **Recommended config** | `ConflictQA-popQA-chatgpt` (PopQA entity QA with ChatGPT-guided parametric memory) |
| **Split** | `test` |

### Download (manual)

```python
from datasets import load_dataset

ds = load_dataset("osunlp/ConflictQA", "ConflictQA-popQA-chatgpt", split="test")
```

Or use the helper in-repo (after `pip install -r requirements.txt`):

```powershell
python -c "from common.conflictqa import export_conflictqa_from_hf; export_conflictqa_from_hf('data/conflictqa_filtered.jsonl', max_items=500)"
```

Inspect raw keys first — configs differ slightly:

```python
print(ds[0].keys())
```

### Target file

Save as **`data/conflictqa_filtered.jsonl`** (one JSON object per line):

```json
{
  "id": "popqa_42",
  "question": "Who wrote Pride and Prejudice?",
  "parametric_answer": "Jane Austen",
  "context_answer": "Charlotte Bronte",
  "passage_agrees": "Jane Austen published Pride and Prejudice in 1813 ...",
  "passage_conflicts": "Charlotte Bronte wrote Pride and Prejudice ...",
  "source": "ConflictQA-popQA-chatgpt"
}
```

| Field | Meaning |
|-------|---------|
| `id` | Stable item id |
| `question` | Closed-book question |
| `parametric_answer` | Gold parametric / memory answer (filter target) |
| `context_answer` | Answer supported by the conflicting passage |
| `passage_agrees` | Context consistent with parametric answer |
| `passage_conflicts` | Context that contradicts parametric answer |
| `source` | Provenance string |

**Filtering (done in scripts, not necessarily in the file):**

1. Closed-book exact match to `parametric_answer` at FP16 **and** at target precision (MR★ double filter).
2. Drop items where `passage_conflicts` is not a genuine contradiction.

For a **20-item smoke file**, copy the first 20 converted rows or use `data/conflictqa_smoke.jsonl` included for offline hook tests.

---

## ConflictBank (Stage 1 comparability — OAR)

| Field | Value |
|-------|--------|
| **QA pairs** | [Warrieryes/CB_qa](https://huggingface.co/datasets/Warrieryes/CB_qa) |
| **Claim–evidence** | [Warrieryes/CB_claim_evidence](https://huggingface.co/datasets/Warrieryes/CB_qa) |
| **GitHub** | [zhaochen0110/conflictbank](https://github.com/zhaochen0110/conflictbank) |
| **Paper** | Su et al., NeurIPS 2024 D&B — *ConflictBank* |

```python
from datasets import load_dataset
qa = load_dataset("Warrieryes/CB_qa")
ce = load_dataset("Warrieryes/CB_claim_evidence")
```

Save as **`data/conflictbank_filtered.jsonl`**:

```json
{
  "id": "cb_001",
  "question": "...",
  "parametric_answer": "...",
  "context_answer": "...",
  "passage_agrees": "...",
  "passage_conflicts": "...",
  "conflict_type": "misinformation",
  "source": "ConflictBank/CB_qa"
}
```

Map ConflictBank fields to the same six core keys as ConflictQA so `stage1_behavioral/` can share loaders. Report **OAR** (original-answer ratio under conflict) alongside MR for comparability with Su et al.

---

## JuICE / ParaConflict (Stage 2 on 8B — not for 3050 Ti)

| Field | Value |
|-------|--------|
| **Dataset** | [gaotang/ParaConfilct](https://huggingface.co/datasets/gaotang/ParaConfilct) |
| **Code** | [GaotangLi/JUICE](https://github.com/GaotangLi/JUICE) |

Defer until A4500 + Llama-3.1-8B gate. Clone JUICE separately under `external/JUICE/` (see `juice_repro/their_setup_smoke.py`).

---

## Scoring reminder

Conflict outputs are labeled **parametric / context / other** (three-way). MR = fraction parametric on `context_conflicts` prompts. MR★ = MR on items with correct closed-book at both FP16 and the precision under test.
