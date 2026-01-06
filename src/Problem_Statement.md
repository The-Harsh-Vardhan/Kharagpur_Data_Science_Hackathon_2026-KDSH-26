
# Kharagpur Data Science Hackathon 2026

## Problem Statement

---

## 🎯 Motivation

Large language models perform well on local text understanding tasks such as summarization and question answering. However, they struggle with **global consistency over long narratives**, where meaning emerges from how events, states, and constraints accumulate over time.

In long-form text, earlier events restrict what can plausibly happen later. Characters change, commitments are made, and causal pathways are either reinforced or ruled out. Correct reasoning in this setting requires **tracking how these constraints evolve** and determining whether a given future is compatible with a proposed past.

Current models often fail at this type of reasoning. They rely on surface-level plausibility, producing explanations that are locally coherent but globally inconsistent. As a result, they confuse correlation with causation and narrative similarity with logical compatibility.

This challenge is designed to evaluate that failure mode. Participants are not asked to generate text or interpret themes. Instead, the core task is a **decision problem**: given evidence distributed across a long narrative, determine whether a hypothesized past can causally and logically produce an observed future.

Although the task is framed in narrative terms, it ultimately reduces to a **structured classification problem over long contexts**, requiring careful evidence aggregation, constraint tracking, and causal reasoning rather than language generation.

---

## About Pathway

Pathway is building the world’s first frontier model for enterprise.

Its breakthrough architecture — **The Dragon Hatchling** — outperforms transformers and provides enterprises with full visibility into how the model works. Combining the foundational model with the fastest data processing engine on the market, Pathway enables enterprises to move beyond incremental optimization toward truly contextualized, experience-driven intelligence.

Pathway is trusted by organizations such as:

* NATO
* La Poste
* Formula 1 racing teams

### Leadership

* **Zuzanna Stamirowska (CEO)** — complexity scientist
* **Jan Chorowski (CTO)** — early pioneer of Attention for speech, former Google Brain researcher
* **Adrian Kosowski (CSO)** — computer scientist and quantum physicist, co-founder of SPOJ

Pathway is backed by leading investors and advisors, including **Lukasz Kaiser**, co-author of the Transformer architecture.

Headquarters: Palo Alto, California
Offices: Paris, Wroclaw

Pathway invites participants to join their open-source community on GitHub and leverage their tools to transform challenges into solutions.

---

## 📖 The Challenge

You are given **two things**:

1. **A complete long-form narrative**
   - A novel, **100k+ words**

2. **A hypothetical backstory** for one of its central characters
   - Newly written
   - Not part of the novel
   - Deliberately plausible

Your task is to decide whether the proposed backstory is **consistent with the story as a whole**.

The goal is **not** to judge writing quality or check for small textual contradictions. Instead, you must determine whether the backstory respects the **key constraints** established throughout the narrative.

### What the System Is Expected to Demonstrate

- **Consistency over time**  
  Check whether the proposed backstory fits how characters and events develop later in the story.

- **Causal reasoning**  
  Determine whether later events still make sense given the earlier conditions introduced by the backstory.

- **Respect for narrative constraints**  
  Some explanations or coincidences don't fit a story even if they don't directly contradict a sentence.

- **Evidence-based decisions**  
  Conclusions should be supported by signals drawn from multiple parts of the text, not a single convenient passage.

---

## Task Definition

### Input

Each example contains:

#### 1. Narrative

- Full text of a novel
- No summaries
- No truncation

#### 2. Hypothetical Backstory

A character outline describing:

- Early-life events
- Formative experiences
- Beliefs
- Fears
- Ambitions
- Assumptions about the world and its rules

The backstory is intentionally:

- Underspecified in some places
- Overly confident in others

---

### Output

For each example, your system must produce:

#### 1. Consistency Judgment

A binary label:

- **Consistent (1)**
- **Contradict (0)**

#### 2. Comprehensive Evidence Rationale (Optional for Track B)

A structured explanation establishing backstory validity or invalidity, rigorously tested against the primary textual source.

---

## 📋 Structure and Requirements (Evidence Dossier)

The dossier must adhere to strict organizational principles:

### 1. Excerpts from the Primary Text

- Direct, verbatim passages from the novel
- Must be directly relevant to a specific backstory claim

### 2. Explicit Linkage to Backstory Claims

- Every excerpt must be paired with a specific claim
- A single excerpt may constrain multiple claims
- A single claim may require multiple excerpts

### 3. Analysis of Constraint or Refutation

- Concise but thorough explanation per excerpt–claim pair

The integrity of the analysis relies entirely on correct linkage and support structure.

---

## 🛠️ Tech Stack and Implementation Tracks

This challenge supports **two parallel tracks**:  
- Participants must choose **one** track  
- Submissions are evaluated **only within the chosen track**

---

## Track A: Systems Reasoning with NLP and Generative AI

Track A encourages strong, well-engineered solutions using established NLP and GenAI techniques.

### Focus

- Correctness
- Robustness
- Evidence-grounded reasoning
- *Not architectural novelty*

### Technical Requirements

All Track A submissions must use **Pathway’s Python framework** in at least one meaningful part of the pipeline.

Pathway may be used for:

- Ingesting and managing long-context narrative data
- Storing and indexing full novels and metadata
- Retrieval over long documents using vector stores
- Connecting to external data sources
- Acting as a document store or orchestration layer

Beyond this requirement, modeling choices are open:

- Transformer-based LLMs
- Agentic pipelines
- Classical NLP pipelines
- Hybrid symbolic–neural approaches
- Rerankers, classifiers, custom heuristics

---

## Track A: Evaluation Focus

Submissions are evaluated on:

- **Accuracy and robustness** on the core classification task
- **Novelty** in reasoning methods (not end-to-end generation)
- **Handling of long context**, including:
  - Chunking strategies
  - Memory mechanisms
  - Retrieval policies
  - Consistency checks

Straightforward off-the-shelf RAG pipelines are **not** rewarded.

---

## Track B: BDH-Driven Continuous Narrative Reasoning

Track B is intended for teams exploring **Baby Dragon Hatchling (BDH)**–inspired modeling.

### Requirements

Track B submissions must incorporate BDH by:

- Using the open-source BDH architecture
- Adapting BDH to narrative signals
- Producing representations fed into classifiers
- Implementing BDH-inspired reasoning mechanisms:
  - Persistent internal state
  - Sparse updates
  - Incremental belief formation

Large-scale training is **not required**.

---

## Track B: Evaluation Focus

- Accuracy and robustness
- Pretraining and representation learning using BDH
- Clarity in how BDH mechanisms influence decisions

Providing evidence rationale is **optional**.  
Submissions are not penalized for focusing on classification quality.

---

## 📊 Dataset

The dataset consists of:

- Long-form narrative texts (100k+ words)
- Hypothetical backstories

Each novel is provided as a full `.txt` file with no truncation.  
Backstories are intentionally plausible yet potentially inconsistent.

Dataset is hosted via a shared Google Drive link.

---

## 📦 Deliverables and Submission

Submit a single ZIP file named:

```
<TEAMNAME>_KDSH_2026.zip
```

### ZIP must contain:

#### 1. Code (Reproducible)

- End-to-end runnable code
- Reads provided inputs
- Generates predictions without manual steps

#### 2. Report (Max 10 pages, excluding appendix)

Describe:

- Overall approach
- Handling of long context
- Distinguishing causal signals from noise
- Key limitations or failure cases

Clarity matters more than length.

#### 3. Results File (CSV)

- Named `results.csv`
- One row per test example

---

## ✅ Evaluation and Reproducibility

- Submissions may be re-run in a clean environment
- Mismatch between outputs and system results may lead to disqualification
- Evaluation prioritizes **reasoning quality and robustness** over raw performance

### Example Output Format

| Story ID | Prediction | Rationale                                      |
| -------- | ---------- | ---------------------------------------------- |
| 1        | 1          | Earlier economic shock makes outcome necessary |
| 2        | 0          | Proposed backstory contradicts later actions   |

- `1` = Consistent
- `0` = Inconsistent
- Rationale is optional but encouraged (1–2 lines)

---

## 💬 Official Communication Channels

- **WhatsApp**: Community 1, Community 2 (join any one)
- **Discord**: Official channel

---

## 📚 Resources

### Pathway Framework

- Core Engine
- LLM App Templates
- Documentation
- Community Showcases
- Bootcamp

### Connectors and Ingestion

- Connectors overview
- Custom Python connectors
- Artificial data streams

### LLM Integration

- LLM xPack overview
- Pathway Vector Store documentation

### Tutorials

- LangGraph Agents Cookbook

---

## 🐉 BDH Resources

### Official

- Main Repository
- Model Architecture
- Training Script

### Paper (Key Sections)

- BDH architecture and distributed graph dynamics
- GPU tensor formulation
- Interpretability findings
- Experimental validation and scaling laws
- Complete BDH-GPU code listings

### Community Projects

- HuggingFace-compatible BDH wrapper
- MLX port for Apple Silicon
- Burn/Rust port
- Educational visualization fork

---