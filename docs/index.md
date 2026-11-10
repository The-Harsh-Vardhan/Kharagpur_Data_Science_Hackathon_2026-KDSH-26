# KDSH'26 Track A: Narrative Consistency Reasoning

A pipeline for the Kharagpur Data Science Hackathon 2026 that decides whether a hypothetical character backstory is logically consistent with a full novel (100k+ words).

## How it works
1. Chunk each novel (800 tokens, 100 overlap) and embed with `all-mpnet-base-v2`
2. Index in FAISS and retrieve the top-5 passages for each backstory claim
3. Label each claim-evidence pair `CONTRADICT`, `SUPPORT` or `NEUTRAL` with Gemini
4. Aggregate to max score, mean score and strong-contradiction count
5. Classify with Logistic Regression: `1` consistent, `0` contradicts

## Status
No measured accuracy is committed to the repository. The 70-80%+ figures in the README are design targets.

## Pages
- [Architecture](architecture.md)
- [Full README and code](https://github.com/The-Harsh-Vardhan/Kharagpur_Data_Science_Hackathon_2026-KDSH-26)
