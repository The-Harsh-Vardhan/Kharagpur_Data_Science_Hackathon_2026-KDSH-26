def contradiction_score(llm_call, claim, evidence):
    """
    Uses an LLM to classify contradiction vs support vs neutral.
    The llm_call function should return raw text output.
    """

    prompt = f"""
You are a logical consistency checker.

Claim:
{claim}

Novel Evidence:
{evidence}

Instructions:
- Use ONLY the provided evidence.
- Do NOT infer missing facts.
- If the evidence is insufficient, answer NEUTRAL.

Respond with exactly ONE word:
CONTRADICT, SUPPORT, or NEUTRAL
"""

    response = llm_call(prompt).strip().upper()

    mapping = {
        "CONTRADICT": 1.0,
        "NEUTRAL": 0.3,
        "SUPPORT": 0.0
    }

    return mapping.get(response, 0.3)
