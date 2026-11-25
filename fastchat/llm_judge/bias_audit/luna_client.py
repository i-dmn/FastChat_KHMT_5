"""Minimal OpenAI client that sends the SAME request as common.chat_completion_openai.

Why standalone: importing common.py pulls in fastchat.model -> torch. These
experiments only call the API, so we replicate the request exactly instead:
  - messages = [system, user]   (common.py: conv.set_system_message + one user turn)
  - gpt-6-* : max_completion_tokens, reasoning_effort="none" for gpt-6-luna, NO temperature
  - others  : max_tokens, temperature
Requires: pip install "openai<1"   and   export OPENAI_API_KEY=...
"""
import os
import re
import time

API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

one_score_pattern = re.compile(r"\[\[(\d+\.?\d*)\]\]")
one_score_pattern_backup = re.compile(r"\[(\d+\.?\d*)\]")

DRY_RUN = False


def _stub(model, system, user):
    """Deterministic fake output so the whole pipeline can be tested without a key."""
    if "Rephrase" in user or "rephrase" in user:
        return "1. (rephrased item one)\n2. (rephrased item two)\n3. (rephrased item three)"
    if "summary" in user.lower() and "Write" in user:
        return "In summary, the answer above covers the points already listed."
    if "[[rating]]" in system + user:
        return "Stub judgment. Rating: [[5]]"
    if "[[A]]" in system + user:
        return "Stub judgment. [[" + "ABC"[hash(user) % 3] + "]]"
    return "Stub answer: lorem ipsum dolor sit amet."


def chat(model, system, user, temperature=0.0, max_tokens=2048):
    if DRY_RUN:
        return _stub(model, system, user)
    import openai  # openai<1, as required by FastChat llm_judge

    if os.environ.get("OPENAI_API_BASE"):
        openai.api_base = os.environ["OPENAI_API_BASE"]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    args = {"model": model, "messages": messages, "n": 1}
    if model.startswith("gpt-6-"):
        args["max_completion_tokens"] = max_tokens
        if model == "gpt-6-luna":
            args["reasoning_effort"] = "none"
    else:
        args["max_tokens"] = max_tokens
        args["temperature"] = temperature
    for _ in range(API_MAX_RETRY):
        try:
            r = openai.ChatCompletion.create(**args)
            return r["choices"][0]["message"]["content"]
        except openai.error.InvalidRequestError:
            raise
        except openai.error.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
    return API_ERROR_OUTPUT


def parse_pair(judgment):
    """Same rule as common.run_judge_pair for output_format [[A]]."""
    if "[[A]]" in judgment:
        return "A"
    if "[[B]]" in judgment:
        return "B"
    if "[[C]]" in judgment:
        return "tie"
    return "error"


def parse_single(judgment):
    m = re.search(one_score_pattern, judgment) or re.search(one_score_pattern_backup, judgment)
    return float(m.groups()[0]) if m else -1
