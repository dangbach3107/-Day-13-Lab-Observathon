"""YOUR mitigation + observability layer. The simulator calls mitigate() around the
opaque agent (a REAL LLM) for every request. This is the ONLY place observability can
live -- the agent is silent. Legal moves: retry / cache / route / guardrail / sanitize
/ fallback / session-reset / PROMPT ROUTING, plus your own logging/tracing/metrics.
Illegal: hardcoding answers, importing the agent internals, reading instructor files,
network exfiltration.

  call_next(question, config) -> result   # the only way to reach the black box
  context = {"session_id","turn_index","qid","cache": <shared dict>, "cache_lock": <Lock>}
  result  = {"answer","status","steps","trace","meta":{latency_ms,usage,...}}

PROMPT ROUTING: you can override the agent's system prompt PER REQUEST by setting it in
the config you pass to call_next, e.g.:
    conf = dict(config); conf["system_prompt"] = my_better_prompt
    result = call_next(question, conf)
(Or just edit solution/prompt.txt for a single static prompt used on every request.)
"""
from __future__ import annotations

import re
import time

# Safe imports — telemetry may not be on the path when the binary loads us
try:
    from telemetry.logger import logger, new_correlation_id, set_correlation_id
    from telemetry.cost import cost_from_usage
    from telemetry.redact import redact
    _HAS_TELEMETRY = True
except Exception:
    _HAS_TELEMETRY = False
    logger = None

    def new_correlation_id():
        return ""

    def set_correlation_id(_):
        pass

    def cost_from_usage(_m, _u):
        return 0.0

    def redact(s):
        return (s, 0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 3
BACKOFF_BASE_MS = 200
DRIFT_RESET_TURN = 6

_RETRIABLE = {"error", "tool_error", "max_steps", "loop", "no_action", "wrapper_error"}

# Simple PII patterns as fallback if telemetry.redact is unavailable
_EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
_PHONE_RE = re.compile(r'\b(?:\+84|0)\d{9}\b')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_input(question):
    """Strip potential injection payloads hidden in order notes / GHI CHU."""
    cleaned = re.sub(
        r'(?i)(ghi\s*ch[uú][\s:]*)(.*)',
        lambda m: m.group(1) + re.sub(
            r'(?i)(gi[aá]\s*(l[aà]|=|:)\s*\d[\d.,]*'
            r'|price\s*(is|=|:)\s*\d[\d.,]*'
            r'|system\s*:'
            r'|instruction\s*:'
            r'|ignore\s+(above|previous)'
            r'|b[oỏ]\s*qua)',
            '[DATA]',
            m.group(2),
        ),
        question,
    )
    return cleaned


def _detect_loop(trace):
    """Return True if the trace contains a repeated identical action >= 3 times."""
    if not trace or not isinstance(trace, list):
        return False
    actions = []
    for step in trace:
        if isinstance(step, dict):
            actions.append(step.get("action") or step.get("tool") or "")
    if len(actions) < 3:
        return False
    tail = actions[-6:]
    for a in set(tail):
        if a and tail.count(a) >= 3:
            return True
    return False


def _cache_key(question):
    """Normalise question for cache lookup."""
    return re.sub(r'\s+', ' ', question.strip().lower())


def _redact_answer(answer):
    """Redact PII from answer, using telemetry.redact or fallback regex."""
    if not answer:
        return answer, 0
    cleaned, count = redact(answer)
    # Fallback: also catch with simple regex if telemetry missed anything
    if not _HAS_TELEMETRY:
        cleaned, n1 = _EMAIL_RE.subn('[REDACTED:EMAIL]', cleaned)
        cleaned, n2 = _PHONE_RE.subn('[REDACTED:PHONE]', cleaned)
        count += n1 + n2
    return cleaned, count


def _log(event_type, data):
    """Safe log — no-op if telemetry is unavailable."""
    if logger:
        try:
            logger.log_event(event_type, data)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def mitigate(call_next, question, config, context):
    """Observability + mitigation wrapper around the opaque agent."""

    cid = new_correlation_id()
    set_correlation_id(cid)

    qid = context.get("qid", "?")
    session_id = context.get("session_id", "?")
    turn_index = context.get("turn_index", 0)
    cache = context.get("cache")
    cache_lock = context.get("cache_lock")

    # ── 1. CHECK CACHE (thread-safe) ─────────────────────────────────────
    if cache is not None:
        ck = _cache_key(question)
        if cache_lock:
            with cache_lock:
                cached = cache.get(ck)
        else:
            cached = cache.get(ck)

        if cached is not None:
            _log("CACHE_HIT", {"qid": qid, "session_id": session_id})
            return cached
    else:
        ck = None

    # ── 2. SANITIZE INPUT (injection defense) ────────────────────────────
    clean_q = _sanitize_input(question)

    # ── 3. BUILD CONFIG OVERRIDES ────────────────────────────────────────
    conf = dict(config)

    # Reset drifting sessions
    if turn_index >= DRIFT_RESET_TURN:
        conf["context_reset_every"] = 1

    # ── 4. CALL AGENT WITH RETRY ─────────────────────────────────────────
    result = None
    wall_ms = 0
    attempts = 0
    last_error_status = None

    for attempt in range(MAX_RETRIES):
        attempts = attempt + 1
        t0 = time.time()
        result = call_next(clean_q, conf)
        wall_ms = int((time.time() - t0) * 1000)

        status = result.get("status", "")

        if status == "ok":
            break

        last_error_status = status

        # Loop detected — retry won't help
        trace = result.get("trace", [])
        if _detect_loop(trace):
            _log("LOOP_DETECTED", {"qid": qid, "attempt": attempts})
            break

        # Retriable and not last attempt
        if status in _RETRIABLE and attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF_BASE_MS * (attempt + 1) / 1000.0)
            continue

        break

    # ── 5. POST-PROCESS: REDACT PII FROM ANSWER ─────────────────────────
    answer = result.get("answer") or ""
    cleaned_answer, pii_count = _redact_answer(answer)
    if pii_count > 0:
        result["answer"] = cleaned_answer

    # ── 6. STORE IN CACHE (thread-safe) ──────────────────────────────────
    if cache is not None and ck is not None and result.get("status") == "ok":
        if cache_lock:
            with cache_lock:
                cache[ck] = result
        else:
            cache[ck] = result

    # ── 7. OBSERVABILITY: LOG EVERYTHING ─────────────────────────────────
    meta = result.get("meta", {})
    usage = meta.get("usage", {})
    tools_used = meta.get("tools_used", [])

    _log("AGENT_CALL", {
        "qid": qid,
        "session_id": session_id,
        "turn_index": turn_index,
        "status": result.get("status"),
        "attempts": attempts,
        "last_error_status": last_error_status,
        "wall_ms": wall_ms,
        "latency_ms": meta.get("latency_ms"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cost_usd": cost_from_usage(meta.get("model", ""), usage),
        "model": meta.get("model"),
        "tools_used": tools_used,
        "tool_count": len(tools_used) if isinstance(tools_used, list) else 0,
        "steps": result.get("steps"),
        "pii_redacted": pii_count,
        "input_sanitized": (clean_q != question),
    })

    return result
