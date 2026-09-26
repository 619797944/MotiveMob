from __future__ import annotations

import re
from typing import Dict, List, Optional


# --- recent-history / context helpers ---

def make_recent_k(history: str, k: int = 30) -> str:
    """Extract the last k check-ins from a (possibly multi-day) history string."""
    if history is None:
        return ""

    s = str(history).strip()
    s = re.sub(r"Activities at [^:]+:\s*", "", s)
    s = re.sub(r"\n+", ", ", s).strip(", ")

    pattern = re.compile(
        r"(.*?#\d+)\s+at\s+(\d{2}:\d{2}:\d{2})(?=,\s+.*?#\d+\s+at\s+\d{2}:\d{2}:\d{2}|$)"
    )
    items = pattern.findall(s)
    if not items:
        return s[:500]
    items = items[-k:]
    return "\n".join([f"{poi.lstrip(',').strip()} at {time}" for poi, time in items])


_DAY_START_MIN = 4 * 60  # bin 0 = 04:00:00, matching transformer_judge.time_str_to_bin


def _time_str_to_bin(t_str: str) -> int:
    # Duplicated (not imported) from transformer_judge.time_str_to_bin / location_utils.time_slot
    # -- both are equivalent 10-minute-bin-from-04:00 formulas, but importing either module here
    # would pull in judge-pipeline or global_loc_map.pkl-loading side effects this pure helper
    # doesn't need.
    h, m, _s = map(int, t_str.split(":"))
    total_min = h * 60 + m
    shifted = (total_min - _DAY_START_MIN) % (24 * 60)
    return shifted // 10


def make_earlier_today(steps: List[str], current_state: str) -> str:
    """Same-day check-ins strictly before current_state, for the judge prompt."""
    current_state = str(current_state).strip()
    for i, step in enumerate(steps):
        if step.strip() == current_state:
            return ", ".join(steps[:i])

    # current_state isn't itself in steps (e.g. a generated candidate): fall
    # back to comparing time bins instead of an exact string match.
    try:
        cur_time = current_state.rsplit(" at ", 1)[1]
        cur_bin = _time_str_to_bin(cur_time)
    except Exception:
        return ""

    prev = []
    for step in steps:
        try:
            t = step.rsplit(" at ", 1)[1]
            if _time_str_to_bin(t) < cur_bin:
                prev.append(step)
        except Exception:
            continue
    return ", ".join(prev)


# --- behavior-report rewriting ---

def rewrite_behavior_report(report: str, keep_covid_context: bool = False) -> str:
    """Collapse a raw behavior report to its weekday/weekend/visit-list lines."""
    if report is None:
        return ""

    text = str(report).strip()
    covid_context = ""
    if keep_covid_context:
        covid_context = next(
            (
                line.strip()
                for line in text.splitlines()
                if line.strip().startswith("COVID-19 restriction context")
            ),
            "",
        )

    weekday = re.search(
        r"During weekday,.*?begin your daily trip at (?P<start>\d{2}:\d{2}:\d{2}) "
        r"and end your daily trip at (?P<end>\d{2}:\d{2}:\d{2}), "
        r"you usually visit (?P<start_loc>.+?) at the beginning of the day "
        r"and go to (?P<end_loc>.+?) before returning home\.",
        text,
    )
    weekend = re.search(
        r"During weekend,.*?begin your daily trip at (?P<start>\d{2}:\d{2}:\d{2}) "
        r"and end your daily trip at (?P<end>\d{2}:\d{2}:\d{2}), "
        r"you usually visit (?P<start_loc>.+?) at the beginning of the day "
        r"and go to (?P<end_loc>.+?) before returning home\.",
        text,
    )
    if weekday is None or weekend is None:
        return text

    weekday_line = (
        f"During weekday, you usually begin your daily trip at "
        f"{weekday.group('start_loc')} at {weekday.group('start')} "
        f"and end your daily trip at {weekday.group('end_loc')} at {weekday.group('end')}."
    )
    weekend_line = (
        f"During weekend, you usually begin your daily trip at "
        f"{weekend.group('start_loc')} at {weekend.group('start')} "
        f"and end your daily trip at {weekend.group('end_loc')} at {weekend.group('end')}."
    )

    visits_match = re.search(r"You usually visit\s+(.+?)(?:\.?$)", text)
    visit_lines = ["You usually visit:"]
    if visits_match:
        visits_text = visits_match.group(1).strip().rstrip(".")
        poi_pattern = re.compile(
            r"(.+?#\d+\s+at\s+\d{2}:\d{2}:\d{2})(?=,\s+.+?#\d+\s+at\s+\d{2}:\d{2}:\d{2}|$)"
        )
        anchors = [m.group(1).lstrip(",").strip() for m in poi_pattern.finditer(visits_text)]
        visit_lines.extend(anchors if anchors else [visits_text])

    output_lines = [weekday_line, weekend_line] + visit_lines
    if covid_context:
        output_lines.insert(0, covid_context)
    return "\n".join(output_lines).strip()


# --- chat-template rendering ---

def render_chat_prompt(tokenizer, system_text: str, user_text: str) -> str:
    """Render prompt using the model's own chat template (Llama/Qwen/...).
    Returns prompt text ending at assistant generation position.
    """
    if getattr(tokenizer, "apply_chat_template", None) is not None:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            # Gemma-style templates often reject a separate system role.
            if "system" not in str(exc).lower():
                raise
            merged_user_text = f"{system_text.strip()}\n\n{user_text.strip()}"
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": merged_user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )

    # tokenizer has no chat template: hand-rolled prompt format
    return f"<|system|>\n{system_text}\n<|user|>\n{user_text}\n<|assistant|>\n"


# --- state predictor prompt (was phase1_system_text / phase1_user_text) ---

def state_system_text() -> str:
    return (
        "You are an urban mobility transition model.\n"
        "Given a behavior pattern, recent history checkins, "
        "current location and time, current motivation, today's date, and is_weekend, "
        "predict the next location and time.\n"
        "Follow the required output format exactly."
    )


def state_user_text(report, history, current_state, neg_motivation, date_str, is_weekend) -> str:
    return (
        "<INPUT 0> (Behavior Pattern - long-term prior):\n"
        f"{str(report).strip()}\n\n"

        "<INPUT 1> (Recent History Checkins):\n"
        f"{str(history).strip()}\n\n"

        "<INPUT 2> (Current Location & Time):\n"
        f"{current_state}\n\n"

        "<INPUT 3> (Target Motivation for the NEXT transition):\n"
        f"{neg_motivation}\n\n"

        "<INPUT 4> (Today's Date):\n"
        f"{date_str}\n\n"

        "<INPUT 5> (Is Weekend, 1=yes 0=no):\n"
        f"{is_weekend}\n\n"

        "Task:\n"
        "Predict the NEXT location and time.\n\n"

        "Rules:\n"
        "- Recent history and current state are more important than the behavior pattern.\n"
        "- Use the behavior pattern only as a general long-term reference.\n"
        "- The next timestamp must not be earlier than the current timestamp under the day definition starting at 04:00:00.\n\n"

        "Output format (IMPORTANT):\n"
        "<Location>#<ID> at HH:MM:SS\n"
        "Do NOT output any extra text.\n\n"

        "Example:\n"
        "Park#12 at 09:10:00"
    )


# --- motivation predictor prompt, base version (was phase2_system_text / phase2_user_text) ---

def motivation_system_text() -> str:
    return (
        "You are an urban mobility intention inference model.\n"
        "Given a behavior pattern, recent history Checkins, "
        "the current location and time, today's date, and is_weekend, "
        "infer the motivation for the next transition from the current state.\n"
        "Output exactly one label from the provided list."
    )


def motivation_user_text(
    report: str, history: str, original: str, date_str: str, is_weekend: str
) -> str:
    return (
        "<INPUT 0> (Behavior Pattern - long-term prior):\n"
        f"{str(report).strip()}\n\n"
        "<INPUT 1> (Recent History Checkins):\n"
        f"{str(history).strip()}\n\n"
        "<INPUT 2> (Current Location & Time):\n"
        f"{str(original).strip()}\n\n"
        "<INPUT 3> (Today's Date):\n"
        f"{str(date_str).strip()}\n\n"
        "<INPUT 4> (Is Weekend, 1=yes 0=no):\n"
        f"{str(is_weekend).strip()}\n\n"
        "Task:\n"
        "Infer the motivation for the next transition.\n\n"
        "Rules:\n"
        "- Recent history is more important than the behavior pattern.\n\n"
        "Allowed labels (choose exactly ONE):\n"
        "- Return\n- Personal Service\n- Entertainment\n- Dining\n- Transit\n- Religious\n"
        "- Education\n- Work\n- Social\n- Shopping\n- Healthcare\n- END\n\n"
        "Output format (IMPORTANT):\n"
        "- Output EXACTLY ONE line.\n"
        "- The line MUST be exactly one of the allowed labels.\n"
        "- Do NOT output any extra text."
    )


# --- candidate judge prompt (was alternative_sample_generation/transformer_llm_judge_sampler.py) ---

def judge_system_text() -> str:
    return (
        "You are a careful urban mobility candidate judge.\n"
        "You must choose exactly one candidate next check-in from a provided candidate list.\n"
        "Do not invent a new place or time."
    )


def judge_user_text(
    report: str,
    recent_history: str,
    earlier_today: str,
    current_state: str,
    current_lat: Optional[float],
    current_lng: Optional[float],
    motivation: str,
    date_str: str,
    is_weekend: str,
    candidates: List[Dict],
    include_location: bool = False,
) -> str:
    cand_lines = []
    for i, c in enumerate(candidates, start=1):
        metadata = [f"transformer_score={c['score']:.6f}"]
        if include_location:
            if c.get("candidate_lat") is not None and c.get("candidate_lng") is not None:
                metadata.append(
                    f"lat={c['candidate_lat']:.6f}, lng={c['candidate_lng']:.6f}"
                )
            if c.get("distance_km") is not None:
                metadata.append(f"distance_from_current_km={c['distance_km']:.3f}")
        cand_lines.append(f"{i}. {c['text']}  [{'; '.join(metadata)}]")

    current_location_line = str(current_state).strip()
    if include_location and current_lat is not None and current_lng is not None:
        current_location_line += f"  [lat={current_lat:.6f}, lng={current_lng:.6f}]"

    return (
        "<INPUT 0> Behavior Pattern:\n"
        f"{str(report).strip()}\n\n"
        "<INPUT 1> Recent History Before Today:\n"
        f"{str(recent_history).strip() or 'None'}\n\n"
        "<INPUT 2> Earlier Check-ins Today:\n"
        f"{str(earlier_today).strip() or 'None'}\n\n"
        "<INPUT 3> Current Location & Time:\n"
        f"{current_location_line}\n\n"
        "<INPUT 4> Target Motivation for the NEXT transition:\n"
        f"{str(motivation).strip()}\n\n"
        "<INPUT 5> Date and Weekend Flag:\n"
        f"date={date_str}, is_weekend={is_weekend}\n\n"
        "<CANDIDATES>\n"
        + "\n".join(cand_lines)
        + "\n\n"
        "Task:\n"
        "Choose the single best candidate for the next check-in. Prefer candidates that fit "
        "the target motivation, the user's recent behavior, the current state, and temporal order.\n\n"
        "Output format:\n"
        "Output ONLY the candidate number, such as 1 or 7. Do not output any other text."
    )
