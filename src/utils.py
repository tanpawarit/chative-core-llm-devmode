import html 
from dataclasses import dataclass
from typing import List, Optional 

@dataclass
class IntentInfo:
    code: str
    confidence: float
    priority_score: float

@dataclass
class LanguageInfo:
    code: str
    confidence: float
    primary: bool

@dataclass
class SentimentInfo:
    sentiment: str
    confidence: float

@dataclass
class DetectIntentResult:
    intents: List[IntentInfo]
    languages: List[LanguageInfo]
    sentiment: Optional[SentimentInfo]


def intentParser(
    raw_output: str,
    tuple_delimiter: str = "<||>",
    record_delimiter: str = "##",
    completed_delimiter: str = "<|COMPLETED|>",
) -> DetectIntentResult:
    """Parse the LLM TSV-style output into structured intent data."""

    if raw_output is None:
        raise ValueError("detect-intent output is None")

    text = html.unescape(raw_output)
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise ValueError("detect-intent output is empty")

    if completed_delimiter in text:
        # Ignore everything after the completion marker.
        text = text.split(completed_delimiter, 1)[0]

    records = [segment.strip() for segment in text.split(record_delimiter)]

    result = DetectIntentResult(intents=[], languages=[], sentiment=None)

    def _to_float(value: str) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for record in records:
        if not record:
            continue

        cleaned = record.strip()
        if not cleaned:
            continue

        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1].strip()
        else:
            cleaned = cleaned.strip("() ")

        if not cleaned:
            continue

        parts = [part.strip() for part in cleaned.split(tuple_delimiter)]
        if not parts:
            continue

        key = parts[0].lower()

        if key == "intent" and len(parts) >= 4:
            code = parts[1]
            confidence = _to_float(parts[2])
            priority = _to_float(parts[3])
            if not code or confidence is None or priority is None:
                continue
            result.intents.append(IntentInfo(code=code, confidence=confidence, priority_score=priority))

        elif key == "language" and len(parts) >= 4:
            code = parts[1]
            confidence = _to_float(parts[2])
            primary_flag = (parts[3] or "").strip().lower()
            if not code or confidence is None:
                continue
            primary = primary_flag in {"1", "true", "yes", "primary"}
            result.languages.append(LanguageInfo(code=code, confidence=confidence, primary=primary))

        elif key == "sentiment" and len(parts) >= 3:
            label = parts[1]
            confidence = _to_float(parts[2])
            if not label or confidence is None:
                continue
            candidate = SentimentInfo(sentiment=label, confidence=confidence)
            if result.sentiment is None or result.sentiment.confidence < candidate.confidence:
                result.sentiment = candidate

    if not result.intents and not result.languages and result.sentiment is None:
        raise ValueError("no valid intent, language, or sentiment found in detect-intent output")

    return result
