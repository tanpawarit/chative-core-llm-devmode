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


@dataclass
class EntityInfo:
    Code: str
    OriginalValue: str
    NormalizedValue: str
    StartIndex: int
    EndIndex: int
    Confidence: float


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


def entitiesParser(
    raw_output: str,
    tuple_delimiter: str = "<||>",
    record_delimiter: str = "##",
    completed_delimiter: str = "<|COMPLETED|>",
) -> List[EntityInfo]:
    """Parse entity extraction output into normalized `EntityInfo` records."""

    if raw_output is None:
        raise ValueError("entity output is None")

    text = html.unescape(raw_output).replace("\r\n", "\n").strip()
    if not text:
        raise ValueError("entity output is empty")

    if completed_delimiter and completed_delimiter in text:
        text = text.split(completed_delimiter, 1)[0]

    if not text.strip():
        raise ValueError("entity output is empty")

    if record_delimiter and record_delimiter in text:
        segments = text.split(record_delimiter)
    else:
        segments = text.splitlines()

    entities: List[EntityInfo] = []

    def _to_int(value: str) -> Optional[int]:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _to_float(value: str) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for segment in segments:
        if not segment:
            continue

        cleaned = segment.strip()
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
        if key != "entity" or len(parts) < 7:
            continue

        code = parts[1]
        original_value = parts[2]
        normalized_value = parts[3]
        start_index = _to_int(parts[4])
        end_index = _to_int(parts[5])
        confidence = _to_float(parts[6])

        if not code or original_value is None or normalized_value is None:
            continue
        if start_index is None or end_index is None or confidence is None:
            continue
        if start_index < 0 or end_index < 0 or end_index < start_index:
            continue

        entities.append(
            EntityInfo(
                Code=code,
                OriginalValue=original_value,
                NormalizedValue=normalized_value,
                StartIndex=start_index,
                EndIndex=end_index,
                Confidence=confidence,
            )
        )

    if not entities:
        raise ValueError("no valid entities found in entity output")

    entities.sort(key=lambda item: item.StartIndex)
    return entities
