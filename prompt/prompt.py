from prompt.detect_intent import INTENT_TEMPLATE
from prompt.extract_entity import ENTITY_TEMPLATE
from typing import List, Dict, Tuple

def RenderDetectIntentSystemPrompt(
    intents: str,
    tuple_delimiter: str = "<||>",
    record_delimiter: str = "##",
    completed_delimiter: str = "<|COMPLETED|>",
) -> str:
    prompt = INTENT_TEMPLATE
    prompt = prompt.replace("{{.Intents}}", intents.strip())
    prompt = prompt.replace("{{.TupleDelimiter}}", tuple_delimiter)
    prompt = prompt.replace("{{.RecordDelimiter}}", record_delimiter)
    prompt = prompt.replace("{{.CompletedDelimiter}}", completed_delimiter)
    return prompt


def RenderExtractEntitySystemPrompt(
    intents: str, # IntentCode from detect_intent output
    entities: List[Dict[str, str]], # list of {"name": ..., "type": ..., "description": ...}
    tuple_delimiter: str = "<||>",
    record_delimiter: str = "##",
    completed_delimiter: str = "<|COMPLETED|>",
) -> str:
    prompt = ENTITY_TEMPLATE
    prompt = prompt.replace("{{.Intents}}", intents.strip())
    prompt = prompt.replace("{{.Entities}}", entities.strip())
    prompt = prompt.replace("{{.TupleDelimiter}}", tuple_delimiter)
    prompt = prompt.replace("{{.RecordDelimiter}}", record_delimiter)
    prompt = prompt.replace("{{.CompletedDelimiter}}", completed_delimiter)
    return prompt


def RenderConversationContextPrompt(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """
    messages: list of {"role": "assistant"|"admin"|"user", "text": "..."}
    Returns: (prompt_str, current_user_text)
    """
    lines = ["<conversation_context>"]
    current = ""
    latest = ""

    for msg in messages:
        role = (msg.get("role") or "user").lower()
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        latest = text

        if role == "assistant":
            lines.append(f"AssistantMessage({text})")
        elif role == "admin":
            lines.append(f"AdminMessage({text})")
        else:
            lines.append(f"UserMessage({text})")
            if not current:
                current = text

    if not current:
        current = latest

    lines.append("</conversation_context>")
    lines.append("<current_message_to_analyze>")
    lines.append(f"UserMessage({current})")
    lines.append("</current_message_to_analyze>")

    return "\n".join(lines) + "\n", current
