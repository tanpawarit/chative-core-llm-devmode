import json
from typing import Dict, List, Tuple, Union, Optional

from .detect_intent import INTENT_TEMPLATE
from .extract_entity import ENTITY_TEMPLATE
from .response import RESPONSE_TEMPLATE

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
    intent: str,  # IntentCode from detect_intent output
    entities: Union[str, List[Dict[str, str]]],  # list of {"name": ..., "type": ..., "description": ...}
    tuple_delimiter: str = "<||>",
    record_delimiter: str = "##",
    completed_delimiter: str = "<|COMPLETED|>",
) -> str:
    prompt = ENTITY_TEMPLATE
    prompt = prompt.replace("{{.Intent}}", intent.strip())
    if isinstance(entities, str):
        entities_str = entities.strip()
    else:
        entities_str = json.dumps(entities, ensure_ascii=False, indent=2)
    prompt = prompt.replace("{{.Entities}}", entities_str)
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


def RenderResponseSystemPrompt(
    *,
    intent: str,
    language: str,
    sentiment: str,
    entities: Optional[str] = None,
    formality: str = "friendly",
    knowledge: Optional[str] = None,
    instruction: str = "Respond helpfully and accurately to the user's request while following business-appropriate tone and policies.",
    restriction: str = "Do not perform irreversible actions. Do not share confidential data. Use tools only when they improve factual accuracy.",
) -> str:
    """Render the response system prompt from the template.

    The template uses simple ``{{.Key}}`` placeholders similar to the other
    render helpers in this module.
    """

    prompt = RESPONSE_TEMPLATE
    replacements = {
        "{{.Intent}}": intent or "",
        "{{.Language}}": language or "",
        "{{.Sentiment}}": sentiment or "",
        "{{.Formality}}": formality or "friendly",
        "{{.Instruction}}": instruction or "",
        "{{.Restriction}}": restriction or "",
    }

    # Optional sections
    if entities:
        prompt = prompt.replace("{{- if .Entities}}\n- Entities: {{.Entities}}\n{{- end}}", f"- Entities: {entities}")
        prompt = prompt.replace("{{- if .Entities}}\n2. Context: Reference relevant entities ({{.Entities}}) naturally when applicable per entity protocol\n{{- end}}", "2. Context: Reference relevant entities naturally when applicable per entity protocol")
        prompt = prompt.replace("{{- if .Entities}}\n- Available entities: {{.Entities}} (integrate per entity protocol)\n{{- end}}", f"- Available entities: {entities} (integrate per entity protocol)")
    else:
        prompt = prompt.replace("{{- if .Entities}}\n- Entities: {{.Entities}}\n{{- end}}", "")
        prompt = prompt.replace("{{- if .Entities}}\n2. Context: Reference relevant entities ({{.Entities}}) naturally when applicable per entity protocol\n{{- end}}", "")
        prompt = prompt.replace("{{- if .Entities}}\n- Available entities: {{.Entities}} (integrate per entity protocol)\n{{- end}}", "")

    if knowledge:
        prompt = prompt.replace("{{- if .Knowledge}}\n<retrieved_knowledge>\nUse the following knowledge snippets as authoritative context. Reference only when relevant and avoid speculation beyond this material.\n{{.Knowledge}}\n</retrieved_knowledge>\n{{- end}}", f"<retrieved_knowledge>\nUse the following knowledge snippets as authoritative context. Reference only when relevant and avoid speculation beyond this material.\n{knowledge}\n</retrieved_knowledge>")
    else:
        prompt = prompt.replace("{{- if .Knowledge}}\n<retrieved_knowledge>\nUse the following knowledge snippets as authoritative context. Reference only when relevant and avoid speculation beyond this material.\n{{.Knowledge}}\n</retrieved_knowledge>\n{{- end}}", "")

    for key, value in replacements.items():
        prompt = prompt.replace(key, value)

    return prompt
