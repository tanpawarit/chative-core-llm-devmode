import json
from typing import Dict, List, Tuple, Union, Optional

from .detect_intent import INTENT_TEMPLATE
from .extract_entity import ENTITY_TEMPLATE
from .response import RESPONSE_TEMPLATE
from .response_entity_fallback import RESPONSE_ENTITY_FALLBACK_TEMPLATE

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
    # Action protocol (simplified)
    action: str = "general_response",
    allowed_tools: Optional[str] = None,
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
        "{{.Action}}": action or "general_response",
        "{{.AllowedTools}}": (allowed_tools or "none"),
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


def RenderResponseEntityFallbackPrompt(
    *,
    intent: str,
    language: str,
    sentiment: str,
    entities_json: str,
    missing_entities_json: str,
    formality: str = "friendly",
    instruction: str = (
        "Respond helpfully and accurately to the user's request while following business-appropriate tone and policies."
    ),
    restriction: str = (
        "Do not perform irreversible actions. Do not share confidential data. Use tools only when they improve factual accuracy."
    ),
) -> str:
    """Render the response-entity fallback system prompt from the template."""

    prompt = RESPONSE_ENTITY_FALLBACK_TEMPLATE

    language_block_template = '''{{if eq .Language "Thai"}}
Thai Language Guidelines:
- Use natural, conversational Thai appropriate to business context
- Apply correct politeness particles based on formality level
- When asking for clarification, use polite question forms (e.g., "ช่วยบอกเพิ่มเติมได้ไหมครับ/ค่ะ")
- Frame information requests as collaborative, not interrogative
- Avoid excessive punctuation or emojis
- Match cultural expectations for professional communication
{{else if eq .Language "English"}}
English Language Guidelines:
- Professional but warm communication style
- Clear, direct sentences
- When requesting information, use inviting language ("Could you share...", "To help me better...")
- Frame gaps as partnership opportunities, not deficiencies
- Avoid corporate jargon unless context demands it
- No unnecessary punctuation or emojis
{{else}}
General Language Guidelines:
- Respond in {{.Language}} exclusively
- Maintain professional clarity
- Adapt information requests to cultural communication norms
- Avoid mixing languages unless user does so
{{end}}'''

    language_value = (language or "").strip()
    normalized_language = language_value.lower()

    if normalized_language == "thai" or language_value == "th":
        language_block = (
            "Thai Language Guidelines:\n"
            "- Use natural, conversational Thai appropriate to business context\n"
            "- Apply correct politeness particles based on formality level\n"
            "- When asking for clarification, use polite question forms (e.g., \"ช่วยบอกเพิ่มเติมได้ไหมครับ/ค่ะ\")\n"
            "- Frame information requests as collaborative, not interrogative\n"
            "- Avoid excessive punctuation or emojis\n"
            "- Match cultural expectations for professional communication"
        )
    elif normalized_language == "english" or language_value == "en":
        language_block = (
            "English Language Guidelines:\n"
            "- Professional but warm communication style\n"
            "- Clear, direct sentences\n"
            "- When requesting information, use inviting language (\"Could you share...\", \"To help me better...\")\n"
            "- Frame gaps as partnership opportunities, not deficiencies\n"
            "- Avoid corporate jargon unless context demands it\n"
            "- No unnecessary punctuation or emojis"
        )
    else:
        language_block = (
            "General Language Guidelines:\n"
            "- Respond in {{.Language}} exclusively\n"
            "- Maintain professional clarity\n"
            "- Adapt information requests to cultural communication norms\n"
            "- Avoid mixing languages unless user does so"
        )

    prompt = prompt.replace(language_block_template, language_block)

    formality_block_template = '''{{if eq .Formality "formal"}}
Formal Protocol:
- Use professional business language and complete sentences
- Frame information requests with maximum courtesy
- "To provide you with the most accurate assistance, may I ask..."
- "Would you be able to specify..."
- Avoid contractions; maintain grammatical precision
- Employ structured, polished phrasing
- Address user with appropriate titles when applicable
- Maintain respectful distance even when clarifying
{{else if eq .Formality "friendly"}}
Friendly Protocol:
- Warm and approachable tone
- Frame clarifications naturally: "Just to make sure I understand...", "Quick question..."
- Balance professionalism with personability
- Moderate use of contractions acceptable
- Make information gathering feel like conversation, not interrogation
- Show genuine helpfulness when asking for details
{{else if eq .Formality "casual"}}
Casual Protocol:
- Relaxed, everyday conversation style
- Simple clarifications: "Just checking - are you asking about...", "Quick thing..."
- Natural contractions and colloquialisms
- Down-to-earth and relatable tone
- Make asks feel effortless: "Mind sharing...", "What's the..."
- Like asking a colleague for quick context
{{else if eq .Formality "playful"}}
Playful Protocol:
- Light, engaging tone even when gathering information
- Creative framing: "Let me make sure I've got this right...", "Ooh, tell me more about..."
- Turn clarifications into engagement opportunities
- Enthusiastic but not overwhelming
- Keep the energy up while staying focused
- Make information gathering feel collaborative and fun
{{else}}
Default Formality Protocol:
- Adapt to context; default to friendly formality
- Mirror the user's question-asking style when clarifying
{{end}}'''

    normalized_formality = (formality or "").lower()
    if normalized_formality == "formal":
        formality_block = (
            "Formal Protocol:\n"
            "- Use professional business language and complete sentences\n"
            "- Frame information requests with maximum courtesy\n"
            "- \"To provide you with the most accurate assistance, may I ask...\"\n"
            "- \"Would you be able to specify...\"\n"
            "- Avoid contractions; maintain grammatical precision\n"
            "- Employ structured, polished phrasing\n"
            "- Address user with appropriate titles when applicable\n"
            "- Maintain respectful distance even when clarifying"
        )
    elif normalized_formality == "friendly":
        formality_block = (
            "Friendly Protocol:\n"
            "- Warm and approachable tone\n"
            "- Frame clarifications naturally: \"Just to make sure I understand...\", \"Quick question...\"\n"
            "- Balance professionalism with personability\n"
            "- Moderate use of contractions acceptable\n"
            "- Make information gathering feel like conversation, not interrogation\n"
            "- Show genuine helpfulness when asking for details"
        )
    elif normalized_formality == "casual":
        formality_block = (
            "Casual Protocol:\n"
            "- Relaxed, everyday conversation style\n"
            "- Simple clarifications: \"Just checking - are you asking about...\", \"Quick thing...\"\n"
            "- Natural contractions and colloquialisms\n"
            "- Down-to-earth and relatable tone\n"
            "- Make asks feel effortless: \"Mind sharing...\", \"What's the...\"\n"
            "- Like asking a colleague for quick context"
        )
    elif normalized_formality == "playful":
        formality_block = (
            "Playful Protocol:\n"
            "- Light, engaging tone even when gathering information\n"
            "- Creative framing: \"Let me make sure I've got this right...\", \"Ooh, tell me more about...\"\n"
            "- Turn clarifications into engagement opportunities\n"
            "- Enthusiastic but not overwhelming\n"
            "- Keep the energy up while staying focused\n"
            "- Make information gathering feel collaborative and fun"
        )
    else:
        formality_block = (
            "Default Formality Protocol:\n"
            "- Adapt to context; default to friendly formality\n"
            "- Mirror the user's question-asking style when clarifying"
        )

    prompt = prompt.replace(formality_block_template, formality_block)

    sentiment_block_template = '''{{if eq .Sentiment "negative"}}
Negative Sentiment Handling:
- **CRITICAL**: Do not add friction with information requests unless absolutely essential
- If clarification needed: Acknowledge frustration first, then minimal ask
- Example: "I understand this is frustrating. To help resolve this quickly, could you confirm [one thing]?"
- Prioritize providing partial value over gathering perfect information
- Lead with empathy and acknowledgment
- Focus immediately on solutions or next steps
- Keep responses solution-oriented and supportive
{{else if eq .Sentiment "positive"}}
Positive Sentiment Handling:
- Information gathering can be energetic and collaborative
- Match enthusiasm appropriately
- Frame asks as partnership: "Great! To make this even better..."
- Build on positive momentum
- Keep the positive energy flowing
{{else if eq .Sentiment "neutral"}}
Neutral Sentiment Handling:
- Clear, direct information requests when needed
- Professional baseline tone
- Straightforward: "To assist you accurately, I need..."
- Efficient and respectful of user's time
{{else}}
Default Sentiment Handling:
- Professional and helpful baseline
- Adapt if sentiment shifts during conversation
{{end}}'''

    normalized_sentiment = (sentiment or "").lower()
    if normalized_sentiment == "negative":
        sentiment_block = (
            "Negative Sentiment Handling:\n"
            "- **CRITICAL**: Do not add friction with information requests unless absolutely essential\n"
            "- If clarification needed: Acknowledge frustration first, then minimal ask\n"
            "- Example: \"I understand this is frustrating. To help resolve this quickly, could you confirm [one thing]?\"\n"
            "- Prioritize providing partial value over gathering perfect information\n"
            "- Lead with empathy and acknowledgment\n"
            "- Focus immediately on solutions or next steps\n"
            "- Keep responses solution-oriented and supportive"
        )
    elif normalized_sentiment == "positive":
        sentiment_block = (
            "Positive Sentiment Handling:\n"
            "- Information gathering can be energetic and collaborative\n"
            "- Match enthusiasm appropriately\n"
            "- Frame asks as partnership: \"Great! To make this even better...\"\n"
            "- Build on positive momentum\n"
            "- Keep the positive energy flowing"
        )
    elif normalized_sentiment == "neutral":
        sentiment_block = (
            "Neutral Sentiment Handling:\n"
            "- Clear, direct information requests when needed\n"
            "- Professional baseline tone\n"
            "- Straightforward: \"To assist you accurately, I need...\"\n"
            "- Efficient and respectful of user's time"
        )
    else:
        sentiment_block = (
            "Default Sentiment Handling:\n"
            "- Professional and helpful baseline\n"
            "- Adapt if sentiment shifts during conversation"
        )

    prompt = prompt.replace(sentiment_block_template, sentiment_block)

    replacements = {
        "{{.Intent}}": intent or "",
        "{{.Language}}": language_value,
        "{{.Sentiment}}": sentiment or "",
        "{{.Formality}}": formality or "friendly",
        "{{.Instruction}}": instruction or "",
        "{{.Restriction}}": restriction or "",
        "{{.Entities}}": entities_json or "[]",
        "{{.MissingEntities}}": missing_entities_json or "[]",
    }

    for key, value in replacements.items():
        prompt = prompt.replace(key, value)

    return prompt
