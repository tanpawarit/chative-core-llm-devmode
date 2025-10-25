import html
import json
import os
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.agents import create_agent

from src.prompt.prompt import (
    RenderDetectIntentSystemPrompt,
    RenderExtractEntitySystemPrompt,
    RenderConversationContextPrompt,
    RenderResponseSystemPrompt,
    RenderResponseEntityFallbackPrompt,
)
from src.utils import DetectIntentResult, IntentInfo, LanguageInfo, SentimentInfo, intentParser, entitiesParser
from src.tools.knowledge import knowledge_search_tool
 
# ---- LangGraph wiring --------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

provider = os.getenv("LLM_PROVIDER", "openai")
model = os.getenv("LLM_MODEL", "gpt-4.1")

# ================================= Global State ===========================================
class IntentLm(TypedDict):
    systemprompt: str                                     
    userprompt: str                                        
    output: str                               

class EntityLm(TypedDict):
    systemprompt: str                                     
    userprompt: str                                        
    output: str                               
 
class State(TypedDict):
    messages: Annotated[list, add_messages]
    conversation_userprompt: str
    intent_model: IntentLm
    entity_model: EntityLm
    missing_entities: List[str]

    intent: str
    language: str
    sentiment: str
    action: str
    response: str

     

# ============================================================================

llm = init_chat_model(f"{provider}:{model}", temperature=0.0)


# Node 1: prepare NLU prompts from history
INTENTS = "greet:0.6, search_product:0.8, farewell:0.6"  

mock_db = [
    {
        "intents": [
            {"name": "greet", "confidence": 0.6}
        ],
        "action": "knowledge_search",
        "entities": [
            {"name": "order_id", "type": "text", "description": "order reference number"},
            {"name": "payment_amount", "type": "currency", "description": "amount paid"},
            {"name": "payment_status", "type": "text", "description": "status of payment"}
        ]
    },
    {
        "intents": [
            {"name": "search_product", "confidence": 0.8}
        ],
        "action": "knowledge_search",
        "entities": [
            {"name": "product_type", "type": "text", "description": "type of product"}, 
        ]
    },
    {
        "intents": [
            {"name": "farewell", "confidence": 0.6}
        ],
        "action": "knowledge_search",
        "entities": [
            {"name": "order_id", "type": "text", "description": "order reference number"},
            {"name": "payment_amount", "type": "currency", "description": "amount paid"},
            {"name": "payment_status", "type": "text", "description": "status of payment"}
        ]
    }
]
def get_action_repo(data, intent_name):
    for item in data:
        if any(i["name"] == intent_name for i in item["intents"]):
            return item["action"]
    return None

def get_entities_repo(data, intent_name):
    for item in data:
        if any(i["name"] == intent_name for i in item["intents"]):
            return item["entities"]
    return []

def get_entities_by_intent_and_action_repo(data, intent_name, action_name):
    result = []
    if not intent_name or not action_name:
        return result
    for item in data:
        has_intent = any(i["name"] == intent_name for i in item["intents"])
        if has_intent and item["action"] == action_name:
            result.extend(item["entities"])
    return result

def intentInputNode(state: State) -> dict: 
    msgs: List[Dict[str, str]] = []
    for m in state["messages"]:
        if isinstance(m, dict):
            role = m.get("role", "user")
            text = (m.get("content") or "").strip()
        elif isinstance(m, BaseMessage):
            role = m.type  # "ai", "human", "system", etc.
            text = (m.content or "").strip()
        else:
            continue
        if not text:
            continue
        # map assistant/admin/user to our tiny schema
        if role in {"assistant", "ai"}:
            mapped_role = "assistant"
        elif role == "system":
            mapped_role = "admin"
        else:
            mapped_role = "user"
        mapped = {"role": mapped_role, "text": text}
        msgs.append(mapped)

    userprompt, _current = RenderConversationContextPrompt(msgs)
    systemprompt = RenderDetectIntentSystemPrompt(INTENTS)

    return {
        "intent_model": {
            "systemprompt": systemprompt,
            "userprompt": userprompt,
            },
        "conversation_userprompt": userprompt,
        }

# Node 2: call LLM with just 2 messages (system + user)
def intentChatmodelNode(state: State) -> dict:
    messages = [
        {"role": "system", "content": state["intent_model"]["systemprompt"]},
        {"role": "user", "content": state["intent_model"]["userprompt"]},
    ]
    result = llm.invoke(messages)
    current = state.get("intent_model", {})
    return {
        "intent_model":{
            **current,
            "output": result.content
            }
        }

# Node 3: prepare NLU prompts from history
def entityInputNode(state: State) -> dict:
    # Use the pre-built conversation prompt from state (set at first node)
    userprompt = state.get("conversation_userprompt", "")
     
    systemprompt = RenderExtractEntitySystemPrompt(state.get("intent"), get_entities_repo(mock_db, state.get("intent")))

    return {
        "entity_model": {
            "systemprompt": systemprompt,
            "userprompt": userprompt,
            }
        }

# Node 4
def entityChatmodelNode(state: State) -> dict:
    messages = [
        {"role": "system", "content": state["entity_model"]["systemprompt"]},
        {"role": "user", "content": state["entity_model"]["userprompt"]},
    ]
    result = llm.invoke(messages)
    current = state.get("entity_model", {})  
    return {
        "entity_model":{
            **current,
            "output": result.content
            }
        }

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

# Node 5: simple action decider based on intent (response or entity)
def actionDeciderNode(state: State) -> dict:
    print("====== Start actionDeciderNode ======")
    intent_output = state["intent_model"]["output"]
    intent_result = intentParser(intent_output)

    # Find the highest confidence intent and save to state
    def _intent_sort_key(item: IntentInfo) -> Tuple[float, float]:
        return (
            item.confidence if item.confidence is not None else float("-inf"),
            item.priority_score if item.priority_score is not None else float("-inf"),
        )

    primary_intent: Optional[IntentInfo] = None
    if intent_result.intents:
        primary_intent = max(intent_result.intents, key=_intent_sort_key)

    # Find the primary language and save to state
    def _language_sort_key(item: LanguageInfo) -> float:
        return item.confidence if item.confidence is not None else float("-inf")

    primary_language: Optional[LanguageInfo] = None
    if intent_result.languages:
        primary_candidates = [lang for lang in intent_result.languages if lang.primary]
        language_pool = (
            [lang for lang in primary_candidates if lang.confidence is not None]
            or [lang for lang in intent_result.languages if lang.confidence is not None]
        )
        if language_pool:
            primary_language = max(language_pool, key=_language_sort_key)

    # Find the highest confidence sentiment and save to state
    dominant_sentiment: Optional[SentimentInfo] = intent_result.sentiment

    # Fetch the default action for the primary intent (i hardcode with ACTION) and save to state
    updates: Dict[str, str] = {}
    if primary_intent:
        updates["intent"] = primary_intent.code
        updates["action"] = get_action_repo(mock_db, primary_intent.code)  
    if primary_language:
        updates["language"] = primary_language.code
    if dominant_sentiment:
        updates["sentiment"] = dominant_sentiment.sentiment

    return updates


def route_after_intent(state: State) -> str:

    if len(get_entities_by_intent_and_action_repo(
        mock_db, state.get("intent"), state.get("action")
    ))!=0:
        print("Routing to entity extraction...")
        return "entityInputNode"
    print("Routing to response...")
    return "responseNode"

def entityEvaluatorNode(state: State) -> dict:
    print("====== Start entityEvaluatorNode ======")
    entity_output = state.get("entity_model", {}).get("output", "")

    try:
        entities_parsed = entitiesParser(entity_output)
    except ValueError:
        entities_parsed = []

    required_entities = get_entities_by_intent_and_action_repo(
        mock_db,
        state.get("intent"),
        state.get("action"),
    )

    observed_codes = {entity.Code for entity in entities_parsed}
    missing_entities = [
        entity.get("name")
        for entity in required_entities
        if entity.get("name") and entity.get("name") not in observed_codes
    ] 
    return  {"missing_entities": missing_entities}

def responseEntityFallbackNode(state: State) -> dict:
    """Generate a graceful response when required entities are missing."""
    print("====== Start responseEntityFallbackNode ======")
    intent = state.get("intent", "unknown_intent")
    language = state.get("language", "English")
    sentiment = state.get("sentiment", "neutral")
    formality = "friendly"
    missing_entities = state.get("missing_entities") or []
    action = state.get("action")

    entities_catalog = get_entities_by_intent_and_action_repo(mock_db, intent, action)

    parsed_entities = []
    try:
        raw_entities = (state.get("entity_model") or {}).get("output")
        if raw_entities:
            parsed_entities = entitiesParser(raw_entities)
    except ValueError:
        parsed_entities = []

    parsed_map = {}
    for entity in parsed_entities:
        value = entity.NormalizedValue or entity.OriginalValue
        parsed_map[entity.Code] = value

    catalog_names = {item.get("name") for item in entities_catalog if item.get("name")}
    entities_payload = []
    for item in entities_catalog:
        name = item.get("name")
        if not name:
            continue
        entities_payload.append(
            {
                "name": name,
                "type": item.get("type", "text"),
                "required": item.get("required", True),
                "description": item.get("description", ""),
                "value": parsed_map.get(name),
            }
        )

    for entity in parsed_entities:
        if entity.Code not in catalog_names:
            entities_payload.append(
                {
                    "name": entity.Code,
                    "type": "text",
                    "required": False,
                    "description": "",
                    "value": entity.NormalizedValue or entity.OriginalValue,
                }
            )

    entities_json = json.dumps(entities_payload, ensure_ascii=False, indent=2)
    missing_entities_json = json.dumps(missing_entities, ensure_ascii=False)

    systemprompt = RenderResponseEntityFallbackPrompt(
        intent=intent,
        language=language,
        sentiment=sentiment,
        entities_json=entities_json,
        missing_entities_json=missing_entities_json,
        formality=formality,
    )

    userprompt = state.get("conversation_userprompt", "")
    messages = [
        SystemMessage(content=systemprompt),
        HumanMessage(content=userprompt),
    ]

    result = llm.invoke(messages)
    response_text = getattr(result, "content", "") or ""

    if not response_text:
        fallback_missing = ", ".join(missing_entities) if missing_entities else "details"
        response_text = (
            f"I'm missing a bit of information to fully resolve this. Could you share the {fallback_missing}?"
        )

    return {"response": response_text}

# response generator with tool-calling (ReAct-style)
def responseNode(state: State) -> dict:
    print("====== Start responseNode ======")
    # Select tools based on routed action (restrict toolset per turn)
    action = state.get("action") or ""
    tools_map = {
        "knowledge_search": [knowledge_search_tool],
        # Future actions can map to their specific tools, e.g.:
        # "checkout": [start_checkout_tool, confirm_payment_tool],
    }
    allowed_tools = tools_map.get(action, [])

    # Prepare context
    intent = state.get("intent", "unknown_intent")
    language = state.get("language", "English")
    sentiment = state.get("sentiment", "neutral")

    # Try parse entities from the entity_model output (if available)
    entities_str = ""
    try:
        raw_entities = (state.get("entity_model") or {}).get("output")
        if raw_entities:
            parsed_entities = entitiesParser(raw_entities)
            # Compact human-readable summary: CODE=NormalizedValue (confidence)
            parts = []
            for e in parsed_entities:
                val = e.NormalizedValue or e.OriginalValue
                parts.append(f"{e.Code}={val}")
            entities_str = ", ".join(parts)
    except Exception:
        # If entity parsing fails, continue without entities
        entities_str = ""

    # Use the conversation prompt built at the start; fallback to build if missing
    userprompt = state.get("conversation_userprompt", "")

    # Render system prompt; knowledge is fetched via tools if the model chooses
    allowed_tool_names = ", ".join([getattr(t, "name", "") for t in allowed_tools]) if allowed_tools else ""
    systemprompt = RenderResponseSystemPrompt(
        intent=intent,
        language=language,
        sentiment=sentiment,
        entities=entities_str or None,
        formality="friendly",
        knowledge=None,
        action=action or "general_response",
        allowed_tools=allowed_tool_names or "none",
    )

    messages = [
        SystemMessage(content=systemprompt),
        HumanMessage(content=userprompt),
    ]

    response_text = ""
    if allowed_tools:
        agent = create_agent(llm, tools=list(allowed_tools))
        agent_state = agent.invoke({"messages": messages})
        final_messages = agent_state.get("messages", [])
        for msg in reversed(final_messages or []):
            if isinstance(msg, AIMessage):
                response_text = msg.content or ""
                if response_text:
                    break
    else:
        result = llm.invoke(messages)
        response_text = getattr(result, "content", "") or ""

    if not response_text:
        response_text = f"Detected intent: {intent}, language: {language}, sentiment: {sentiment}."

    return {"response": response_text}


# Build graph
builder = StateGraph(State)
builder.add_node("intentInputNode", intentInputNode)
builder.add_node("intentChatmodelNode", intentChatmodelNode)
builder.add_node("actionDeciderNode", actionDeciderNode)
builder.add_node("entityInputNode", entityInputNode)
builder.add_node("entityChatmodelNode", entityChatmodelNode)
builder.add_node("entityEvaluatorNode", entityEvaluatorNode)
builder.add_node("responseNode", responseNode)
builder.add_node("responseEntityFallbackNode", responseEntityFallbackNode)


builder.add_edge(START, "intentInputNode")
builder.add_edge("intentInputNode", "intentChatmodelNode")
builder.add_edge("intentChatmodelNode", "actionDeciderNode")
builder.add_conditional_edges(
    "actionDeciderNode",
    route_after_intent,
    {
        "entityInputNode": "entityInputNode",
        "responseNode": "responseNode",
    },
)
builder.add_edge("entityInputNode", "entityChatmodelNode")
builder.add_edge("entityChatmodelNode", "entityEvaluatorNode")
builder.add_conditional_edges(
    "entityEvaluatorNode",
    lambda state: bool(state.get("missing_entities")),
    {
        True: "responseEntityFallbackNode",
        False: "responseNode",
    },
)
builder.add_edge("responseNode", END)
builder.add_edge("responseEntityFallbackNode", END)

graph = builder.compile()

# ---- Generate graph visualization -------------------------------------------------
graph_png = graph.get_graph().draw_mermaid_png()
output_path = Path("graph_mermaid.png")
output_path.write_bytes(graph_png)
print(f"Saved graph visualization to {output_path.resolve()}")
# ---- Demo runner -------------------------------------------------------------
def run_once(user_input: str):
    # Start with a tiny history and feed the new user message
    initial_messages = [
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": user_input},
    ]
    state = {
        "messages": initial_messages,
        "intent_model": {
            "systemprompt": "",
            "userprompt": "",
            "output": "",
        },
        "entity_model": {
            "systemprompt": "",
            "userprompt": "",
            "output": "",
        },
        "missing_entities": [],
        "conversation_userprompt": "",
        "response": "",
    }

    final = graph.invoke(state)
    # print("=== SYSTEM PROMPT ===")
    # print(final["intent_model"]["systemprompt"])
    # print("=== USER PROMPT ===")
    # print(final["intent_model"]["userprompt"])
    # print("=== NLU OUTPUT ===")
    # print(final["intent_model"]["output"]) 
    # print("=== ENTITY SYSTEM PROMPT  ===")
    # print(final["entity_model"]["systemprompt"])
    # print("=== ENTITY USER PROMPT ===")
    # print(final["entity_model"]["userprompt"])
    print("=== ENTITY RESULT ===")
    print(final["entity_model"]["output"])
    print("=== Missing ENTITY RESULT ===")
    print(final["missing_entities"] )  
    print("=== FINAL RESPONSE ===")
    print(final.get("response", ""))

if __name__ == "__main__":
    run_once("MBTI คืออะไรหรอครับ มีหนังสือขายใหม ")
