import html
import os 
from typing import Annotated, List, Dict, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prompt.prompt import RenderDetectIntentSystemPrompt, RenderExtractEntitySystemPrompt, RenderConversationContextPrompt
from src.utils import DetectIntentResult, IntentInfo, LanguageInfo, SentimentInfo


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
    intent_model: IntentLm 
    entity_model: EntityLm

    intent: str
    language: str
    sentiment: str
    action: str


# ============================================================================

llm = init_chat_model(f"{provider}:{model}", temperature=0.0)

# Node 1: prepare NLU prompts from history
INTENTS = "greet:0.6, search_product:0.8, farewell:0.6"
ACTION = "knowledge_search"  # or "knowledge_search"
ENTITIES = [
  {"name": "order_id", "type": "text", "description": "order reference number"},
  {"name": "payment_amount", "type": "currency", "description": "amount paid"},
  {"name": "payment_status", "type": "text", "description": "status of payment"}
]

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
            }
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
    # Convert LangGraph messages -> our minimal schema
    # Expect entries like {"role": "user"|"assistant", "content": "..."}
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
    systemprompt = RenderExtractEntitySystemPrompt(INTENTS)

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
def actionDeciderBranch(state: State) -> dict:
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
        updates["action"] = ACTION
    if primary_language:
        updates["language"] = primary_language.code
    if dominant_sentiment:
        updates["sentiment"] = dominant_sentiment.sentiment

    if len(ENTITIES) > 0:
        # select entity extraction path

    else:
        # select direct response path
    return updates

# Build graph
builder = StateGraph(State)
builder.add_node("intentInputNode", intentInputNode)
# builder.add_node("entityInputNode", entityInputNode)
builder.add_node("intentChatmodelNode", intentChatmodelNode)
# builder.add_node("entityChatmodelNode", entityChatmodelNode)

builder.add_edge(START, "intentInputNode")
builder.add_edge("intentInputNode", "intentChatmodelNode")
builder.add_edge("intentChatmodelNode", END)

graph = builder.compile()

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
    }

    final = graph.invoke(state)
    print("=== SYSTEM PROMPT ===")
    print(final["intent_model"]["systemprompt"])
    print("=== USER PROMPT ===")
    print(final["intent_model"]["userprompt"])
    print("=== NLU OUTPUT ===")
    print(final["intent_model"]["output"])
    print("=== PARSED RESULT ===")
    print(intentParser(final["intent_model"]["output"]))

if __name__ == "__main__":
    run_once("I want to find running shoes.")
