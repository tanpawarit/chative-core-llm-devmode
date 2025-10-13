import os
from typing import Annotated, List, Dict, Tuple, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from prompt.detect_intent import INTENT_TEMPLATE
from prompt.extract_entity import ENTITY_TEMPLATE

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

# ---- LangGraph wiring --------------------------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set")

provider = os.getenv("LLM_PROVIDER", "openai")
model = os.getenv("LLM_MODEL", "gpt-4.1")

# ============================================================================
class IntentLm(TypedDict):
    systemprompt: str                                         # generated system prompt
    userprompt: str                                           # generated user prompt
    output: str                                           # model's tuple output

class EntityLm(TypedDict):
    systemprompt: str                                         # generated system prompt
    userprompt: str                                           # generated user prompt
    output: str                                           # model's tuple output
 
class State(TypedDict):
    messages: Annotated[list, add_messages]      
    intent_model: IntentLm 
    entity_model: EntityLm

# ============================================================================

llm = init_chat_model(f"{provider}:{model}", temperature=0.0)

# Node 1: prepare NLU prompts from history
INTENTS = "greet:0.6, search_product:0.8, farewell:0.6"
ENTITIES = [
  {"name": "order_id", "type": "text", "description": "order reference number"},
  {"name": "payment_amount", "type": "currency", "description": "amount paid"},
  {"name": "payment_status", "type": "text", "description": "status of payment"}
]

def intentInputNode(state: State) -> dict:
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
    return {
        "intent_model":{
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
    return {
        "entity_model":{
            "output": result.content
            }
        }

def intentParser():
    pass

# Build graph
builder = StateGraph(State)
builder.add_node("intentInputNode", intentInputNode)
builder.add_node("entityInputNode", entityInputNode)
builder.add_node("intentChatmodelNode", intentChatmodelNode)
builder.add_node("entityChatmodelNode", entityChatmodelNode)

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
        "systemprompt": "",
        "userprompt": "",
        "nlu_output": "",
    }

    final = graph.invoke(state)
    print("=== SYSTEM PROMPT ===")
    print(final["systemprompt"])
    print("=== USER PROMPT ===")
    print(final["userprompt"])
    print("=== NLU OUTPUT ===")
    print(final["nlu_output"])

if __name__ == "__main__":
    run_once("I want to find running shoes.")
