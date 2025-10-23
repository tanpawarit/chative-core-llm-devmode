สรุปให้เป็นตัวอย่าง Action แบบ “tools” 3 กลุ่ม พร้อมแนวทางผูกกับ flow ที่มีอยู่ และสเกลสำหรับหลาย action

แนวคิดหลัก

สร้าง Action เป็น “tools” ให้ responseChatmodelNode เรียกเมื่อจำเป็น
จำกัด tools ต่อรอบตาม action ที่เลือก (จาก Action Router หลัง NLU)
Params ของ tools มาจาก entities mapping + context ปัจจุบัน (current message, history)
ฝั่งเซิร์ฟเวอร์ validate/authorize ก่อน execute เสมอ
KnowledgeBase (Retriever)

Use case: ดึง knowledge จาก Milvus ผ่าน KnowledgeRepo.Search
Tool spec
name: retrieve_knowledge
description: “Hybrid search in workspace knowledge base”
input schema:
query string (required)
top_k int (optional, default จาก config)
filters object (optional: doc_name, doc_hash, etc.)
use_entities boolean (optional; true = เติม entity values ลง query)
Output: list of snippets [{id, text, doc_name, score, ...}]
Mapping/Behavior
ถ้าตั้ง use_entities=true ให้ compose query = current message + high-signal entities (เช่น order_number/product_name)
Enforce workspace scope ฝั่งเซิร์ฟเวอร์; ปรับ top_k ≤ config.TopK
Cache ผลลัพธ์ระยะสั้น (per conversation+query+filters)
ตัวอย่าง LLM call (แนวคิด)
“need product pricing detail” → call retrieve_knowledge{query: 'premium plan pricing', top_k: 3}
ข้อควรใส่ใน system prompt
“เรียก retrieve_knowledge เมื่อข้อมูลไม่พอ หรือเป็นคำถามความรู้/นโยบาย/คู่มือ”
OpenAPI v3 (Business API)

Use case: เรียก external/internal API ตาม OpenAPI spec
แบบที่แนะนำ: “หนึ่ง operation = หนึ่ง tool” เพื่อให้ LLM เข้าใจหน้าที่เฉพาะเจาะจงและลด error
Tool generation
โหลด OpenAPI spec → gen tool ต่อ operation
name: openapi_{operationId} เช่น openapi_getOrderStatus
description: มาจาก summary/description
input schema: มาจาก path/query/header/body schema (แปลงเป็น JSON Schema)
security: inject credentials/keys จาก server-side; allowlist เฉพาะ operations ที่อนุญาต
Execution server-side
Resolve server URL, substitute path params, attach query/body
Validate input ตาม schema อีกครั้ง
Rate limit / timeout / redact logs
ตัวอย่าง
Spec: GET /orders/{order_number} operationId: getOrderStatus
Tool:
name: openapi_getOrderStatus
input: { "order_number": "A1234" }
Mapping entities → params: order_number ← entities["order_number"]
Prompt policy
“เมื่อ intent = check_order ให้ใช้ openapi_getOrderStatus โดยใส่ order_number จากบริบท; ถ้าขาดให้ถาม 1 คำถามก่อน”
Another Plugins (เช่น CRM/Ticketing/Sheets)

แนวทางสถาปัตยกรรม
Plugin registry: ลงทะเบียน plugin ต่อ workspace พร้อม capability descriptors
แต่ละ plugin expose 1..N tools ด้วย ToolInfo + JSON Schema
ตัวอย่าง capability
CRM: crm_find_customer(email|phone), crm_update_ticket(id, status, note)
Ticketing: create_ticket(subject, body, priority), get_ticket(id)
Sheets: sheet_append_row(sheet_id, values[])
Input schema
สร้างจาก entity mapping + เพิ่ม optional fields ให้ LLM ใช้ถาม/เติม (แต่ถามต่อครั้งเดียว)
Security/Compliance
Scope ต่อ workspace; secret management; audit log call/args/result size
Idempotency key สำหรับ action ที่มี side-effect
ผูกเข้ากับ Flow ปัจจุบัน

หลัง actionDeciderBranch
ถ้า missing required entities → ไป responseEntityFallbackNode (ถาม 1 ช่อง)
ถ้าครบ → ไป responseNode และเปิด toolset เฉพาะของ action + retrieve_knowledge
ภายใน responseNode/responseChatmodelNode
Bind tools เฉพาะ (จาก action ที่เลือก + core tools)
Include .Entities ใน system prompt และให้หลักเกณฑ์ “เลือกเรียก tool เมื่อจำเป็น”
Slot filling
แผนที่ entity code → tool param name
Validate และ normalize type (email/phone/date/number) ฝั่ง server ก่อน execute
ตัวอย่างสั้นๆ (โค้ดสเก็ตช์)

Knowledge tool (ใช้ Eino tool-calling)
Register
สร้าง ToolInfo ชื่อ retrieve_knowledge และ JSON Schema ตามข้างต้น
chatModel.BindTools([]*schema.ToolInfo{...})
toolsNode := compose.NewToolsNode(ctx, &compose.ToolsNodeConfig{Tools: []tool.BaseTool{knowledgeTool}})
Handler
ดึง query/top_k/filters/use_entities → build query → knowledgeRepo.Search(ctx, workspaceID, query) → คืนผลเป็น schema.Message แบบ ToolMessage
OpenAPI tool
ตอนโหลด spec: generate tools ต่อ operationId ที่ allowlist
Handler: build request → call → parse response → คืนผล summary JSON (อย่าส่ง raw ใหญ่เกิน)
Registry
ต่อรอบ: resolve action → เลือกชุด tools (actionTools + coreTools) → bind เข้ากับ chat model และ tools node
การสเกลเมื่อหลาย action

Action Router: จาก intent → เลือก action หรือถาม disambiguation สั้นๆ
จำกัด toolset ต่อรอบ: ลด hallucination/side-effect
Observability: log tool name, duration, result size, partial args (redact PII)
Reliability: retry policy แบบ safe (เฉพาะ GET), idempotency สำหรับ POST/PATCH

Proposed Flow (Tools-Based, multi‑action)

START
→ Build Context (agent + histories + current message)
→ NLU (detect intent, language, sentiment)
→ Action Router (map intent → action candidates)
→ If multiple actions → Disambiguation (ask 1 brief question) → Route
→ Slot Filling (map entities → action params)
→ If required missing → Fallback Ask (1 question) → END TURN (wait user)
→ Else continue
→ Response w/ Tool‑Calling (bind allowed tools = action‑specific + retrieve_knowledge)
→ Tool Loop (0..N)
→ Final Answer (tone/formality per config + knowledge/tool results)
→ Publish + Persist + Update Memory
→ END
Tool Loop (inside Response Model)

LLM decides need tool?
→ No → Draft final answer → Return
→ Yes → ToolCall(name, args)
→ Execute (server validates/authz, timeouts, redact)
→ ToolResult → Back to LLM
→ Repeat until sufficient or budget reached → Final Answer
Notes within “bind allowed tools”:

KnowledgeBase: retrieve_knowledge(query[, top_k, filters, use_entities])
OpenAPI v3: one tool per operationId (openapi_<opId>(params))
Other Plugins: plugin_scoped tools (e.g., crm_find_customer, create_ticket)
Limit tools to the routed action (+ core retriever) per turn