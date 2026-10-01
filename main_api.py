#main_api.py


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
import os
import uuid
import json
import asyncio

from langchain_core.messages import HumanMessage
# ایمپورت توابع و متغیرهای لازم از agents_core
from agents_core import (
    AgentState,
    supervisor_node,
    code_architect_agent_node,
    book_advisor_agent_node,
    data_agent_node,
    travel_agent_node,
    general_agent_node,
    llm,
    get_agent_prompt
)
from graph_runner import extract_and_create_files

app = FastAPI(
    title="Jarvis Multi-Agent API",
    version="1.3.0",
    description="Scalable & Token-Streaming Backend API for Jarvis Cybernetic Multi-Agent System"
)

# تنظیمات CORS برای ارتباط آزادانه‌ی فرانت‌اند با بک‌اند
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ساختار ورودی درخواست از سمت فرانت‌اند (شامل session_id برای مدیریت سشن کاربران)
class UserRequest(BaseModel):
    message: str
    history: Optional[List[str]] = []
    session_id: Optional[str] = None

# ساختار پاسخ خروجی استاندارد
class AgentResponse(BaseModel):
    selected_agent: str
    response_content: str
    files_generated: bool = False
    session_id: str


@app.post("/api/chat", response_model=AgentResponse)
async def chat_with_jarvis(req: UserRequest):
    """
    اندپوینت همگام (Non-streaming) - پاسخ یکجا همراه با مدیریت سشن و فایل‌ها
    """
    try:
        current_session_id = req.session_id if req.session_id else str(uuid.uuid4())
        user_history = req.history[-4:] if req.history else []
        user_history.append(req.message)

        state: AgentState = {
            "messages": user_history,
            "next_agent": "",
            "task_result": "",
            "project_context": ""
        }

        # ۱. تصمیم‌گیری سوپروایزر
        routing_decision = supervisor_node(state)
        selected_agent = routing_decision["next_agent"]

        content = ""
        files_created = False
        user_project_dir = os.path.join("generated_projects", current_session_id)

        # ۲. اجرا توسط ایجنت منتخب
        if selected_agent == "CodeArchitectAgent":
            result_dict = code_architect_agent_node(state)
            content = result_dict.get("task_result", "")
            try:
                extract_and_create_files(content, user_project_dir)
                files_created = True
            except Exception as e:
                print(f"File generation error for session {current_session_id}: {e}")
                
        elif selected_agent == "BookAdvisorAgent":
            res = book_advisor_agent_node(state)
            content = res.get("task_result", "")
        elif selected_agent == "DataAgent":
            res = data_agent_node(state)
            content = res.get("task_result", "")
        elif selected_agent == "TravelAgent":
            res = travel_agent_node(state)
            content = res.get("task_result", "")
        else:
            res = general_agent_node(state)
            content = res.get("task_result", "")

        return AgentResponse(
            selected_agent=selected_agent,
            response_content=content,
            files_generated=files_created,
            session_id=current_session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def jarvis_event_generator(req: UserRequest) -> AsyncGenerator[str, None]:
    """
    ژنراتور پیشرفته استریمینگ (SSE): ابتدا مراحل تفکر و انتخاب ایجنت را ارسال می‌کند،
    سپس پاسخ مدل را به صورت زنده، کلمه به کلمه (Token-by-Token) برای کلاینت استریم می‌کند.
    """
    current_session_id = req.session_id if req.session_id else str(uuid.uuid4())
    user_history = req.history[-4:] if req.history else []
    user_history.append(req.message)

    state: AgentState = {
        "messages": user_history,
        "next_agent": "",
        "task_result": "",
        "project_context": ""
    }

    # ۱. اطلاع‌‌رسانی وضعیت تفکر سوپروایزر همراه با سشن‌آیدی
    yield f"data: {json.dumps({'type': 'status', 'stage': 'supervisor', 'message': '🧠 در حال تحلیل درخواست توسط Supervisor...', 'session_id': current_session_id})}\n\n"
    await asyncio.sleep(0.1)

    # اجرای سوپروایزر در ترد مجزا جهت عدم بلاک شدن Event Loop
    routing_decision = await asyncio.to_thread(supervisor_node, state)
    selected_agent = routing_decision.get("next_agent", "GeneralAgent")

    yield f"data: {json.dumps({'type': 'agent_selected', 'selected_agent': selected_agent, 'message': f'🤖 درخواست به ایجنت {selected_agent} ارجاع شد.', 'session_id': current_session_id})}\n\n"
    await asyncio.sleep(0.1)

    # ۲. آماده‌سازی پرامپت مناسب بر اساس ایجنت انتخاب‌شده
    prompt = get_agent_prompt(selected_agent, user_history[-1])

    yield f"data: {json.dumps({'type': 'status', 'stage': 'processing', 'message': f'⚙️ ایجنت {selected_agent} در حال تولید پاسخ زنده...', 'session_id': current_session_id})}\n\n"

    full_content = ""

    # ۳. استریم زنده کلمه به کلمه (Token Streaming) از مدل لوکال Ollama
    try:
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            chunk_text = chunk.content
            if chunk_text:
                full_content += chunk_text
                token_payload = {
                    "type": "token",
                    "content": chunk_text,
                    "session_id": current_session_id
                }
                yield f"data: {json.dumps(token_payload)}\n\n"
    except Exception as e:
        full_content = f"خطا در استریم مدل: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'message': full_content, 'session_id': current_session_id})}\n\n"

    # ۴. اگر ایجنت معمار کد بود، فایل‌ها را در پوشه اختصاصی سشن استخراج و ذخیره کن
    files_created = False
    user_project_dir = os.path.join("generated_projects", current_session_id)
    if selected_agent == "CodeArchitectAgent":
        yield f"data: {json.dumps({'type': 'status', 'stage': 'file_generation', 'message': '📁 در حال استخراج و ذخیره‌سازی فایل‌های پروژه...', 'session_id': current_session_id})}\n\n"
        try:
            await asyncio.to_thread(extract_and_create_files, full_content, user_project_dir)
            files_created = True
        except Exception as e:
            print(f"File generation error for session {current_session_id}: {e}")

    # ۵. ارسال پکت نهایی و پایان استریم
    final_payload = {
        "type": "final_result",
        "selected_agent": selected_agent,
        "response_content": full_content,
        "files_generated": files_created,
        "session_id": current_session_id
    }
    yield f"data: {json.dumps(final_payload)}\n\n"


@app.post("/api/chat/stream")
async def chat_with_jarvis_stream(req: UserRequest):
    """
    اندپوینت استریمینگ پیشرفته (SSE) - دریافت زنده مراحل تفکر، توکن‌های زنده مدل و سشن
    """
    return StreamingResponse(
        jarvis_event_generator(req),
        media_type="text/event-stream"
    )


@app.get("/")
def read_root():
    return {"status": "Jarvis Advanced Token-Streaming Multi-Agent Backend is running successfully!"}
