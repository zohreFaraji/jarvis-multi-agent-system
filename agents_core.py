#agents_core.py


from typing import Annotated, List, TypedDict
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# راه‌اندازی مدل لوکال gemma4:31b روی Ollama
llm = ChatOllama(
    model="gemma4:31b",
    temperature=0.1,  # دمای پایین‌تر برای تمرکز بیشتر روی کدنویسی دقیق
    base_url="http://localhost:11434"
)

class AgentState(TypedDict):
    messages: List[Annotated[str, "The conversation messages"]]
    next_agent: str
    task_result: str
    project_context: str

def supervisor_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1].lower()
    
    if any(keyword in last_message for keyword in ["کد", "پروژه", "فریم‌ورک", "برنامه‌نویسی", "ساخت", "توسعه", "دیباگ", "اپلیکیشن", "سایت"]):
        return {"next_agent": "CodeArchitectAgent"}
    elif any(keyword in last_message for keyword in ["کتاب", "رمان", "منبع", "مطالعه", "نویسنده", "خلاصه کتاب"]):
        return {"next_agent": "BookAdvisorAgent"}
    elif any(keyword in last_message for keyword in ["قیمت", "فلز", "ارز", "فولاد", "آهن", "دلار", "نرخ"]):
        return {"next_agent": "DataAgent"}
    elif any(keyword in last_message for keyword in ["پرواز", "سفر", "هواپیما", "بلیط", "رزرو"]):
        return {"next_agent": "TravelAgent"}
    else:
        return {"next_agent": "GeneralAgent"}

def get_agent_prompt(agent_name: str, user_prompt: str) -> str:
    """تولید پرامپت اختصاصی بر اساس نوع ایجنت برای هدایت دقیق مدل"""
    if agent_name == "CodeArchitectAgent":
        return (
            f"تو یک معمار ارشد نرم‌افزار هستی. برای درخواست زیر، باید حتماً کدهای کامل و کاربردی پروژه را بنویسید.\n"
            f"درخواست کاربر: {user_prompt}\n\n"
            "دستورالعمل‌های اجباری:\n"
            "1. به هیچ وجه به ساخت یک فایل خالی اکتفا نکن. کدهای واقعی، کامل و اجرایی پایتون را درون فایل‌ها قرار ده.\n"
            "2. نام مسیر نسبی هر فایل را دقیقاً در خط بالای بلوک کد بنویس.\n"
            "3. کدهای هر فایل را دقیقاً داخل بلوک مارک‌داون (مانند python) قرار بده.\n"
            "فرمت دقیق خروجی که باید رعایت کنی:\n\n"
            "main.py\n"
            "python\n"
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n\n"
            '@app.get("/")\n'
            "def read_root():\n"
            '    return {"message": "Hello World"}\n'
            "```\n\n"
            "حالا کدهای پروژه درخواستی کاربر را با همین فرمت کامل تولید کن:"
        )
    elif agent_name == "BookAdvisorAgent":
        return f"تو یک مشاور کتاب هستی. به این درخواست پاسخ بده:\n{user_prompt}"
    elif agent_name == "DataAgent":
        return f"تو تحلیل‌گر بازار هستی:\n{user_prompt}"
    elif agent_name == "TravelAgent":
        return f"تو متخصص سفر هستی:\n{user_prompt}"
    else:
        return user_prompt

# توابع همگام قبلی برای درخواست‌های معمولی حفظ شده‌اند
def code_architect_agent_node(state: AgentState):
    user_prompt = state["messages"][-1]
    prompt = get_agent_prompt("CodeArchitectAgent", user_prompt)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
    except Exception as e:
        content = f"خطا در ارتباط با مدل لوکال: {str(e)}"
    return {"task_result": content}

def book_advisor_agent_node(state: AgentState):
    user_prompt = state["messages"][-1]
    prompt = get_agent_prompt("BookAdvisorAgent", user_prompt)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
    except Exception as e:
        content = f"خطا: {str(e)}"
    return {"task_result": content}

def data_agent_node(state: AgentState):
    user_prompt = state["messages"][-1]
    prompt = get_agent_prompt("DataAgent", user_prompt)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
    except Exception as e:
        content = f"خطا: {str(e)}"
    return {"task_result": content}

def travel_agent_node(state: AgentState):
    user_prompt = state["messages"][-1]
    prompt = get_agent_prompt("TravelAgent", user_prompt)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
    except Exception as e:
        content = f"خطا: {str(e)}"
    return {"task_result": content}

def general_agent_node(state: AgentState):
    user_prompt = state["messages"][-1]
    try:
        response = llm.invoke([HumanMessage(content=user_prompt)])
        content = response.content
    except Exception as e:
        content = f"خطا: {str(e)}"
    return {"task_result": content}
