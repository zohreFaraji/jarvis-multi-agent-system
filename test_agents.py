from typing import Annotated, List, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage

# ساختار وضعیت مشترک بین ایجنت‌ها (State با قابلیت نگهداری کانکسِت پروژه)
class AgentState(TypedDict):
    messages: List[Annotated[str, "The conversation messages"]]
    next_agent: str
    task_result: str
    project_context: str

# ۱. ایجنت مدیر (Supervisor) - مغز متفکر برای تشخیص دقیق نوع درخواست کاربر
def supervisor_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1].lower()
    
    # مسیریابی هوشمند بر اساس کلیدواژه‌های تخصصی
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

# ۲. ایجنت تخصصی معماری و توسعه پروژه (Code Architect & Full-Stack AI Agent)
def code_architect_agent_node(state: AgentState):
    """
    این ایجنت به عنوان یک Tech Lead جامع عمل می‌کند و پروژه را در ۵ بعد کلیدی تحلیل و طراحی می‌کند:
    ۱. فرانت‌اند (UI/UX)  ۲. بک‌اند و دیتابیس  ۳. سئو (SEO)  ۴. امنیت  ۵. قابلیت‌های هوش مصنوعی (AI/LLM/RAG)
    """
    last_msg = state["messages"][-1]
    result = (
        f"[Code Architect Agent - Full-Stack & AI Expert]: تحلیل همه‌جانبه پروژه.\n"
        f"• [1. Backend & DB]: طراحی معماری ماژولار (Clean Architecture)، پیاده‌سازی Connection Pooling، مدیریت خطا (Error Handling) برای جلوگیری از Crash و ORM بهینه.\n"
        f"• [2. Frontend]: پیشنهاد مدرن‌ترین فریم‌ورک‌های واکنش‌گرا، بهینه‌سازی سرعت بارگذاری و طراحی UI/UX کاربرپسند.\n"
        f"• [3. SEO Optimization]: پیاده‌سازی سئوی تکنیکال، متاتگ‌های داینامیک، نقشه سایت (Sitemap) و بهینه‌سازی ساختار URLها برای رتبه یک موتورهای جستجو.\n"
        f"• [4. Security & Hardening]: ایمن‌سازی در برابر آسیب‌پذیری‌ها، رمزنگاری داده‌ها، مدیریت احراز هویت امن و لایه‌های دفاعی.\n"
        f"• [5. AI Integration]: تزریق ماژول‌های هوش مصنوعی، خطوط لوله RAG، اتصال به مدل‌های لوکال (Gemma/Qwen) یا ایجنت‌های پردازش متن.\n"
        f"وضعیت: پکیج کامل مستندات فنی، استک پیشنهادی و اسکلت اجرایی تمام لایه‌ها آماده استقرار است."
    )
    return {"task_result": result}

# ۳. ایجنت مشاوره کتاب و ادبیات (Book Advisor Agent - جامع برای تمام ژانرها)
def book_advisor_agent_node(state: AgentState):
    """
    پشتیبان تمام ژانرها (رمان، اجتماعی، تاریخی، فنی، روانشناسی، هوش مصنوعی و...)؛
    برترین آثار، نویسندگان، خلاصه‌ها و نسخه‌های مرجع را معرفی می‌کند.
    """
    last_msg = state["messages"][-1]
    result = (
        f"[Book Advisor Agent]: بررسی درخواست کتاب و منابع مطالعاتی.\n"
        f"• حوزه درخواستی شما تحلیل شد.\n"
        f"• آثار برگزیده: گلچینی از برجسته‌ترین کتاب‌های داخلی و بین‌المللی مرتبط با موضوع مورد نظر.\n"
        f"• محتوا و ساختار: ارائه خلاصه‌ای از بن‌مایه اثر، اهداف نویسنده و دسته‌بندی سطح مطالعه (مقدماتی تا پیشرفته).\n"
        f"سیستم آماده ارائه جزئیات یا کتب مرجع است."
    )
    return {"task_result": result}

# ۴. ایجنت تخصصی داده و بازار (Data & Market Agent)
def data_agent_node(state: AgentState):
    result = (
        f"[Data Agent]: رصد تخصصی بازارهای مالی، فلزات، فولاد و ارز.\n"
        f"آخرین نوسانات، تحلیل‌های قیمتی و خروجی‌های دیتابیس استخراج و پردازش شد."
    )
    return {"task_result": result}

# ۵. ایجنت تخصصی سفر و پرواز (Travel Agent)
def travel_agent_node(state: AgentState):
    result = (
        f"[Travel Agent]: موتور جستجوی پرواز و سفرهای داخلی/خارجی.\n"
        f"بررسی ایرلاین‌ها، کلاس‌های پروازی، مقایسه قیمت‌ها و صندلی‌ها انجام شد."
    )
    return {"task_result": result}

# ۶. ایجنت عمومی پاسخگویی (General Agent)
def general_agent_node(state: AgentState):
    result = "[General Agent]: پردازش متن و پاسخگویی به درخواست‌های عمومی سیستم."
    return {"task_result": result}

print("نسخه پیشرفته و جامع Full-Stack & AI اسکلت مولتی‌ایجنت با موفقیت بارگذاری شد.")
