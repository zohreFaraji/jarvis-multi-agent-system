#graph_runner.py


import os
import re
from agents_core import (
    AgentState,
    supervisor_node,
    code_architect_agent_node,
    book_advisor_agent_node,
    data_agent_node,
    travel_agent_node,
    general_agent_node
)

def extract_and_create_files(response_text: str, base_dir: str = "generated_project"):
    """
    استخراج دقیق فایل‌ها از پاسخ مدل حتی با تغییرات جزئی در فرمت
    """
    if not isinstance(response_text, str):
        return

    # الگوی جامع‌تر برای پیدا کردن نام فایل و بلوک کد زیر آن
    pattern = r"([a-zA-Z0-9_\-/\.]+\.(?:py|txt|json|md|html|css|js))\s*\n+[a-zA-Z]*\n(.*?)\n"
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    if matches:
        os.makedirs(base_dir, exist_ok=True)
        print(f"\033[96m📁 در حال ساخت فایل‌های پروژه در '{base_dir}'...\033[0m")
        for file_path, file_content in matches:
            file_path = file_path.strip()
            if file_path.startswith("/") or "etc" in file_path or "var" in file_path:
                file_path = os.path.basename(file_path)

            full_path = os.path.join(base_dir, file_path)
            
            try:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(file_content.strip())
                print(f"   ✅ فایل با موفقیت ایجاد شد: {full_path}")
            except Exception as e:
                print(f"   ⚠️ خطا در ایجاد فایل {file_path}: {e}")
    else:
        print("⚠️ هشدار: هیچ بلوک کدی با فرمت استاندارد در پاسخ مدل پیدا نشد تا در فایل ذخیره شود!")

def run_multi_agent_system():
    print("=" * 60)
    print("🚀 سیستم پایداری‌‌سنج و معمار پروژه آماده است (نسخه اصلاح‌شده).")
    print("=" * 60)

    user_history = []

    while True:
        try:
            user_input = input("\n👤 کاربر: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["خروج", "exit", "quit"]:
                print("\n👋 خدانگهدار!")
                break

            user_history.append(user_input)
            if len(user_history) > 4:
                user_history = user_history[-4:]

            state: AgentState = {
                "messages": user_history,
                "next_agent": "",
                "task_result": "",
                "project_context": ""
            }

            routing_decision = supervisor_node(state)
            selected_agent = routing_decision["next_agent"]
            print(f"⚙️ [مدیر سیستم]: هدایت درخواست به -> {selected_agent}")

            if selected_agent == "CodeArchitectAgent":
                result_dict = code_architect_agent_node(state)
                task_content = result_dict.get("task_result", "")
                
                print(f"\n🤖 [معمار پروژه]: در حال تولید و نوشتن کدهای واقعی...")
                try:
                    extract_and_create_files(task_content, "generated_project")
                except Exception as file_err:
                    print(f"⚠️ خطا در ساخت فایل‌ها: {file_err}")
            else:
                # برای سایر ایجنت‌ها
                if selected_agent == "BookAdvisorAdvisor":
                    res = book_advisor_agent_node(state)
                elif selected_agent == "DataAgent":
                    res = data_agent_node(state)
                elif selected_agent == "TravelAgent":
                    res = travel_agent_node(state)
                else:
                    res = general_agent_node(state)
                print(f"\n🤖 پاسخ:\n" + "-" * 40)
                print(res.get("task_result", ""))
                print("-" * 40)

            print("\n✨ درخواست پردازش شد.")

        except KeyboardInterrupt:
            print("\n\n👋 عملیات متوقف شد.")
            break
        except Exception as e:
            print(f"\n❌ خطا: {str(e)}")

if __name__ == "__main__":
    run_multi_agent_system()
