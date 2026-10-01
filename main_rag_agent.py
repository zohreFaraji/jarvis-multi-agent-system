#main_rag_agent.py

from jarvis_v3 import SmartJarvis
from smart_jarvis_core import SmartJarvisCore

class UnifiedJarvisAgent:
    def __init__(self, model_name: str = "gemma4:31b"):
        print("🔗 در حال راه‌اندازی ایجنت یکپارچه و چندایجنتی جارویس...")
        # راه‌اندازی حافظه وکتور (ChromaDB)
        self.memory_bot = SmartJarvis()
        # راه‌اندازی هسته مدل زبانی و روتر
        self.brain = SmartJarvisCore(model_name=model_name)

    def handle_rag(self, query: str) -> str:
        """ایجنت جستجو در اسناد (RAG)"""
        print(f"\n[Agent RAG]: جستجوی معنایی برای پرسش: '{query}'")
        retrieved_docs = self.memory_bot.search_memory(query, n_results=1)
        
        context = retrieved_docs[0] if retrieved_docs else "سند مرتبطی یافت نشد."
        
        system_prompt = (
            "تو یک دستیار متخصص هستی. بر اساس اطلاعات زمینه‌ای (Context) داده‌شده "
            "و دانش خودت، به پرسش کاربر به زبان فارسی و به صورت دقیق پاسخ بده."
        )
        user_prompt = f"اطلاعات زمینه‌ای:\n{context}\n\nپرسش کاربر:\n{query}"
        
        return self.brain.generate_response(prompt=user_prompt, system_prompt=system_prompt)

    def handle_code(self, query: str) -> str:
        """ایجنت برنامه‌نویسی و تحلیل کد"""
        print(f"\n[Agent CODE]: تحلیل منطق برنامه‌نویسی...")
        system_prompt = "تو یک مهندس ارشد نرم‌افزار و متخصص کدنویسی هستی."
        return self.brain.generate_response(prompt=query, system_prompt=system_prompt)

    def handle_general(self, query: str) -> str:
        """ایجنت پاسخ عمومی"""
        print(f"\n[Agent GENERAL]: پردازش گفتگوی آزاد...")
        system_prompt = "تو یک دستیار هوش مصنوعی دقیق و حرفه‌ای هستی."
        return self.brain.generate_response(prompt=query, system_prompt=system_prompt)

    def run(self, user_query: str) -> str:
        """مسیریابی هوشمند و اجرای ایجنت متناظر"""
        print(f"\n--- دریافت درخواست: '{user_query}' ---")
        response = self.brain.delegate(
            user_query=user_query,
            rag_callback=self.handle_rag,
            code_callback=self.handle_code,
            general_callback=self.handle_general
        )
        return response

if __name__ == "__main__":
    print("=== تست سیستم یکپارچه چندایجنتی جارویس ===")
    agent = UnifiedJarvisAgent(model_name="gemma4:31b")
    
    # تست ۱: سوالی که باید برود سراغ RAG (اسناد)
    query_1 = "چه دیتابیسی استفاده کردیم؟"
    ans_1 = agent.run(query_1)
    print("\nپاسخ نهایی:\n", ans_1)
    print("=" * 50)
    
    # تست ۲: سوالی که مربوط به کدنویسی است
    query_2 = "چطور یک تابع اتصال به دیتابیس در پایتون بنویسم؟"
    ans_2 = agent.run(query_2)
    print("\nپاسخ نهایی:\n", ans_2)
    print("=" * 50)
