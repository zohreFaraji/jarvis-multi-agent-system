import logging
import sys
import ollama

# تنظیمات پیشرفته لاگ‌گرفتن برای نظارت دقیق بر وضعیت ایجنت
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SmartJarvisCore")

class SmartJarvisCore:
    def __init__(self, model_name: str = "gemma4:31b"):
        self.model_name = model_name
        logger.info(f"در حال راه‌اندازی هسته هوشمند با مدل: {self.model_name}")
        self._verify_environment()

    def _verify_environment(self) -> bool:
        """بررسی صحت اتصال به سرور اولاما و حضور مدل مورد نظر"""
        try:
            response_list = ollama.list()
            available_models = [m.get('model', '') for m in response_list.get('models', [])]
            
            if self.model_name in available_models:
                logger.info(f"مدل {self.model_name} با موفقیت روی سرور تأیید شد.")
                return True
            else:
                logger.error(f"خطا: مدل {self.model_name} در لیست مدل‌های اولاما یافت نشد!")
                raise ValueError(f"Model {self.model_name} not available in Ollama.")
        except Exception as e:
            logger.critical(f"خطای بحرانی در ارتباط با سرویس اولاما: {e}")
            raise

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """ارسال درخواست به مدل با مدیریت کامل خطا و ساختار پیام‌ها"""
        if not prompt or not prompt.strip():
            logger.warning("درخواست خالی دریافت شد.")
            return "لطفاً یک متن معتبر ارسال کنید."

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        
        messages.append({'role': 'user', 'content': prompt})

        try:
            logger.info("ارسال درخواست پردازش به مدل...")
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )
            answer = response.get('message', {}).get('content', '')
            logger.info("پاسخ با موفقیت از مدل دریافت شد.")
            return answer

        except Exception as e:
            logger.error(f"خطا در هنگام تولید پاسخ توسط مدل: {e}")
            return f"متأسفانه در پردازش درخواست شما خطایی رخ داد: {str(e)}"

    def route_query(self, user_query: str) -> str:
        """
        لایه مسیریابی هوشمند (Router Agent):
        تشخیص می‌دهد سوال کاربر مربوط به RAG (اسناد)، CODE (کدنویسی) یا GENERAL (عمومی) است.
        """
        prompt = f"""
        Analyze the following user query and classify it into one of these three categories: RAG, CODE, GENERAL.
        User Query: "{user_query}"
        
        Return ONLY the category name (RAG, CODE, or GENERAL). No extra text.
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            category = response['message']['content'].strip().upper()
            if "RAG" in category:
                return "RAG"
            elif "CODE" in category:
                return "CODE"
            else:
                return "GENERAL"
        except Exception as e:
            logger.error(f"[Router Error] {e}")
            return "GENERAL"

    def delegate(self, user_query: str, rag_callback, code_callback, general_callback):
        """
        مسیریابی و ارجاع درخواست به ایجنت متناظر
        """
        agent_type = self.route_query(user_query)
        logger.info(f"ایجنتِ هدف انتخاب شد --> {agent_type}")

        if agent_type == "RAG":
            return rag_callback(user_query)
        elif agent_type == "CODE":
            return code_callback(user_query)
        else:
            return general_callback(user_query)

# --- بخش تست و اجرای مستقیم ---
if __name__ == "__main__":
    print("=== تست جامع هسته ایجنت هوشمند با قابلیت مسیریابی چندایجنتی ===")
    try:
        jarvis = SmartJarvisCore(model_name="gemma4:31b")
        
        # تعریف توابع تستی برای هر ایجنت
        def handle_rag(query):
            return f"[Agent RAG]: در حال جستجو در پایگاه دانش و ChromaDB برای پرسش: '{query}'"

        def handle_code(query):
            return f"[Agent CODE]: در حال تحلیل منطق برنامه‌نویسی و کد برای پرسش: '{query}'"

        def handle_general(query):
            system_role = "تو یک دستیار هوش مصنوعی فوق‌العاده متخصص و دقیق هستی."
            return jarvis.generate_response(prompt=query, system_prompt=system_role)

        # تست نمونه
        test_query = "می‌خواهم کدهای مربوط به اتصال دیتابیس را بهینه‌سازی کنم"
        print(f"\nسوال تست: {test_query}\n")
        
        # اجرای سیستم دلیگیت و مسیریابی
        result = jarvis.delegate(test_query, handle_rag, handle_code, handle_general)
        
        print("-" * 40)
        print("خروجی نهایی سیستم:\n", result)
        print("-" * 40)
        
    except Exception as e:
        print(f"اجرای تست با خطا مواجه شد: {e}")
