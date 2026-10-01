import os
import ollama

class JarvisAgentRouter:
    def _init_(self, model_name="gemma4:31b"):
        self.model_name = model_name
        print(f"[AgentRouter] Initialized with model: {self.model_name}")

    def route_query(self, user_query: str) -> str:
        """
        تشخیص می‌دهد که سوال کاربر به کدام ایجنت نیاز دارد:
        1. RAG (جستجو در اسناد و پایگاه دانش)
        2. CODE (تولید یا تحلیل کد)
        3. GENERAL (پاسخ عمومی و گفتگوی آزاد)
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
            print(f"[Router Error] {e}")
            return "GENERAL"

    def delegate(self, user_query: str, rag_callback, code_callback, general_callback):
        """
        درخواست را مسیریابی کرده و به ایجنت مربوطه تحویل می‌دهد
        """
        agent_type = self.route_query(user_query)
        print(f"[Router] Target Agent Selected: --> {agent_type}")

        if agent_type == "RAG":
            return rag_callback(user_query)
        elif agent_type == "CODE":
            return code_callback(user_query)
        else:
            return general_callback(user_query)
