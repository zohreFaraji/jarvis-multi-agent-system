#jarvis_v3.py

import os
import chromadb
from sentence_transformers import SentenceTransformer

print("🚀 شروع اجرای اسکریپت نسخه جستجو...")

class SmartJarvis:
    def __init__(self):
        print("🧠 سازنده کلاس فراخوانی شد.")
        print("⏳ در حال لود مدل امبدینگ...")
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')
        print("✅ مدل امبدینگ با موفقیت لود شد.")
        
        # اتصال به دیتابیس وکتور
        self.client = chromadb.PersistentClient(path="./jarvis_vector_db")
        self.collection = self.client.get_or_create_collection(name="persian_agent_memory")

    def add_memory(self, text_content):
        print(f"🔍 در حال ذخیره حافظه: '{text_content}'")
        doc_id = f"mem_{self.collection.count() + 1}"
        vector = self.embedding_model.encode(text_content).tolist()
        
        self.collection.add(
            embeddings=[vector],
            documents=[text_content],
            ids=[doc_id]
        )
        print(f"✅ حافظه با موفقیت ثبت شد! شناسه: {doc_id}")

    def search_memory(self, query, n_results=1):
        print(f"🔎 در حال جستجو برای پرسش: '{query}'")
        query_vector = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        
        print("🎯 نتایج جستجوی معنایی:")
        documents = results.get("documents", [[]])[0]
        for i, doc in enumerate(documents):
            print(f"  {i+1}. {doc}")
        return documents

if __name__ == "__main__":
    bot = SmartJarvis()
    
    # ثبت چند نمونه حافظه تستی
    bot.add_memory("من در حال توسعه یک ایجنت هوش مصنوعی فارسی روی لینوکس هستم.")
    bot.add_memory("برای پایگاه داده وکتور از ChromaDB و برای امبدینگ از مدل multilingual-e5-base استفاده می‌کنیم.")
    
    # تست جستجوی معنایی
    print("-" * 40)
    bot.search_memory("چه دیتابیسی استفاده کردیم؟")
    print("🎉 هورا! جستجو و بازیابی با موفقیت انجام شد.")
