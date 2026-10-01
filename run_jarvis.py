import chromadb
from sentence_transformers import SentenceTransformer

print(">>> [تست زنده] فایل جدید اجرا شد")

class MyJarvis:
    def __init__(self):
        print(">>> [تست زنده] تابع سازنده init با موفقیت کار کرد!")
        self.model = SentenceTransformer('intfloat/multilingual-e5-base')

    def save(self, text):
        print(">>> [تست زنده] وارد تابع ذخیره شدیم...")
        vec = self.model.encode(text).tolist()
        print(">>> [تست زنده] بردار با موفقیت ساخته شد، طول بردار:", len(vec))

if __name__ == "__main__":
    print(">>> [تست زنده] در حال نمونه‌سازی...")
    j = MyJarvis()
    j.save("تست نهایی")
