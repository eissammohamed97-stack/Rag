# RAG Assistant — مشروع RAG بسيط ومحترف

مشروع Retrieval-Augmented Generation (RAG) مبني بملفات Python منفصلة، حسب
تسلسل المعمل المطلوب:

```
documents -> preprocessing -> chunking -> vector representation
-> vector store -> context retrieval -> prompting -> Streamlit UI
```

## 1. هيكل الملفات (Required File Structure)

| الملف | الوظيفة |
|---|---|
| `01_documents.py` | تحميل الملفات الخام (txt / pdf / csv) من `data/documents/` |
| `02_preprocessing.py` | تنظيف النصوص (مسافات زائدة، أسطر فارغة...) |
| `03_chunking.py` | تقسيم كل مستند إلى قطع (chunks) متداخلة بحجم ثابت |
| `04_vector_representation.py` | تحويل كل chunk إلى متجه (embedding) باستخدام `sentence-transformers` |
| `05_create_chroma_store.py` | بناء وحفظ قاعدة بيانات المتجهات (Chroma) على القرص |
| `06_retrieve_context.py` | البحث عن أقرب chunks للسؤال وتجميعها في context واحد مع ترقيم المصادر |
| `07_prompting.py` | بناء الـ prompt الصارم (grounded) واستدعاء نموذج LLM عبر OpenRouter |
| `streamlit_app.py` | واجهة المستخدم النهائية (Streamlit) |
| `requirements.txt` | المكتبات المطلوبة |

كل ملف قابل للتشغيل بمفرده لتجربته:
```bash
python 01_documents.py
python 02_preprocessing.py
python 03_chunking.py
python 04_vector_representation.py
python 05_create_chroma_store.py
python 06_retrieve_context.py
python 07_prompting.py
```

## 2. التشغيل محليًا (Local Setup)

```bash
# 1) تثبيت المكتبات
pip install -r requirements.txt

# 2) إنشاء ملف المفاتيح المحلي (لا يُرفع إلى GitHub أبدًا)
cp .env.example .env
# ثم افتح .env وضع مفتاحك الحقيقي:
#   OPENROUTER_API_KEY=sk-...
#   OPENROUTER_MODEL=openai/gpt-4o-mini

# 3) بناء قاعدة بيانات المتجهات لأول مرة
python 05_create_chroma_store.py

# 4) تشغيل التطبيق
streamlit run streamlit_app.py
```

المشروع يأتي مع 3 ملفات نصية جاهزة داخل `data/documents/` (سياسات شركة
وهمية، FAQ، ودليل onboarding) حتى تستطيع تجربة كل شيء فورًا. لاستخدام
بياناتك الخاصة، فقط ضع ملفات `.txt` أو `.pdf` أو `.csv` (بعمود `text`) في
نفس المجلد وأعد بناء الفهرس (زر "Rebuild vector store" داخل التطبيق، أو
`python 05_create_chroma_store.py`).

## 3. قواعد مفاتيح الـ API (API Key Rules)

- **لا** تكتب مفتاحك الحقيقي داخل أي ملف Python.
- **لا** ترفع ملف `.env` الحقيقي إلى GitHub (موجود بالفعل في `.gitignore`).
- عند النشر على Streamlit Cloud، استخدم Secrets بصيغة TOML بدلاً من `.env`.

## 4. النشر على Streamlit Cloud (Deployment)

1. ارفع المشروع (بدون `.env` وبدون `data/chroma_store/`) إلى مستودع GitHub عام.
2. أنشئ تطبيقًا جديدًا على [share.streamlit.io](https://share.streamlit.io) واختر `streamlit_app.py`.
3. من داخل التطبيق: **Manage app → Secrets**، وأضف:

```toml
OPENROUTER_API_KEY = "your_openrouter_key_here"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

4. أعد تشغيل التطبيق ثم اضغط زر **Rebuild vector store** من الشريط الجانبي
   لبناء الفهرس على السحابة (لأن `data/chroma_store/` لم يُرفع مع الكود).

## 5. الفحص النهائي قبل التسليم (Final Checklist)

- [ ] كل الملفات المطلوبة موجودة (01 → 07 + `streamlit_app.py` + `requirements.txt`).
- [ ] `requirements.txt` موجود ومحدّث.
- [ ] مفتاح الـ API الحقيقي **غير** موجود في ملف ZIP ولا في مستودع GitHub.
- [ ] Streamlit secrets مُهيّأة بصيغة TOML صحيحة.
- [ ] التطبيق يعمل بنجاح على Streamlit Cloud.
- [ ] الإجابة تعتمد فعليًا على الـ context المسترجَع.
- [ ] الإجابة تذكر المصادر (Sources) التي استُخدمت.

## 6. كيف يعمل النظام باختصار (How It Works)

1. **Documents**: نقرأ الملفات الخام من `data/documents/`.
2. **Preprocessing**: ننظّف النصوص (مسافات، أسطر فارغة، إلخ).
3. **Chunking**: نقسّم كل مستند إلى قطع من ~120 كلمة مع تداخل 20 كلمة، حتى لا
   تُقطع فكرة في منتصفها.
4. **Vector Representation**: نحوّل كل قطعة إلى متجه رقمي (embedding) بنموذج
   `all-MiniLM-L6-v2` — نموذج صغير وسريع ويعمل محليًا بدون تكلفة.
5. **Vector Store**: نخزّن كل المتجهات في قاعدة Chroma على القرص (`data/chroma_store/`)
   حتى لا نعيد بناءها في كل مرة.
6. **Context Retrieval**: عند وصول سؤال، نحوّله إلى متجه ونبحث عن أقرب K قطع
   له، ثم نجمعها في نص واحد مرقّم المصادر (`[Source 1]`, `[Source 2]`, ...).
7. **Prompting**: نبني تعليمات صارمة للنموذج: "أجب فقط من هذا الـ context،
   وإن لم تجد الإجابة فقل ذلك بوضوح، واذكر أرقام المصادر."
8. **Streamlit UI**: واجهة بسيطة لطرح الأسئلة، عرض الإجابة، وعرض المصادر
   التي استُخدمت (قابلة للطي/التوسيع).
