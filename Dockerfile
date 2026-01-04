# استخدم نسخة Python خفيفة
FROM python:3.11-slim

# تعيين مجلد العمل
WORKDIR /app

# نسخ متطلبات المشروع أولًا لتسريع التخزين المؤقت
COPY requirements.txt .

# تثبيت المتطلبات
# ترقية pip أولاً
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# تثبيت PyTorch من المصدر الرسمي (CPU only)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.1.0 torchvision==0.16.0

# تثبيت باقي المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# نسخ بقية الملفات
COPY . .

# إعداد البيئة
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# أمر التشغيل
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "2", "--timeout", "120"]
