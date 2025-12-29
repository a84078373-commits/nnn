#!/bin/bash

echo "===================================="
echo "نظام تعلم اللغة الإنجليزية بلغة الإشارة"
echo "===================================="
echo ""

echo "جاري التحقق من تثبيت المكتبات..."
if ! pip list | grep -q flask; then
    echo "جاري تثبيت المكتبات المطلوبة..."
    pip install -r requirements.txt
fi

echo ""
echo "جاري تشغيل التطبيق..."
echo "افتح المتصفح على: http://localhost:5000"
echo ""

python app.py

