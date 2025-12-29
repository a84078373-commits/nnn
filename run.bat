@echo off
echo ====================================
echo  نظام تعلم اللغة الإنجليزية بلغة الإشارة
echo ====================================
echo.

echo جاري التحقق من تثبيت المكتبات...
pip list | findstr flask >nul
if errorlevel 1 (
    echo جاري تثبيت المكتبات المطلوبة...
    pip install -r requirements.txt
)

echo.
echo جاري تشغيل التطبيق...
echo افتح المتصفح على: http://localhost:5000
echo.

python app.py

pause

