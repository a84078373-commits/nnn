@echo off
echo ====================================
echo  اختبار Flask
echo ====================================
echo.

echo جاري التحقق من Flask على localhost:5000...
echo.

curl -s http://localhost:5000 >nul 2>&1
if errorlevel 1 (
    echo ❌ Flask لا يعمل على localhost:5000
    echo.
    echo يرجى تشغيل Flask أولاً:
    echo   python app.py
    echo.
) else (
    echo ✅ Flask يعمل بشكل صحيح!
    echo.
    echo يمكنك فتح http://localhost:5000 في المتصفح
    echo.
)

pause

