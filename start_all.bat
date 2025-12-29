@echo off
chcp 65001 >nul
echo ====================================
echo  تشغيل كامل للنظام
echo ====================================
echo.

REM Check if cloudflared exists
if not exist "cloudflared-windows-amd64.exe" (
    echo ❌ ملف cloudflared-windows-amd64.exe غير موجود!
    pause
    exit
)

echo جاري تشغيل Flask...
start "Flask Server" cmd /k "python app.py"

echo.
echo انتظر 10 ثواني حتى يبدأ Flask...
timeout /t 10 /nobreak >nul

echo.
echo جاري التحقق من Flask...
timeout /t 2 /nobreak >nul

REM Try to check if Flask is running
curl -s http://localhost:5000 >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠️  Flask قد لا يكون جاهزاً بعد
    echo.
    echo يرجى:
    echo 1. التحقق من نافذة Flask
    echo 2. التأكد من عدم وجود أخطاء
    echo 3. فتح http://localhost:5000 في المتصفح
    echo.
    echo هل تريد المتابعة على أي حال؟ (Y/N)
    choice /C YN /N /M ""
    if errorlevel 2 (
        echo تم الإلغاء
        pause
        exit
    )
)

echo.
echo ====================================
echo جاري إنشاء نفق Cloudflare...
echo ====================================
echo.
echo ⚠️  سيظهر رابط عام - انسخه وشاركه مع الآخرين
echo.
echo اضغط Ctrl+C لإيقاف النفق
echo.

REM Start cloudflared tunnel
cloudflared-windows-amd64.exe tunnel --url http://127.0.0.1:5000

pause

