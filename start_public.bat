@echo off
echo ====================================
echo  تشغيل النظام مع Cloudflare Tunnel
echo ====================================
echo.

REM Check if cloudflared exists
if not exist "cloudflared-windows-amd64.exe" (
    echo ❌ ملف cloudflared-windows-amd64.exe غير موجود!
    echo يرجى التأكد من وجود الملف في نفس المجلد
    pause
    exit
)

echo جاري تشغيل Flask...
start "Flask Server" cmd /k "python app.py"

REM Wait longer for Flask to fully start
echo.
echo انتظر قليلاً حتى يبدأ Flask...
timeout /t 5 /nobreak >nul

REM Check if Flask is running
echo جاري التحقق من Flask...
timeout /t 2 /nobreak >nul

echo.
echo ====================================
echo جاري إنشاء نفق Cloudflare...
echo ====================================
echo.
echo ⚠️  سيظهر رابط عام - انسخه وشاركه مع الآخرين
echo ⚠️  تأكد من أن Flask يعمل في النافذة الأخرى
echo.
echo اضغط Ctrl+C لإيقاف النفق
echo.

REM Start cloudflared tunnel
cloudflared-windows-amd64.exe tunnel --url http://127.0.0.1:5000

pause

