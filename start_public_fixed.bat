@echo off
chcp 65001 >nul
echo ====================================
echo  تشغيل النظام مع Cloudflare Tunnel
echo ====================================
echo.

REM Check if cloudflared exists
if not exist "cloudflared-windows-amd64.exe" (
    echo ❌ ملف cloudflared-windows-amd64.exe غير موجود!
    pause
    exit
)

echo خطوات التشغيل:
echo.
echo 1. تأكد من أن Flask يعمل في نافذة منفصلة
echo 2. افتح http://localhost:5000 في المتصفح للتأكد
echo 3. إذا كان يعمل، اضغط Enter للمتابعة
echo.
pause

echo.
echo جاري التحقق من Flask...
timeout /t 2 /nobreak >nul

REM Try to check if Flask is running
curl -s http://localhost:5000 >nul 2>&1
if errorlevel 1 (
    echo.
    echo ⚠️  تحذير: Flask قد لا يعمل على localhost:5000
    echo.
    echo يرجى التأكد من:
    echo - Flask يعمل في نافذة منفصلة
    echo - يمكنك فتح http://localhost:5000 في المتصفح
    echo.
    echo هل تريد المتابعة على أي حال؟ (Y/N)
    choice /C YN /N /M ""
    if errorlevel 2 exit
)

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

REM Start cloudflared tunnel with explicit localhost
cloudflared-windows-amd64.exe tunnel --url http://127.0.0.1:5000

pause

