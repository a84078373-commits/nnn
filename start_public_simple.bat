@echo off
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
echo 1. شغّل Flask في نافذة منفصلة: python app.py
echo 2. انتظر حتى ترى "Running on http://127.0.0.1:5000"
echo 3. ثم اضغط Enter هنا لبدء Cloudflare Tunnel
echo.
pause

echo.
echo جاري إنشاء نفق Cloudflare...
echo.
echo ⚠️  سيظهر رابط عام - انسخه وشاركه مع الآخرين
echo.

REM Start cloudflared tunnel
cloudflared-windows-amd64.exe tunnel --url http://127.0.0.1:5000

pause

