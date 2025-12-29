#!/bin/bash

echo "===================================="
echo "تشغيل النظام مع Cloudflare Tunnel"
echo "===================================="
echo ""

# Check if cloudflared exists
if [ ! -f "cloudflared-windows-amd64.exe" ]; then
    echo "❌ ملف cloudflared-windows-amd64.exe غير موجود!"
    echo "يرجى التأكد من وجود الملف في نفس المجلد"
    exit 1
fi

echo "جاري تشغيل Flask في الخلفية..."
python app.py &
FLASK_PID=$!

# Wait for Flask to start
sleep 3

echo ""
echo "جاري إنشاء نفق Cloudflare..."
echo ""
echo "⚠️  سيظهر رابط عام - انسخه وشاركه مع الآخرين"
echo ""

# Start cloudflared tunnel
./cloudflared-windows-amd64.exe tunnel --url http://localhost:5000

# Cleanup
kill $FLASK_PID 2>/dev/null

