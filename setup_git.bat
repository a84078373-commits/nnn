@echo off
chcp 65001 >nul
echo ====================================
echo  إعداد Git للنشر
echo ====================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git غير مثبت!
    echo.
    echo يرجى تثبيت Git من: https://git-scm.com/download/win
    pause
    exit
)

echo ✅ Git مثبت
echo.

echo خطوات النشر:
echo.
echo 1. اذهب إلى https://github.com/new
echo 2. أنشئ مستودع جديد (Public)
echo 3. انسخ رابط المستودع
echo.
set /p REPO_URL="أدخل رابط المستودع (مثل: https://github.com/username/repo.git): "

if "%REPO_URL%"=="" (
    echo ❌ لم تدخل رابط!
    pause
    exit
)

echo.
echo جاري تهيئة Git...
git init

echo.
echo جاري إضافة الملفات...
git add .

echo.
echo جاري حفظ التغييرات...
git commit -m "Initial commit - ASL Learning System"

echo.
echo جاري ربط المشروع بـ GitHub...
git branch -M main
git remote add origin %REPO_URL%

echo.
echo جاري رفع الملفات...
git push -u origin main

if errorlevel 1 (
    echo.
    echo ⚠️  قد تحتاج لتسجيل الدخول إلى GitHub
    echo.
    echo جرب:
    echo   git push -u origin main
    echo.
) else (
    echo.
    echo ✅ تم رفع الملفات بنجاح!
    echo.
    echo الآن اذهب إلى Render.com ونشر المشروع
)

pause

