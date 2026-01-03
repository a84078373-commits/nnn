import socket
import webbrowser
import pyperclip  # إذا تريد نسخ تلقائي

def main():
    # الحصول على IP المحلي
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    local_ip = s.getsockname()[0]
    s.close()
    
    print("\n" + "="*50)
    print("🌐 **روابط المشاركة**")
    print("="*50)
    
    print(f"\n🔗 **للمستخدمين على نفس الواي فاي:**")
    link1 = f"http://{local_ip}:5000"
    print(f"📋 {link1}")
    
    print(f"\n🌍 **للحصول على رابط خارجي:**")
    print("1. افتح: https://localhost.page")
    print("2. أدخل: 5000")
    print("3. انسخ الرابط المعطى")
    
    print(f"\n⚡ **اختبار سريع:**")
    print(f"• افتح هذا الرابط من جوالك: {link1}")
    
    # نسخ الرابط تلقائياً إذا pyperclip موجود
    try:
        import pyperclip
        pyperclip.copy(link1)
        print("✅ تم نسخ الرابط المحلي للحافظة")
    except:
        pass
    
    input("\n🎯 اضغط Enter لفتح موقع localhost.page...")
    webbrowser.open("https://localhost.page")
    
    input("\n🎯 اضغط Enter لفتح موقعك المحلي...")
    webbrowser.open(f"http://{local_ip}:5000")

if __name__ == "__main__":
    main()