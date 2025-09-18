#!/usr/bin/env python3
"""
Launch the CE Idea Interest Analysis Streamlit Dashboard
"""

import subprocess
import sys
import webbrowser
import time
import os

def main():
    print("🚀 CE Idea Interest Analysis - Streamlit Dashboard")
    print("=" * 60)
    print()
    print("🔗 Dashboard URL: http://localhost:8501")
    print("📊 Features:")
    print("  ✅ Executive Summary (perfect for newcomers)")
    print("  ✅ Interactive Analysis (filters & exploration)")
    print("  ✅ Detailed Statistical Explorer")
    print("  ✅ Complete Verification Tools")
    print("  ✅ Professional responsive design")
    print("  ✅ All results fully transparent and verifiable")
    print()
    
    # Check if already running
    try:
        import requests
        response = requests.get("http://localhost:8501", timeout=2)
        if response.status_code == 200:
            print("✅ Dashboard is already running!")
            print("🌐 Opening browser...")
            webbrowser.open("http://localhost:8501")
            return
    except:
        pass
    
    print("🎬 Launching dashboard...")
    
    # Try to open browser after a short delay
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Launch Streamlit
    try:
        env = os.environ.copy()
        env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py", 
            "--server.port", "8501",
            "--server.headless", "true"
        ]
        
        # Send empty email response
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            env=env,
            text=True
        )
        
        # Send empty email
        process.stdin.write("\n")
        process.stdin.flush()
        
        print("✅ Dashboard launched successfully!")
        print("⏹️  Press Ctrl+C to stop the dashboard")
        
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("📋 Manual launch: streamlit run streamlit_dashboard.py")

if __name__ == "__main__":
    main()