#!/usr/bin/env python3
"""
Simple launcher for the CE Analysis Web Dashboard
"""

import webbrowser
import time
from web_dashboard import main

if __name__ == "__main__":
    print("🚀 CE Idea Interest Analysis - Web Dashboard Launcher")
    print("=" * 60)
    print()
    print("This will:")
    print("✅ Create professional interactive visualizations")
    print("✅ Launch a localhost web server")
    print("✅ Open your browser automatically")
    print("✅ Provide full verification tools")
    print()
    print("🔗 Dashboard will be available at: http://localhost:8000")
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Try running: pip install plotly pandas numpy scipy matplotlib seaborn")