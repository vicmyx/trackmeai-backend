#!/usr/bin/env python3
"""
Demo script to run the complete AI Mechanic RAG pipeline
"""

import subprocess
import time
import signal
import sys
import os
import webbrowser
from threading import Thread

def start_api_server():
    """Start the FastAPI server"""
    print("🚀 Starting AI Mechanic API server...")
    return subprocess.Popen([
        sys.executable, "api_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def start_react_app():
    """Start the React development server"""
    print("⚛️  Starting React frontend...")
    
    # Install dependencies if needed
    if not os.path.exists("node_modules"):
        print("📦 Installing React dependencies...")
        subprocess.run(["npm", "install"], check=True)
    
    return subprocess.Popen([
        "npm", "start"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down demo...")
    sys.exit(0)

def main():
    print("=== AI Mechanic Demo ===")
    print("This will start the complete RAG pipeline:")
    print("1. FastAPI backend (port 8000)")
    print("2. React frontend (port 3000)")
    print("\nPress Ctrl+C to stop both servers")
    print("=" * 40)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start API server
    api_process = start_api_server()
    
    # Give API server time to start
    print("⏳ Waiting for API server to start...")
    time.sleep(3)
    
    # Start React app
    react_process = start_react_app()
    
    # Give React time to start
    print("⏳ Waiting for React app to start...")
    time.sleep(10)
    
    # Open browser
    print("🌐 Opening browser...")
    webbrowser.open("http://localhost:3000")
    
    print("\n✅ Demo is running!")
    print("📍 Frontend: http://localhost:3000")
    print("📍 API: http://localhost:8000")
    print("📍 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the demo")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n🛑 Stopping servers...")
        
        if api_process:
            api_process.terminate()
            api_process.wait()
        
        if react_process:
            react_process.terminate()
            react_process.wait()
        
        print("✅ Demo stopped")

if __name__ == "__main__":
    main()