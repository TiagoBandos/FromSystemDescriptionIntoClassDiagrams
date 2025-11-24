#!/usr/bin/env python3
"""
Startup script for the Diagram Generation API
This script checks dependencies and starts the Flask server
"""
from dotenv import load_dotenv
import os
import sys
import subprocess
from pathlib import Path

load_dotenv()

def check_environment():
    """Check if all required components are available"""
    print("🔍 Checking environment...")
    
    # Check for ANTHROPIC_API_KEY
    print(os.getenv("ANTHROPIC_API_KEY"))
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        print("   Please create a .env file with: ANTHROPIC_API_KEY=your_key_here")
        return False
    else:
        print("✅ ANTHROPIC_API_KEY found")
    
    # Check for PlantUML JAR
    jar_file = "plantuml-1.2025.3.jar"
    if not os.path.exists(jar_file):
        print(f"❌ {jar_file} not found in current directory")
        print(f"   Please download PlantUML JAR file and place it here")
        print(f"   Download from: https://plantuml.com/download")
        return False
    else:
        print(f"✅ {jar_file} found")
    
    # Check for Java
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Java is available")
        else:
            print("❌ Java not found - required for PlantUML rendering")
            return False
    except FileNotFoundError:
        print("❌ Java not found - required for PlantUML rendering")
        return False
    
    # Check for required Python files
    required_files = ["FinalImplementation.py", "flask_api_integration.py"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ {file} not found")
            return False
        else:
            print(f"✅ {file} found")
    
    # Check for data directory
    if not os.path.exists("data"):
        print("⚠️  'data' directory not found - creating it")
        os.makedirs("data", exist_ok=True)
        print("   Please add your PDF files to the 'data' directory")
    else:
        print("✅ 'data' directory found")
    
    return True

def install_dependencies():
    """Install required Python dependencies"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def start_api_server():
    """Start the Flask API server"""
    print("🚀 Starting Flask API server...")
    print("   Server will run on: http://127.0.0.1:5000")
    print("   Press Ctrl+C to stop the server")
    print("   You can now open the React interface and start generating diagrams!")
    print("-" * 50)
    
    try:
        # Import and run the Flask app
        from flask_api_integration import app
        app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def main():
    """Main function"""
    print("=" * 50)
    print("🎯 Diagram Generation API Startup")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)
    
    print("\n✅ Environment check passed!")
    
    # Ask user if they want to install dependencies
    install_deps = input("\n📦 Install/update Python dependencies? (y/n): ").lower().strip()
    if install_deps in ['y', 'yes']:
        if not install_dependencies():
            print("\n❌ Failed to install dependencies")
            sys.exit(1)
    
    # Create output directory
    os.makedirs("uml_outputs", exist_ok=True)
    
    # Start server
    print("\n" + "=" * 50)
    start_api_server()

if __name__ == "__main__":
    main()
