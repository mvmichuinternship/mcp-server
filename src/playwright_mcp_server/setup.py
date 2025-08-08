#!/usr/bin/env python3
"""
Setup script for MCP LangChain Browser Automation
"""

import os
import sys
import subprocess
import requests
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")

    requirements = [
        "langchain>=0.1.0",
        "langchain-community>=0.0.20",
        "langchain-core>=0.1.0",
        "openai>=1.0.0",
        "requests>=2.31.0",
        "pydantic>=2.5.0",
        "aiohttp>=3.9.0",
        "playwright>=1.40.0",
        "mcp>=0.1.0"
    ]

    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False

    print("✅ All packages installed successfully")
    return True

def install_playwright_browsers():
    """Install Playwright browser binaries"""
    print("\n🌐 Installing Playwright browsers...")
    try:
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        print("✅ Playwright browsers installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Playwright browsers: {e}")
        return False

def check_lm_studio():
    """Check if LM Studio is running and accessible"""
    print("\n🤖 Checking LM Studio connection...")

    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            if models.get("data"):
                print("✅ LM Studio is running")
                print(f"Available models: {[m['id'] for m in models['data']]}")
                return True
            else:
                print("⚠️  LM Studio is running but no models are loaded")
                return False
        else:
            print(f"❌ LM Studio responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to LM Studio at localhost:1234")
        print("Make sure LM Studio is running and listening on port 1234")
        return False
    except Exception as e:
        print(f"❌ Error checking LM Studio: {e}")
        return False

def create_example_config():
    """Create an example configuration file"""
    config = {
        "lm_studio": {
            "url": "http://localhost:1234/v1",
            "model": "gemma",
            "temperature": 0.1,
            "max_tokens": 1000
        },
        "mcp_server": {
            "script_path": "playwright_mcp_server.py",
            "timeout": 30
        },
        "browser": {
            "headless": False,
            "viewport": {"width": 1920, "height": 1080}
        }
    }

    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("✅ Created example config.json file")

def create_start_script():
    """Create a simple start script"""
    start_script = '''#!/bin/bash
# Start script for MCP LangChain Browser Automation

echo "🚀 Starting MCP LangChain Browser Automation..."

# Check if LM Studio is running
if ! curl -s http://localhost:1234/v1/models > /dev/null; then
    echo "❌ LM Studio is not running or not accessible at localhost:1234"
    echo "Please start LM Studio and load a Gemma model first"
    exit 1
fi

echo "✅ LM Studio is running"

# Run the client
python3 mcp_langchain_client.py
'''

    with open("start.sh", "w") as f:
        f.write(start_script)

    os.chmod("start.sh", 0o755)
    print("✅ Created start.sh script")

def main():
    """Main setup function"""
    print("🔧 Setting up MCP LangChain Browser Automation")
    print("=" * 50)

    # Check Python version
    check_python_version()

    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)

    # Install Playwright browsers
    if not install_playwright_browsers():
        print("⚠️  Playwright browser installation failed, but continuing...")

    # Check LM Studio
    lm_studio_ok = check_lm_studio()

    # Create config files
    create_example_config()
    create_start_script()

    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Make sure your MCP Playwright server script is named 'playwright_mcp_server.py'")

    if not lm_studio_ok:
        print("2. Start LM Studio and load a Gemma model")
        print("3. Make sure LM Studio is running on localhost:1234")

    print("4. Run: python3 mcp_langchain_client.py")
    print("   Or use: ./start.sh")

    print("\n📝 Configuration:")
    print("- Edit config.json to customize settings")
    print("- Check requirements.txt for all dependencies")

    print("\n🔍 Troubleshooting:")
    print("- If tools don't work, check MCP server logs")
    print("- If LM Studio connection fails, verify the API is enabled")
    print("- For Playwright issues, try: playwright install chromium")

if __name__ == "__main__":
    main()