#!/bin/bash
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
