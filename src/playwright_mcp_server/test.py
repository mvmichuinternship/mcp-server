#!/usr/bin/env python3
"""
Simple MCP communication test to debug the protocol
"""

import asyncio
import json
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_mcp_communication():
    """Test direct MCP communication"""

    # Start the MCP server
    process = await asyncio.create_subprocess_exec(
        "python3", "playwright_mcp_server.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        # Wait for server to start
        await asyncio.sleep(2)

        print("🔧 Testing MCP protocol communication...")

        # Step 1: Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }

        request_json = json.dumps(init_request) + "\n"
        print(f"📤 Sending: {request_json.strip()}")

        process.stdin.write(request_json.encode())
        await process.stdin.drain()

        # Read response
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10)
        response_text = response_line.decode().strip()
        print(f"📥 Received: {response_text}")

        try:
            response = json.loads(response_text)
            if "error" in response:
                print(f"❌ Initialize error: {response['error']}")
                return False
            else:
                print("✅ Initialize successful")
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response: {response_text}")
            return False

        # Step 2: Send initialized notification
        init_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }

        notification_json = json.dumps(init_notification) + "\n"
        print(f"📤 Sending notification: {notification_json.strip()}")

        process.stdin.write(notification_json.encode())
        await process.stdin.drain()

        # Step 3: List tools
        list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        request_json = json.dumps(list_request) + "\n"
        print(f"📤 Sending: {request_json.strip()}")

        process.stdin.write(request_json.encode())
        await process.stdin.drain()

        # Read response
        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10)
        response_text = response_line.decode().strip()
        print(f"📥 Received: {response_text}")

        try:
            response = json.loads(response_text)
            if "error" in response:
                print(f"❌ Tools list error: {response['error']}")
                return False
            else:
                print("✅ Tools list successful")
                if "result" in response and "tools" in response["result"]:
                    tools = response["result"]["tools"]
                    print(f"🛠️  Available tools: {[t['name'] for t in tools]}")

                    # Step 4: Test a simple tool call
                    if tools:
                        test_tool = "get_browser_status"  # This should be a safe tool to test

                        tool_request = {
                            "jsonrpc": "2.0",
                            "id": 3,
                            "method": "tools/call",
                            "params": {
                                "name": test_tool,
                                "arguments": {}
                            }
                        }

                        request_json = json.dumps(tool_request) + "\n"
                        print(f"📤 Testing tool call: {request_json.strip()}")

                        process.stdin.write(request_json.encode())
                        await process.stdin.drain()

                        # Read response
                        response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10)
                        response_text = response_line.decode().strip()
                        print(f"📥 Tool response: {response_text}")

                        try:
                            response = json.loads(response_text)
                            if "error" in response:
                                print(f"❌ Tool call error: {response['error']}")
                            else:
                                print("✅ Tool call successful")
                                return True
                        except json.JSONDecodeError:
                            print(f"❌ Invalid JSON in tool response: {response_text}")

        except json.JSONDecodeError:
            print(f"❌ Invalid JSON response: {response_text}")
            return False

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")

        # Check stderr for server errors
        try:
            stderr_data = await asyncio.wait_for(process.stderr.read(1024), timeout=1)
            if stderr_data:
                print(f"🚨 Server stderr: {stderr_data.decode()}")
        except:
            pass

        return False

    finally:
        # Cleanup
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

async def test_server_directly():
    """Test if the server starts at all"""
    print("🧪 Testing if MCP server starts...")

    process = await asyncio.create_subprocess_exec(
        "python3", "playwright_mcp_server.py",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        # Wait a moment
        await asyncio.sleep(3)

        # Check if process is still running
        if process.returncode is None:
            print("✅ MCP server process started successfully")

            # Try to read any output
            try:
                stdout_data = await asyncio.wait_for(process.stdout.read(512), timeout=1)
                if stdout_data:
                    print(f"📄 Server stdout: {stdout_data.decode()}")
            except asyncio.TimeoutError:
                print("📄 No immediate stdout output (this is normal)")

            try:
                stderr_data = await asyncio.wait_for(process.stderr.read(512), timeout=1)
                if stderr_data:
                    print(f"🚨 Server stderr: {stderr_data.decode()}")
            except asyncio.TimeoutError:
                print("📄 No stderr output (this is good)")

            return True
        else:
            print(f"❌ MCP server exited with code: {process.returncode}")
            stderr_data = await process.stderr.read()
            if stderr_data:
                print(f"🚨 Server stderr: {stderr_data.decode()}")
            return False

    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()

async def main():
    print("🧪 MCP Communication Debug Test")
    print("=" * 40)

    # First test if server starts
    if not await test_server_directly():
        print("\n❌ Server startup test failed. Check your MCP server file.")
        return

    print("\n" + "=" * 40)

    # Test communication
    if await test_mcp_communication():
        print("\n🎉 MCP communication test passed!")
        print("You can now try the full integration.")
    else:
        print("\n❌ MCP communication test failed.")
        print("Check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main())