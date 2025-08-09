"""
Enhanced MCP Client with proper image handling for LangChain
Supports screenshot analysis by converting images to base64 format
"""

import asyncio
import json
import subprocess
import logging
import base64
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from io import BytesIO

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI

import requests

# For direct LM Studio API calls
try:
    import aiohttp
except ImportError:
    print("Warning: aiohttp not installed. Direct API calls will not work.")
    print("Install with: pip install aiohttp")
    aiohttp = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPToolInput(BaseModel):
    """Dynamic input schema for MCP tools"""
    pass

class MCPTool(BaseTool):
    """Enhanced MCP tool with proper image handling for vision models"""

    mcp_client: Any = None

    def __init__(self, tool_name: str, tool_description: str, mcp_client: 'MCPClient', input_schema: Optional[Dict] = None):
        # Create dynamic input model if schema is provided
        args_schema = None

        if input_schema and input_schema.get("properties"):
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            annotations = {}
            field_info = {}

            for prop_name, prop_info in properties.items():
                if prop_info.get("type") == "integer":
                    field_type = int
                elif prop_info.get("type") == "boolean":
                    field_type = bool
                elif prop_info.get("type") == "array":
                    field_type = List[str]
                else:
                    field_type = str

                if prop_name in required:
                    annotations[prop_name] = field_type
                    field_info[prop_name] = Field(description=prop_info.get("description", ""))
                else:
                    annotations[prop_name] = Optional[field_type]
                    field_info[prop_name] = Field(default=None, description=prop_info.get("description", ""))

            if annotations:
                DynamicInput = type(
                    f"{tool_name}Input",
                    (BaseModel,),
                    {
                        "__annotations__": annotations,
                        **field_info
                    }
                )
                args_schema = DynamicInput

        super().__init__(
            name=tool_name,
            description=tool_description,
            args_schema=args_schema
        )
        self.mcp_client = mcp_client

    def _run(self, **kwargs) -> str:
        """Run the tool synchronously"""
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        """Enhanced async tool execution with proper image handling"""
        try:
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            result = await self.mcp_client.call_tool(self.name, filtered_kwargs)

            if isinstance(result, dict) and "content" in result:
                return await self._process_mcp_response(result["content"])

            return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)

        except Exception as e:
            logger.error(f"Error calling {self.name}: {e}")
            return f"Error calling {self.name}: {str(e)}"

    async def _process_mcp_response(self, content_items: List[Dict]) -> str:
        """Process MCP response content, handling both text and images"""
        if not isinstance(content_items, list):
            return str(content_items)

        text_parts = []
        images = []

        for item in content_items:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                image_info = await self._process_image_content(item)
                if image_info:
                    images.append(image_info)
                    text_parts.append(f"[SCREENSHOT CAPTURED: {image_info.get('description', 'Image')}]")

        # Store images for potential model analysis
        if images:
            self.mcp_client.store_images(images)
            logger.info(f"Stored {len(images)} images for analysis")

        text_result = "\n".join(text_parts) if text_parts else "Operation completed"

        # For screenshot tools, try immediate analysis
        if images and "screenshot" in self.name.lower():
            logger.info("Screenshot tool detected, attempting analysis...")
            try:
                analysis = await self.mcp_client.analyze_last_screenshot()
                if analysis and "not working properly" not in analysis and "doesn't support vision" not in analysis:
                    text_result += f"\n\n🔍 Screenshot Analysis:\n{analysis}"
                else:
                    text_result += f"\n\n📸 Screenshot saved for reference"
                    if analysis:
                        text_result += f"\nNote: {analysis}"
            except Exception as e:
                logger.error(f"Screenshot analysis failed: {e}")
                text_result += f"\n\n📸 Screenshot saved but analysis failed: {str(e)}"

        if images and "screenshot" not in self.name.lower():
            text_result += f"\n\n📸 {len(images)} image(s) captured and available for analysis"

        return text_result

    async def _process_image_content(self, image_item: Dict) -> Optional[Dict]:
        """Process image content from MCP response"""
        try:
            image_data = image_item.get("data", "")
            mime_type = image_item.get("mimeType", "image/png")

            if not image_data:
                return None

            # Handle base64 data URLs
            if image_data.startswith("data:"):
                # Extract base64 part from data URL
                base64_data = image_data.split(",", 1)[1] if "," in image_data else image_data
            else:
                base64_data = image_data

            return {
                "type": "image",
                "base64": base64_data,
                "mime_type": mime_type,
                "description": f"Screenshot ({mime_type})",
                "size_info": f"Base64 length: {len(base64_data)}"
            }

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

class MCPClient:
    """Enhanced MCP Client with image storage and analysis capabilities"""

    def __init__(self, server_command: List[str]):
        self.server_command = server_command
        self.process = None
        self.tools_info = {}
        self.request_id = 0
        self.stored_images = []  # Store images for analysis
        self.llm = None  # Will be set by the agent

    def store_images(self, images: List[Dict]):
        """Store images for later analysis"""
        self.stored_images.extend(images)
        # Keep only the last 5 images to avoid memory issues
        if len(self.stored_images) > 5:
            self.stored_images = self.stored_images[-5:]

    async def debug_last_image(self) -> Optional[str]:
        """Comprehensive debug information about the last captured image"""
        if not self.stored_images:
            return "No images stored"

        latest_image = self.stored_images[-1]

        # Check if base64 is valid
        try:
            base64_bytes = base64.b64decode(latest_image['base64'])
            is_valid_base64 = True
            decoded_size = len(base64_bytes)
        except Exception as e:
            is_valid_base64 = False
            decoded_size = 0

        # Check image format
        image_format = "Unknown"
        if latest_image['base64'].startswith('/9j/'):
            image_format = "JPEG"
        elif latest_image['base64'].startswith('iVBORw0KGgo'):
            image_format = "PNG"
        elif latest_image['base64'].startswith('UklGR'):
            image_format = "WebP"

        debug_info = f"""
🔍 Image Debug Info:
- Storage: {'✅ Image stored' if latest_image else '❌ No image'}
- Type: {latest_image['type']}
- MIME Type: {latest_image['mime_type']}
- Description: {latest_image['description']}
- Base64 Length: {len(latest_image['base64'])} characters
- Base64 Preview (first 100 chars): {latest_image['base64'][:100]}...
- Base64 Preview (last 50 chars): ...{latest_image['base64'][-50:]}
- Valid Base64: {'✅ Yes' if is_valid_base64 else '❌ No'}
- Decoded Size: {decoded_size} bytes ({decoded_size/1024:.1f} KB)
- Detected Format: {image_format}
- Data URL Format: data:{latest_image['mime_type']};base64,[{len(latest_image['base64'])} chars]
"""
        return debug_info

    async def test_lm_studio_vision(self) -> str:
        """Test LM Studio's vision capabilities - optimized for Qwen 2.5-VL"""
        try:
            if not aiohttp:
                return "❌ aiohttp not installed. Run: pip install aiohttp"

            base_url = getattr(self.llm, 'base_url', 'http://localhost:1234/v1')
            if hasattr(self.llm, 'openai_api_base'):
                base_url = self.llm.openai_api_base

            api_url = f"{base_url.rstrip('/')}/chat/completions"

            # Test 1: Basic connection
            test_payload = {
                "model": "qwen2.5-vl-7b",  # Use specific model name
                "messages": [{"role": "user", "content": "Hello! You are Qwen2.5-VL. Can you see and analyze images?"}],
                "max_tokens": 100
            }

            async with aiohttp.ClientSession() as session:
                # Test basic connection
                try:
                    async with session.post(
                        api_url,
                        json=test_payload,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            return f"❌ Basic connection failed: {resp.status} - {error_text}"

                        result = await resp.json()
                        basic_response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")

                except Exception as e:
                    return f"❌ Connection error: {str(e)}"

                # Test 2: Try with Qwen 2.5-VL optimized test image (small red square)
                test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="

                # Format 1: Standard multimodal format
                vision_payload_1 = {
                    "model": "qwen2.5-vl-7b",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{test_image_b64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": "What color is this image? Just say the color."
                                }
                            ]
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 50
                }

                vision_response_1 = None
                vision_status_1 = None

                try:
                    async with session.post(
                        api_url,
                        json=vision_payload_1,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=45)  # Qwen can be slow
                    ) as resp:
                        vision_status_1 = resp.status
                        if resp.status == 200:
                            result = await resp.json()
                            vision_response_1 = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
                        else:
                            vision_response_1 = await resp.text()

                except Exception as e:
                    vision_response_1 = f"Error: {str(e)}"
                    vision_status_1 = "Exception"

                # Test 3: Try alternative Qwen format
                vision_payload_2 = {
                    "model": "current",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"<image>data:image/png;base64,{test_image_b64}</image>\n\nDescribe this image briefly."
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 50
                }

                vision_response_2 = None
                vision_status_2 = None

                try:
                    async with session.post(
                        api_url,
                        json=vision_payload_2,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=45)
                    ) as resp:
                        vision_status_2 = resp.status
                        if resp.status == 200:
                            result = await resp.json()
                            vision_response_2 = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
                        else:
                            vision_response_2 = await resp.text()

                except Exception as e:
                    vision_response_2 = f"Error: {str(e)}"
                    vision_status_2 = "Exception"

            # Analyze results
            format1_works = (vision_status_1 == 200 and
                           vision_response_1 and
                           ('red' in vision_response_1.lower() or 'color' in vision_response_1.lower()))

            format2_works = (vision_status_2 == 200 and
                           vision_response_2 and
                           ('red' in vision_response_2.lower() or 'color' in vision_response_2.lower()))

            status = "✅ Working" if (format1_works or format2_works) else "❌ Not Working"

            return f"""
🧪 Qwen 2.5-VL Test Results:
- API URL: {api_url}
- Basic Connection: ✅ Success (Status 200)
- Basic Response: "{basic_response[:150]}..."

📊 Vision Tests:
- Format 1 (Standard): {'✅' if vision_status_1 == 200 else '❌'} {vision_status_1}
  Response: "{(vision_response_1 or '')[:100]}..."

- Format 2 (Alternative): {'✅' if vision_status_2 == 200 else '❌'} {vision_status_2}
  Response: "{(vision_response_2 or '')[:100]}..."

🎯 Overall Vision Status: {status}
{'✅ Qwen 2.5-VL vision is working!' if (format1_works or format2_works) else '❌ Vision may not be properly configured'}

Recommended: Use Format {'1' if format1_works else '2' if format2_works else '1 (if you fix the setup)'}
"""

        except Exception as e:
            return f"❌ Test failed with error: {str(e)}"

    async def analyze_last_screenshot(self) -> Optional[str]:
        """Analyze the last screenshot using LM Studio's vision format - optimized for Qwen 2.5-VL"""
        if not self.stored_images or not self.llm:
            return None

        try:
            latest_image = self.stored_images[-1]
            logger.info(f"Attempting to analyze image with Qwen 2.5-VL: {latest_image['description']}")

            # Qwen 2.5-VL specific format attempts

            # Approach 1: Qwen 2.5-VL prefers simple message format
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{latest_image['mime_type']};base64,{latest_image['base64']}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Describe this screenshot in detail. What do you see on this webpage? Include layout, text content, buttons, forms, and any interactive elements."
                            }
                        ]
                    }
                ]

                response = await self.llm.ainvoke(messages)

                if (response.content and
                    "I would need to see it directly" not in response.content and
                    "cannot see" not in response.content.lower() and
                    len(response.content.strip()) > 50):
                    logger.info("Qwen 2.5-VL analysis successful via LangChain!")
                    return response.content

            except Exception as e1:
                logger.debug(f"Qwen 2.5-VL LangChain format failed: {e1}")

            # Approach 2: Direct API call with Qwen-optimized format
            try:
                if not aiohttp:
                    raise Exception("aiohttp not available")

                # Qwen 2.5-VL API format
                payload = {
                    "model": "qwen2.5-vl-7b",  # Specific model name
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{latest_image['mime_type']};base64,{latest_image['base64']}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": "Analyze this screenshot. Describe the webpage layout, visible text, buttons, forms, navigation elements, and overall content. Be specific and detailed."
                                }
                            ]
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "stream": False
                }

                # Get the base URL properly
                base_url = getattr(self.llm, 'base_url', 'http://localhost:1234/v1')
                if hasattr(self.llm, 'openai_api_base'):
                    base_url = self.llm.openai_api_base

                api_url = f"{base_url.rstrip('/')}/chat/completions"
                logger.info(f"Trying Qwen 2.5-VL direct API call to: {api_url}")

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        api_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=90)  # Qwen can be slower
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            if result.get("choices") and result["choices"][0].get("message"):
                                content = result["choices"][0]["message"]["content"]
                                if (content and
                                    "I would need to see it directly" not in content and
                                    "cannot see" not in content.lower() and
                                    len(content.strip()) > 50):
                                    logger.info("Qwen 2.5-VL direct API call successful!")
                                    return content
                        else:
                            error_text = await resp.text()
                            logger.error(f"Qwen 2.5-VL API call failed: {resp.status} - {error_text}")

            except Exception as e2:
                logger.debug(f"Qwen 2.5-VL direct API approach failed: {e2}")

            # Approach 3: Try alternative message structure
            try:
                if aiohttp:
                    # Some versions of Qwen prefer text first
                    alt_payload = {
                        "model": "current",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"<image>data:{latest_image['mime_type']};base64,{latest_image['base64']}</image>\n\nPlease analyze this screenshot and describe what you see. Include details about the webpage layout, text content, interactive elements, and overall design."
                            }
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2000
                    }

                    api_url = f"{base_url.rstrip('/')}/chat/completions"

                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            api_url,
                            json=alt_payload,
                            headers={"Content-Type": "application/json"},
                            timeout=aiohttp.ClientTimeout(total=90)
                        ) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                if result.get("choices") and result["choices"][0].get("message"):
                                    content = result["choices"][0]["message"]["content"]
                                    if (content and
                                        "I would need to see it directly" not in content and
                                        "cannot see" not in content.lower() and
                                        len(content.strip()) > 50):
                                        logger.info("Qwen 2.5-VL alternative format successful!")
                                        return content

            except Exception as e3:
                logger.debug(f"Qwen 2.5-VL alternative format failed: {e3}")

            # Test basic vision capability
            try:
                test_prompt = "Are you a vision-language model? Can you analyze images? Please answer YES or NO."
                test_response = await self.llm.ainvoke(test_prompt)

                if ("NO" in test_response.content.upper() or
                    "cannot" in test_response.content.lower() or
                    "do not" in test_response.content.lower()):
                    return "Screenshot captured but Qwen 2.5-VL model reports it cannot see images. Check that the vision model is properly loaded in LM Studio."

            except Exception as e4:
                logger.debug(f"Qwen 2.5-VL vision capability test failed: {e4}")

            return f"""Screenshot captured ({latest_image['size_info']}) but Qwen 2.5-VL cannot process the image.

Possible issues:
1. Make sure 'qwen/qwen2.5-vl-7b' is fully loaded in LM Studio
2. Check that LM Studio shows 'Vision' capability for this model
3. Verify the model has enough VRAM allocated
4. Try reloading the model in LM Studio

Image info: {latest_image['mime_type']}, {len(latest_image['base64'])} characters base64"""

        except Exception as e:
            logger.error(f"Error analyzing screenshot with Qwen 2.5-VL: {e}")
            return f"Screenshot captured but analysis failed: {str(e)}"

    async def start(self):
        """Start the MCP server process with enhanced buffer handling"""
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024*1024*50  # 50MB buffer for large images
            )

            await asyncio.sleep(2)

            # Initialize MCP protocol
            init_response = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "langchain-mcp-client",
                    "version": "1.0.0"
                }
            })

            logger.info(f"MCP Server initialized: {init_response}")

            await self._send_notification("notifications/initialized")

            # List available tools
            tools_response = await self._send_request("tools/list", {})
            if "tools" in tools_response:
                for tool in tools_response["tools"]:
                    self.tools_info[tool["name"]] = tool

            logger.info(f"Connected to MCP server. Available tools: {list(self.tools_info.keys())}")

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            if self.process and self.process.stderr:
                try:
                    stderr_output = await asyncio.wait_for(self.process.stderr.read(1024), timeout=1)
                    if stderr_output:
                        logger.error(f"Server stderr: {stderr_output.decode()}")
                except asyncio.TimeoutError:
                    logger.error("Could not read stderr from server")
            raise

    async def stop(self):
        """Stop the MCP server process"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

    def _get_next_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id

    async def _send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Send request with enhanced chunked response handling for large images"""
        if not self.process:
            raise Exception("MCP server not started")

        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": method,
            "params": params or {}
        }

        request_json = json.dumps(request) + "\n"
        logger.debug(f"Sending request: {request_json.strip()}")

        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()

        # Enhanced response reading for large images
        try:
            response_buffer = b""
            timeout = 120  # Increased timeout for large screenshots
            chunk_size = 65536  # 64KB chunks

            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.process.stdout.read(chunk_size),
                        timeout=timeout
                    )

                    if not chunk:
                        break

                    response_buffer += chunk
                    response_text = response_buffer.decode('utf-8', errors='ignore')

                    # Look for complete JSON responses
                    lines = response_text.split('\n')
                    for line in lines[:-1]:
                        line = line.strip()
                        if line:
                            try:
                                response = json.loads(line)
                                if "error" in response:
                                    raise Exception(f"MCP Error: {response['error']}")
                                if "result" in response and response.get("id"):
                                    return response["result"]
                            except json.JSONDecodeError:
                                continue

                    # Keep the last incomplete line
                    if lines:
                        last_line = lines[-1]
                        response_buffer = last_line.encode('utf-8')

                except asyncio.TimeoutError:
                    # Try to parse partial response
                    if response_buffer:
                        try:
                            response_text = response_buffer.decode('utf-8', errors='ignore')
                            response = json.loads(response_text.strip())
                            if "error" in response:
                                raise Exception(f"MCP Error: {response['error']}")
                            return response.get("result", {})
                        except json.JSONDecodeError:
                            pass
                    raise Exception("MCP server response timeout")

        except Exception as e:
            logger.error(f"Error reading response: {e}")
            raise

        raise Exception("No valid response received from MCP server")

    async def _send_notification(self, method: str, params: Optional[Dict] = None):
        """Send notification to MCP server"""
        if not self.process:
            raise Exception("MCP server not started")

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }

        notification_json = json.dumps(notification) + "\n"
        logger.debug(f"Sending notification: {notification_json.strip()}")

        self.process.stdin.write(notification_json.encode())
        await self.process.stdin.drain()

    async def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Call a tool on the MCP server"""
        return await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

    def create_langchain_tools(self) -> List[MCPTool]:
        """Create LangChain tools from MCP tools"""
        langchain_tools = []

        for tool_name, tool_info in self.tools_info.items():
            try:
                tool = MCPTool(
                    tool_name=tool_name,
                    tool_description=tool_info.get("description", f"MCP tool: {tool_name}"),
                    mcp_client=self,
                    input_schema=tool_info.get("inputSchema")
                )
                langchain_tools.append(tool)
            except Exception as e:
                logger.warning(f"Failed to create tool {tool_name}: {e}")
                continue

        return langchain_tools

class BrowserAutomationAgent:
    """Enhanced browser automation agent with vision capabilities"""

    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1", use_vision: bool = False):
        self.lm_studio_url = lm_studio_url
        self.use_vision = use_vision
        self.mcp_client = None
        self.agent_executor = None

    async def setup(self, mcp_server_script_path: str):
        """Setup agent with enhanced vision capabilities"""
        try:
            # Initialize MCP client
            self.mcp_client = MCPClient([
                "python3", mcp_server_script_path
            ])
            await self.mcp_client.start()

            # Create tools
            tools = self.mcp_client.create_langchain_tools()
            if not tools:
                raise ValueError("No tools were successfully created from MCP server")

            logger.info(f"Created {len(tools)} tools from MCP server")

            # Setup LM Studio with vision support if requested
            model_name = "gemma-vision" if self.use_vision else "gemma"

            llm = ChatOpenAI(
                base_url=self.lm_studio_url,
                api_key="not-needed",
                model=model_name,
                temperature=0.1,
                max_tokens=2000
            )

            # Store LLM reference in MCP client for image analysis
            self.mcp_client.llm = llm

            # Test connection
            try:
                test_response = await llm.ainvoke("Hello, respond with 'Connection successful'")
                logger.info(f"LM Studio connection test: {test_response.content}")
            except Exception as e:
                logger.error(f"LM Studio connection failed: {e}")
                raise ValueError(f"Cannot connect to LM Studio: {str(e)}")

            # Enhanced prompt with vision capabilities
            system_prompt = """You are an advanced browser automation assistant with vision capabilities.

Available tools:
- launch_browser: Start a new browser session
- navigate: Go to a specific URL
- take_screenshot: Capture the current page (automatically analyzed).
- take_marked_screenshot: Capture with highlighted elements
- get_element_data: Inspect page elements
- click: Click on elements using the coordinates received from the moondream model
- input_text: Type into input fields
- key_press: Press keyboard keys
- scroll: Scroll the page
- wait_for_element: Wait for elements to appear
- get_page_info: Get current page information
- close_browser: Close the browser

When working with screenshots:
1. Screenshots are automatically analyzed when captured
2. Use the analysis to understand page content and layout
3. Make decisions based on what you can see in the screenshots
4. Be specific about elements you want to interact with

Best practices:
1. Always take a screenshot first to see the current state
2. Use element inspection before clicking or typing
3. Wait for elements to load when necessary
4. Provide clear feedback about your actions
5. Handle errors gracefully and suggest alternatives

Be helpful, accurate, and thorough in your browser automation tasks."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])

            # Create agent
            agent = create_tool_calling_agent(llm, tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=10,
                return_intermediate_steps=True
            )

            logger.info("Enhanced browser automation agent setup complete")

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            if self.mcp_client:
                await self.mcp_client.stop()
            raise

    async def run(self, query: str) -> str:
        """Run the agent with enhanced error handling"""
        if not self.agent_executor:
            raise ValueError("Agent not setup. Call setup() first.")

        try:
            result = await self.agent_executor.ainvoke({"input": query})

            # Include image information in response if available
            if self.mcp_client.stored_images:
                image_count = len(self.mcp_client.stored_images)
                result["output"] += f"\n\n📸 Session includes {image_count} screenshot(s) for reference."

            return result["output"]

        except Exception as e:
            logger.error(f"Error running agent: {e}")
            return f"Error: {str(e)}"

    async def get_last_screenshot_analysis(self) -> Optional[str]:
        """Get analysis of the last screenshot"""
        if self.mcp_client:
            return await self.mcp_client.analyze_last_screenshot()
        return None

    async def test_lm_studio_vision(self) -> str:
        """Test LM Studio's vision capabilities"""
        if self.mcp_client:
            return await self.mcp_client.test_lm_studio_vision()
        return "MCP client not available"

    async def save_last_screenshot(self, filename: str = "debug_screenshot.png") -> str:
        """Save the last screenshot to a file"""
        if self.mcp_client:
            return await self.mcp_client.save_last_image_to_file(filename)
        return "MCP client not available"

    async def debug_last_screenshot(self) -> Optional[str]:
        """Get debug information about the last screenshot"""
        if self.mcp_client:
            return await self.mcp_client.debug_last_image()
        return None

    async def cleanup(self):
        """Cleanup resources"""
        if self.mcp_client:
            await self.mcp_client.stop()

# Example usage with vision
async def example_with_vision():
    """Example using vision capabilities"""
    agent = BrowserAutomationAgent(use_vision=True)
    await agent.setup("playwright_mcp_server.py")

    try:
        result = await agent.run("""
        1. Launch a browser
        2. Navigate to https://example.com
        3. Take a screenshot and analyze the page
        4. Describe what you see and suggest possible interactions
        """)
        print(result)

        # Get additional screenshot analysis
        analysis = await agent.get_last_screenshot_analysis()
        if analysis:
            print(f"\nDetailed Analysis: {analysis}")

    finally:
        await agent.cleanup()

async def main():
    """Main interactive function"""
    import sys

    # Check for vision flag
    use_vision = "--vision" in sys.argv

    # Path to your MCP server script
    mcp_server_path = "playwright_mcp_server.py"  # Update this path as needed

    # Initialize the agent
    agent = BrowserAutomationAgent(use_vision=use_vision)

    try:
        # Setup the agent
        print("🤖 Setting up browser automation agent...")
        if use_vision:
            print("🔍 Vision capabilities enabled")
        await agent.setup(mcp_server_path)

        print("🤖 Browser Automation Agent is ready!")
        print("You can ask me to:")
        print("- Navigate to websites")
        print("- Take screenshots")
        print("- Click on elements")
        print("- Fill out forms")
        print("- Extract data from pages")
        if use_vision:
            print("- Analyze screenshots automatically")
        print("- And much more!")
        print("\nSpecial commands:")
        print("- 'debug' - Show debug info about last screenshot")
        print("- 'analyze' - Force analyze last screenshot")
        print("- 'test vision' - Test if vision is working")
        print("- 'test api' - Test LM Studio vision API directly")
        print("- 'save image' - Save last screenshot to file")
        print("\nType 'quit' to exit\n")

        # Interactive loop
        while True:
            try:
                user_input = input("\n👤 What would you like me to do? ")

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input.strip():
                    continue

                # Handle special debug commands
                if user_input.lower() == 'debug':
                    debug_info = await agent.debug_last_screenshot()
                    print(f"\n🔧 {debug_info}")
                    continue

                if user_input.lower() == 'analyze':
                    analysis = await agent.get_last_screenshot_analysis()
                    if analysis:
                        print(f"\n🔍 Analysis: {analysis}")
                    else:
                        print("\n❌ No screenshots to analyze")
                    continue

                if user_input.lower() == 'test vision':
                    result = await agent.run("Can you see and process images? Please respond with YES or NO and explain your capabilities.")
                    print(f"\n🧪 Vision Test: {result}")
                    continue

                if user_input.lower() == 'test api':
                    api_test = await agent.test_lm_studio_vision()
                    print(f"\n🔧 API Test Results: {api_test}")
                    continue

                if user_input.lower() == 'save image':
                    save_result = await agent.save_last_screenshot()
                    print(f"\n💾 {save_result}")
                    continue

                print("\n🤖 Working on it...")
                result = await agent.run(user_input)
                print(f"\n✅ Result: {result}")

                # Show additional screenshot analysis if available
                if use_vision:
                    analysis = await agent.get_last_screenshot_analysis()
                    if analysis and "Screenshot captured but" not in analysis:
                        print(f"\n🔍 Additional Analysis: {analysis}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    except Exception as e:
        print(f"❌ Failed to setup agent: {str(e)}")
        print("\nMake sure:")
        print("1. LM Studio is running on localhost:1234")
        print("2. A model is loaded in LM Studio")
        if use_vision:
            print("3. The loaded model supports vision (like LLaVA)")
        print("3. MCP server script path is correct")
        print("4. All required dependencies are installed")
        print("   - pip install langchain langchain-community langchain-openai playwright aiohttp")
        print("   - playwright install")
        if use_vision:
            print("5. Load a VISION model in LM Studio, such as:")
            print("   - qwen2-vl-2b-instruct")
            print("   - llava-v1.6-mistral-7b")
            print("   - minicpm-v-2_6")
            print("   - Make sure the model shows 'Vision' capabilities in LM Studio")

    finally:
        # Cleanup
        await agent.cleanup()
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "example":
        # Run example instead of interactive mode
        asyncio.run(example_with_vision())
    else:
        # Run interactive mode
        asyncio.run(main())





































