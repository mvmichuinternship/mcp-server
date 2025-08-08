# #!/usr/bin/env python3
# """
# MCP Client using LangChain with Gemma model through LM Studio
# Connects to the Playwright MCP server for browser automation
# """

# import asyncio
# import json
# import subprocess
# import logging
# from typing import Any, Dict, List, Optional, Union
# from pathlib import Path

# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.tools import BaseTool
# from langchain_core.callbacks import CallbackManagerForToolRun
# from langchain_core.pydantic_v1 import BaseModel, Field
# try:
#     from langchain_openai import ChatOpenAI
# except ImportError:
#     try:
#         from langchain_community.chat_models import ChatOpenAI
#     except ImportError:
#         from langchain.chat_models import ChatOpenAI

# import requests

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class MCPToolInput(BaseModel):
#     """Dynamic input schema for MCP tools"""
#     pass

# class MCPTool(BaseTool):
#     """Base class for MCP tools with proper LangChain compatibility"""

#     # Define mcp_client as a class attribute to satisfy Pydantic
#     mcp_client: Any = None

#     def __init__(self, tool_name: str, tool_description: str, mcp_client: 'MCPClient', input_schema: Optional[Dict] = None):
#         # Create a dynamic input model if schema is provided
#         args_schema = None

#         if input_schema and input_schema.get("properties"):
#             properties = input_schema.get("properties", {})
#             required = input_schema.get("required", [])

#             # Create dynamic fields
#             annotations = {}
#             field_info = {}

#             for prop_name, prop_info in properties.items():
#                 # Basic type mapping
#                 if prop_info.get("type") == "integer":
#                     field_type = int
#                 elif prop_info.get("type") == "boolean":
#                     field_type = bool
#                 elif prop_info.get("type") == "array":
#                     field_type = List[str]
#                 else:
#                     field_type = str

#                 # Handle optional vs required fields
#                 if prop_name in required:
#                     annotations[prop_name] = field_type
#                     field_info[prop_name] = Field(description=prop_info.get("description", ""))
#                 else:
#                     annotations[prop_name] = Optional[field_type]
#                     field_info[prop_name] = Field(default=None, description=prop_info.get("description", ""))

#             # Create dynamic model
#             if annotations:
#                 DynamicInput = type(
#                     f"{tool_name}Input",
#                     (BaseModel,),
#                     {
#                         "__annotations__": annotations,
#                         **field_info
#                     }
#                 )
#                 args_schema = DynamicInput

#         # Initialize with all required parameters
#         super().__init__(
#             name=tool_name,
#             description=tool_description,
#             args_schema=args_schema
#         )

#         # Set the mcp_client after initialization
#         self.mcp_client = mcp_client

#     def _run(self, **kwargs) -> str:
#         """Run the tool synchronously"""
#         return asyncio.run(self._arun(**kwargs))

#     async def _arun(self, **kwargs) -> str:
#         """Run the tool asynchronously"""
#         try:
#             # Filter out None values
#             filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

#             result = await self.mcp_client.call_tool(self.name, filtered_kwargs)

#             # Handle the MCP response format
#             if isinstance(result, dict):
#                 # Check if it's a list of content items
#                 if "content" in result:
#                     content_items = result["content"]
#                     if isinstance(content_items, list):
#                         text_content = []
#                         for item in content_items:
#                             if item.get("type") == "text":
#                                 text_content.append(item.get("text", ""))
#                         return "\n".join(text_content) if text_content else json.dumps(result, indent=2)

#                 return json.dumps(result, indent=2)
#             else:
#                 return str(result)

#         except Exception as e:
#             logger.error(f"Error calling {self.name}: {e}")
#             return f"Error calling {self.name}: {str(e)}"

# class MCPClient:
#     """Client for communicating with MCP servers using stdio transport"""

#     def __init__(self, server_command: List[str]):
#         self.server_command = server_command
#         self.process = None
#         self.tools_info = {}
#         self.request_id = 0

#     async def start(self):
#         """Start the MCP server process"""
#         try:
#             self.process = await asyncio.create_subprocess_exec(
#                 *self.server_command,
#                 stdin=asyncio.subprocess.PIPE,
#                 stdout=asyncio.subprocess.PIPE,
#                 stderr=asyncio.subprocess.PIPE
#             )

#             # Wait a moment for the server to start
#             await asyncio.sleep(2)

#             # Initialize the server with proper MCP protocol
#             init_response = await self._send_request("initialize", {
#                 "protocolVersion": "2024-11-05",
#                 "capabilities": {
#                     "roots": {"listChanged": True},
#                     "sampling": {}
#                 },
#                 "clientInfo": {
#                     "name": "langchain-mcp-client",
#                     "version": "1.0.0"
#                 }
#             })

#             logger.info(f"MCP Server initialized: {init_response}")

#             # Send initialized notification
#             await self._send_notification("notifications/initialized")

#             # List available tools
#             tools_response = await self._send_request("tools/list", {})
#             if "tools" in tools_response:
#                 for tool in tools_response["tools"]:
#                     self.tools_info[tool["name"]] = tool

#             logger.info(f"Connected to MCP server. Available tools: {list(self.tools_info.keys())}")

#         except Exception as e:
#             logger.error(f"Failed to start MCP server: {e}")
#             if self.process and self.process.stderr:
#                 try:
#                     stderr_output = await asyncio.wait_for(self.process.stderr.read(1024), timeout=1)
#                     if stderr_output:
#                         logger.error(f"Server stderr: {stderr_output.decode()}")
#                 except asyncio.TimeoutError:
#                     logger.error("Could not read stderr from server")
#             raise

#     async def stop(self):
#         """Stop the MCP server process"""
#         if self.process:
#             try:
#                 self.process.terminate()
#                 await asyncio.wait_for(self.process.wait(), timeout=5)
#             except asyncio.TimeoutError:
#                 self.process.kill()
#                 await self.process.wait()

#     def _get_next_id(self) -> int:
#         """Get next request ID"""
#         self.request_id += 1
#         return self.request_id

#     async def _send_request(self, method: str, params: Optional[Dict] = None) -> Dict:
#         """Send a request to the MCP server"""
#         if not self.process:
#             raise Exception("MCP server not started")

#         request = {
#             "jsonrpc": "2.0",
#             "id": self._get_next_id(),
#             "method": method,
#             "params": params or {}
#         }

#         request_json = json.dumps(request) + "\n"
#         logger.debug(f"Sending request: {request_json.strip()}")

#         self.process.stdin.write(request_json.encode())
#         await self.process.stdin.drain()

#         # Read response with timeout
#         try:
#             response_line = await asyncio.wait_for(
#                 self.process.stdout.readline(),
#                 timeout=30
#             )
#         except asyncio.TimeoutError:
#             raise Exception("MCP server response timeout")

#         if not response_line:
#             raise Exception("MCP server closed connection")

#         response_text = response_line.decode().strip()
#         logger.debug(f"Received response: {response_text}")

#         try:
#             response = json.loads(response_text)
#         except json.JSONDecodeError as e:
#             raise Exception(f"Invalid JSON response: {response_text}")

#         if "error" in response:
#             raise Exception(f"MCP Error: {response['error']}")

#         return response.get("result", {})

#     async def _send_notification(self, method: str, params: Optional[Dict] = None):
#         """Send a notification to the MCP server"""
#         if not self.process:
#             raise Exception("MCP server not started")

#         notification = {
#             "jsonrpc": "2.0",
#             "method": method,
#             "params": params or {}
#         }

#         notification_json = json.dumps(notification) + "\n"
#         logger.debug(f"Sending notification: {notification_json.strip()}")

#         self.process.stdin.write(notification_json.encode())
#         await self.process.stdin.drain()

#     async def call_tool(self, tool_name: str, arguments: Dict) -> Any:
#         """Call a tool on the MCP server"""
#         return await self._send_request("tools/call", {
#             "name": tool_name,
#             "arguments": arguments
#         })

#     def create_langchain_tools(self) -> List[MCPTool]:
#         """Create LangChain tools from MCP tools"""
#         langchain_tools = []

#         for tool_name, tool_info in self.tools_info.items():
#             try:
#                 tool = MCPTool(
#                     tool_name=tool_name,
#                     tool_description=tool_info.get("description", f"MCP tool: {tool_name}"),
#                     mcp_client=self,
#                     input_schema=tool_info.get("inputSchema")
#                 )
#                 langchain_tools.append(tool)
#             except Exception as e:
#                 logger.warning(f"Failed to create tool {tool_name}: {e}")
#                 continue

#         return langchain_tools

# class BrowserAutomationAgent:
#     """LangChain agent for browser automation using MCP tools"""

#     def __init__(self, lm_studio_url: str = "http://localhost:1234/v1"):
#         self.lm_studio_url = lm_studio_url
#         self.mcp_client = None
#         self.agent_executor = None

#     async def setup(self, mcp_server_script_path: str):
#         """Setup the agent with MCP client and LM Studio connection"""

#         try:
#             # Initialize MCP client
#             self.mcp_client = MCPClient([
#                 "python3", mcp_server_script_path
#             ])
#             await self.mcp_client.start()

#             # Create LangChain tools from MCP tools
#             tools = self.mcp_client.create_langchain_tools()

#             if not tools:
#                 raise ValueError("No tools were successfully created from MCP server")

#             logger.info(f"Created {len(tools)} tools from MCP server")

#             # Setup LM Studio connection (using OpenAI-compatible API)
#             llm = ChatOpenAI(
#                 base_url=self.lm_studio_url,
#                 api_key="not-needed",  # LM Studio doesn't require API key
#                 model="gemma",  # This should match your loaded model in LM Studio
#                 temperature=0.1,
#                 max_tokens=1000
#             )

#             # Test LM Studio connection
#             try:
#                 test_response = await llm.ainvoke("Hello, can you respond with 'Connection successful'?")
#                 logger.info(f"LM Studio connection test: {test_response.content}")
#             except Exception as e:
#                 logger.error(f"LM Studio connection test failed: {e}")
#                 raise ValueError(f"Cannot connect to LM Studio: {str(e)}. Make sure it's running on localhost:1234 with a model loaded.")

#             # Create prompt template
#             prompt = ChatPromptTemplate.from_messages([
#                 ("system", """You are a helpful assistant that can control a web browser using various tools.

# Available browser automation tools:
# - launch_browser: Launch a new browser instance
# - navigate: Navigate to a URL
# - take_screenshot: Take a screenshot of the page
# - take_marked_screenshot: Take a screenshot with highlighted elements
# - get_element_data: Get information about elements on the page
# - click: Click on elements
# - input_text: Type text into input fields
# - key_press: Press keyboard keys
# - scroll: Scroll the page
# - wait_for_element: Wait for elements to appear
# - get_page_info: Get current page information
# - close_browser: Close the browser

# When helping with browser automation:
# 1. Always take screenshots to see what's on the page
# 2. Use get_element_data to inspect elements before interacting with them
# 3. Be specific about selectors (CSS selectors work best)
# 4. Wait for elements to load when needed
# 5. Provide clear feedback about what you're doing

# Answer the user's request step by step, using the available tools as needed."""),
#                 ("human", "{input}"),
#                 ("placeholder", "{agent_scratchpad}")
#             ])

#             # Create the agent
#             try:
#                 agent = create_tool_calling_agent(llm, tools, prompt)
#                 self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#                 logger.info("Browser automation agent setup complete")
#             except Exception as e:
#                 logger.error(f"Failed to create agent: {e}")
#                 raise ValueError(f"Failed to create LangChain agent: {str(e)}")

#         except Exception as e:
#             logger.error(f"Setup failed: {e}")
#             if self.mcp_client:
#                 await self.mcp_client.stop()
#             raise

#     async def run(self, query: str) -> str:
#         """Run the agent with a query"""
#         if not self.agent_executor:
#             raise ValueError("Agent not setup. Call setup() first.")

#         try:
#             result = await self.agent_executor.ainvoke({"input": query})
#             return result["output"]
#         except Exception as e:
#             logger.error(f"Error running agent: {e}")
#             return f"Error: {str(e)}"

#     async def cleanup(self):
#         """Cleanup resources"""
#         if self.mcp_client:
#             await self.mcp_client.stop()

# async def main():
#     """Main function to demonstrate the browser automation agent"""

#     # Path to your MCP server script
#     mcp_server_path = "playwright_mcp_server.py"  # Update this path

#     # Initialize the agent
#     agent = BrowserAutomationAgent()

#     try:
#         # Setup the agent
#         print("🤖 Setting up browser automation agent...")
#         await agent.setup(mcp_server_path)

#         print("🤖 Browser Automation Agent is ready!")
#         print("You can ask me to:")
#         print("- Navigate to websites")
#         print("- Take screenshots")
#         print("- Click on elements")
#         print("- Fill out forms")
#         print("- Extract data from pages")
#         print("- And much more!")
#         print("\nType 'quit' to exit\n")

#         # Interactive loop
#         while True:
#             try:
#                 user_input = input("\n👤 What would you like me to do? ")

#                 if user_input.lower() in ['quit', 'exit', 'q']:
#                     break

#                 if not user_input.strip():
#                     continue

#                 print("\n🤖 Working on it...")
#                 result = await agent.run(user_input)
#                 print(f"\n✅ Result: {result}")

#             except KeyboardInterrupt:
#                 break
#             except Exception as e:
#                 print(f"\n❌ Error: {str(e)}")

#     except Exception as e:
#         print(f"❌ Failed to setup agent: {str(e)}")
#         print("\nMake sure:")
#         print("1. LM Studio is running on localhost:1234")
#         print("2. A model (like Gemma) is loaded in LM Studio")
#         print("3. MCP server script path is correct")
#         print("4. All required dependencies are installed")
#         print("   - pip install langchain langchain-community playwright mcp")
#         print("   - playwright install")

#     finally:
#         # Cleanup
#         await agent.cleanup()
#         print("\n👋 Goodbye!")

# # Example usage functions
# async def example_web_scraping():
#     """Example: Web scraping workflow"""
#     agent = BrowserAutomationAgent()
#     await agent.setup("playwright_mcp_server.py")

#     try:
#         # Navigate to a website and extract data
#         result = await agent.run("""
#         1. Launch a browser
#         2. Navigate to https://example.com
#         3. Take a screenshot
#         4. Get information about all links on the page
#         5. Click on the first link (if any)
#         """)
#         print(result)

#     finally:
#         await agent.cleanup()

# async def example_form_filling():
#     """Example: Form filling workflow"""
#     agent = BrowserAutomationAgent()
#     await agent.setup("playwright_mcp_server.py")

#     try:
#         result = await agent.run("""
#         1. Launch a browser
#         2. Navigate to a website with a search form
#         3. Find the search input field
#         4. Type "playwright automation" in the search field
#         5. Click the search button
#         6. Take a screenshot of the results
#         """)
#         print(result)

#     finally:
#         await agent.cleanup()

# if __name__ == "__main__":
#     # Run the interactive agent
#     asyncio.run(main())

#     # Or run specific examples:
#     # asyncio.run(example_web_scraping())
#     # asyncio.run(example_form_filling())


























#!/usr/bin/env python3
"""
MCP Client using LangChain with Gemma model through LM Studio
Connects to the Playwright MCP server for browser automation
"""

import asyncio
import json
import subprocess
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPToolInput(BaseModel):
    """Dynamic input schema for MCP tools"""
    pass

class MCPTool(BaseTool):
    """Base class for MCP tools with proper LangChain compatibility"""

    # Define mcp_client as a class attribute to satisfy Pydantic
    mcp_client: Any = None

    def __init__(self, tool_name: str, tool_description: str, mcp_client: 'MCPClient', input_schema: Optional[Dict] = None):
        # Create a dynamic input model if schema is provided
        args_schema = None

        if input_schema and input_schema.get("properties"):
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            # Create dynamic fields
            annotations = {}
            field_info = {}

            for prop_name, prop_info in properties.items():
                # Basic type mapping
                if prop_info.get("type") == "integer":
                    field_type = int
                elif prop_info.get("type") == "boolean":
                    field_type = bool
                elif prop_info.get("type") == "array":
                    field_type = List[str]
                else:
                    field_type = str

                # Handle optional vs required fields
                if prop_name in required:
                    annotations[prop_name] = field_type
                    field_info[prop_name] = Field(description=prop_info.get("description", ""))
                else:
                    annotations[prop_name] = Optional[field_type]
                    field_info[prop_name] = Field(default=None, description=prop_info.get("description", ""))

            # Create dynamic model
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

        # Initialize with all required parameters
        super().__init__(
            name=tool_name,
            description=tool_description,
            args_schema=args_schema
        )

        # Set the mcp_client after initialization
        self.mcp_client = mcp_client

    def _run(self, **kwargs) -> str:
        """Run the tool synchronously"""
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        """Run the tool asynchronously"""
        try:
            # Filter out None values
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

            result = await self.mcp_client.call_tool(self.name, filtered_kwargs)

            # Handle the MCP response format
            if isinstance(result, dict):
                # Check if it's a list of content items
                if "content" in result:
                    content_items = result["content"]
                    if isinstance(content_items, list):
                        text_content = []
                        for item in content_items:
                            if item.get("type") == "text":
                                text_content.append(item.get("text", ""))
                        return "\n".join(text_content) if text_content else json.dumps(result, indent=2)

                return json.dumps(result, indent=2)
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Error calling {self.name}: {e}")
            return f"Error calling {self.name}: {str(e)}"

class MCPClient:
    """Client for communicating with MCP servers using stdio transport"""

    def __init__(self, server_command: List[str]):
        self.server_command = server_command
        self.process = None
        self.tools_info = {}
        self.request_id = 0

    async def start(self):
        """Start the MCP server process"""
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Wait a moment for the server to start
            await asyncio.sleep(2)

            # Initialize the server with proper MCP protocol
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

            # Send initialized notification
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
        """Send a request to the MCP server"""
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

        # Read response with timeout
        try:
            response_line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=30
            )
        except asyncio.TimeoutError:
            raise Exception("MCP server response timeout")

        if not response_line:
            raise Exception("MCP server closed connection")

        response_text = response_line.decode().strip()
        logger.debug(f"Received response: {response_text}")

        try:
            response = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {response_text}")

        if "error" in response:
            raise Exception(f"MCP Error: {response['error']}")

        return response.get("result", {})

    async def _send_notification(self, method: str, params: Optional[Dict] = None):
        """Send a notification to the MCP server"""
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
    """LangChain agent for browser automation using MCP tools"""

    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1"):
        self.lm_studio_url = lm_studio_url
        self.mcp_client = None
        self.agent_executor = None

    async def setup(self, mcp_server_script_path: str):
        """Setup the agent with MCP client and LM Studio connection"""

        try:
            # Initialize MCP client
            self.mcp_client = MCPClient([
                "python3", mcp_server_script_path
            ])
            await self.mcp_client.start()

            # Create LangChain tools from MCP tools
            tools = self.mcp_client.create_langchain_tools()

            if not tools:
                raise ValueError("No tools were successfully created from MCP server")

            logger.info(f"Created {len(tools)} tools from MCP server")

            # Setup LM Studio connection (using OpenAI-compatible API)
            llm = ChatOpenAI(
                base_url=self.lm_studio_url,
                api_key="not-needed",  # LM Studio doesn't require API key
                model="gemma",  # This should match your loaded model in LM Studio
                temperature=0.1,
                max_tokens=1000
            )

            # Test LM Studio connection
            try:
                test_response = await llm.ainvoke("Hello, can you respond with 'Connection successful'?")
                logger.info(f"LM Studio connection test: {test_response.content}")
            except Exception as e:
                logger.error(f"LM Studio connection test failed: {e}")
                raise ValueError(f"Cannot connect to LM Studio: {str(e)}. Make sure it's running on localhost:1234 with a model loaded.")

            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that can control a web browser using various tools.

Available browser automation tools:
- launch_browser: Launch a new browser instance
- navigate: Navigate to a URL
- take_screenshot: Take a screenshot of the page
- take_marked_screenshot: Take a screenshot with highlighted elements
- get_element_data: Get information about elements on the page
- click: Click on elements to interact with them
- input_text: Type text into input fields
- key_press: Press keyboard keys
- scroll: Scroll the page
- wait_for_element: Wait for elements to appear
- get_page_info: Get current page information
- close_browser: Close the browser

When helping with browser automation:
1. Always take screenshots to see what's on the page
2. Use get_element_data to inspect elements before interacting with them
3. Be specific about selectors (CSS selectors work best)
4. Wait for elements to load when needed
5. Provide clear feedback about what you're doing

Answer the user's request step by step, using the available tools as needed."""),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])

            # Create the agent
            try:
                agent = create_tool_calling_agent(llm, tools, prompt)
                self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                logger.info("Browser automation agent setup complete")
            except Exception as e:
                logger.error(f"Failed to create agent: {e}")
                raise ValueError(f"Failed to create LangChain agent: {str(e)}")

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            if self.mcp_client:
                await self.mcp_client.stop()
            raise

    async def run(self, query: str) -> str:
        """Run the agent with a query"""
        if not self.agent_executor:
            raise ValueError("Agent not setup. Call setup() first.")

        try:
            result = await self.agent_executor.ainvoke({"input": query})
            return result["output"]
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            return f"Error: {str(e)}"

    async def cleanup(self):
        """Cleanup resources"""
        if self.mcp_client:
            await self.mcp_client.stop()

async def main():
    """Main function to demonstrate the browser automation agent"""

    # Path to your MCP server script
    mcp_server_path = "playwright_mcp_server.py"  # Update this path

    # Initialize the agent
    agent = BrowserAutomationAgent()

    try:
        # Setup the agent
        print("🤖 Setting up browser automation agent...")
        await agent.setup(mcp_server_path)

        print("🤖 Browser Automation Agent is ready!")
        print("You can ask me to:")
        print("- Navigate to websites")
        print("- Take screenshots")
        print("- Click on elements")
        print("- Fill out forms")
        print("- Extract data from pages")
        print("- And much more!")
        print("\nType 'quit' to exit\n")

        # Interactive loop
        while True:
            try:
                user_input = input("\n👤 What would you like me to do? ")

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input.strip():
                    continue

                print("\n🤖 Working on it...")
                result = await agent.run(user_input)
                print(f"\n✅ Result: {result}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    except Exception as e:
        print(f"❌ Failed to setup agent: {str(e)}")
        print("\nMake sure:")
        print("1. LM Studio is running on localhost:1234")
        print("2. A model (like Gemma) is loaded in LM Studio")
        print("3. MCP server script path is correct")
        print("4. All required dependencies are installed")
        print("   - pip install langchain langchain-community playwright mcp")
        print("   - playwright install")

    finally:
        # Cleanup
        await agent.cleanup()
        print("\n👋 Goodbye!")

# Example usage functions
async def example_web_scraping():
    """Example: Web scraping workflow"""
    agent = BrowserAutomationAgent()
    await agent.setup("playwright_mcp_server.py")

    try:
        # Navigate to a website and extract data
        result = await agent.run("""
        1. Launch a browser
        2. Navigate to https://example.com
        3. Take a screenshot
        4. Get information about all links on the page
        5. Click on the first link (if any)
        """)
        print(result)

    finally:
        await agent.cleanup()

async def example_form_filling():
    """Example: Form filling workflow"""
    agent = BrowserAutomationAgent()
    await agent.setup("playwright_mcp_server.py")

    try:
        result = await agent.run("""
        1. Launch a browser
        2. Navigate to a website with a search form
        3. Find the search input field
        4. Type "playwright automation" in the search field
        5. Click the search button
        6. Take a screenshot of the results
        """)
        print(result)

    finally:
        await agent.cleanup()

if __name__ == "__main__":
    # Run the interactive agent
    asyncio.run(main())

    # Or run specific examples:
    # asyncio.run(example_web_scraping())
    # asyncio.run(example_form_filling())