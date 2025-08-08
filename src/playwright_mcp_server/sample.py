#!/usr/bin/env python3
"""
Playwright LangChain Tools - Browser automation tools for LangChain agents
Provides screenshot, element interaction, and navigation capabilities
"""

import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from pathlib import Path
import tempfile
import os

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Locator
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global browser state
playwright_instance = None
browser: Optional[Browser] = None
context: Optional[BrowserContext] = None
page: Optional[Page] = None

async def ensure_browser() -> Page:
    """Ensure browser, context, and page are initialized"""
    global playwright_instance, browser, context, page

    if not playwright_instance:
        playwright_instance = await async_playwright().start()

    if not browser:
        # Launch browser with useful options
        browser = await playwright_instance.chromium.launch(
            headless=False,  # Set to True for headless mode
            args=['--no-sandbox', '--disable-dev-shm-usage']
        )

    if not context:
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )

    if not page:
        page = await context.new_page()

    return page

# Input schemas for tools
class LaunchBrowserInput(BaseModel):
    headless: bool = Field(default=False, description="Run browser in headless mode")
    browser_type: str = Field(default='chromium', description="Browser type: chromium, firefox, or webkit")
    viewport_width: int = Field(default=1920, description="Viewport width")
    viewport_height: int = Field(default=1080, description="Viewport height")
    user_agent: Optional[str] = Field(default=None, description="Custom user agent string")
    extra_args: Optional[List[str]] = Field(default=None, description="Additional browser arguments")
    slow_mo: int = Field(default=0, description="Slow down operations by specified milliseconds")
    devtools: bool = Field(default=False, description="Open browser with devtools")

class NavigateInput(BaseModel):
    url: str = Field(description="URL to navigate to")

class TakeScreenshotInput(BaseModel):
    full_page: bool = Field(default=False, description="Take full page screenshot")
    element_selector: Optional[str] = Field(default=None, description="CSS selector for specific element")
    format: str = Field(default='png', description="Image format: png or jpeg")

class TakeMarkedScreenshotInput(BaseModel):
    selectors: List[str] = Field(description="List of CSS selectors to highlight")
    full_page: bool = Field(default=False, description="Take full page screenshot")
    format: str = Field(default='png', description="Image format: png or jpeg")

class GetElementDataInput(BaseModel):
    selector: str = Field(description="CSS selector to find elements")
    include_parent: bool = Field(default=True, description="Include parent element information")

class ClickInput(BaseModel):
    selector: str = Field(description="CSS selector for element to click")
    index: int = Field(default=0, description="Index of element if multiple match selector")
    button: str = Field(default='left', description="Mouse button: left, right, or middle")
    delay: int = Field(default=0, description="Delay in milliseconds before click")
    force: bool = Field(default=False, description="Force click even if element not actionable")

class InputTextInput(BaseModel):
    selector: str = Field(description="CSS selector for input element")
    text: str = Field(description="Text to input")
    index: int = Field(default=0, description="Index of element if multiple match selector")
    clear: bool = Field(default=True, description="Clear existing text before input")
    delay: int = Field(default=0, description="Delay between keystrokes in milliseconds")

class KeyPressInput(BaseModel):
    key: str = Field(description="Key to press (e.g., 'Enter', 'Escape', 'ArrowDown')")
    selector: Optional[str] = Field(default=None, description="CSS selector to focus before key press")
    index: int = Field(default=0, description="Index of element if multiple match selector")

class ScrollInput(BaseModel):
    direction: str = Field(default='down', description="Scroll direction: down, up, left, right, page_down, page_up, home, end")
    amount: int = Field(default=3, description="Number of scroll actions")
    selector: Optional[str] = Field(default=None, description="CSS selector for element to scroll")
    index: int = Field(default=0, description="Index of element if multiple match selector")

class TapInput(BaseModel):
    selector: str = Field(description="CSS selector for element to tap")
    index: int = Field(default=0, description="Index of element if multiple match selector")
    force: bool = Field(default=False, description="Force tap even if element not actionable")

class WaitForElementInput(BaseModel):
    selector: str = Field(description="CSS selector for element to wait for")
    state: str = Field(default='visible', description="State to wait for: visible, hidden, attached, detached")
    timeout: int = Field(default=5000, description="Timeout in milliseconds")

class NewPageInput(BaseModel):
    url: Optional[str] = Field(default=None, description="URL to navigate to in new page")

# Tool classes
class LaunchBrowserTool(BaseTool):
    name: str = "launch_browser"
    description: str = "Launch browser with custom configuration"
    args_schema: Type[BaseModel] = LaunchBrowserInput

    def _run(self, **kwargs) -> str:
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        global playwright_instance, browser, context, page

        try:
            # Close existing browser if running
            if browser:
                await self._close_browser()

            # Start playwright
            if not playwright_instance:
                playwright_instance = await async_playwright().start()

            # Prepare browser args
            browser_args = ['--no-sandbox', '--disable-dev-shm-usage']
            if kwargs.get('extra_args'):
                browser_args.extend(kwargs['extra_args'])

            # Launch browser based on type
            browser_type = kwargs.get('browser_type', 'chromium')
            headless = kwargs.get('headless', False)
            slow_mo = kwargs.get('slow_mo', 0)
            devtools = kwargs.get('devtools', False)

            if browser_type.lower() == 'chromium':
                browser = await playwright_instance.chromium.launch(
                    headless=headless,
                    args=browser_args,
                    slow_mo=slow_mo,
                    devtools=devtools
                )
            elif browser_type.lower() == 'firefox':
                browser = await playwright_instance.firefox.launch(
                    headless=headless,
                    args=browser_args,
                    slow_mo=slow_mo,
                    devtools=devtools
                )
            elif browser_type.lower() == 'webkit':
                browser = await playwright_instance.webkit.launch(
                    headless=headless,
                    args=browser_args,
                    slow_mo=slow_mo,
                    devtools=devtools
                )
            else:
                raise ValueError(f"Unsupported browser type: {browser_type}")

            # Set up default user agent if not provided
            user_agent = kwargs.get('user_agent') or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

            # Create context
            context = await browser.new_context(
                viewport={'width': kwargs.get('viewport_width', 1920), 'height': kwargs.get('viewport_height', 1080)},
                user_agent=user_agent
            )

            # Create page
            page = await context.new_page()

            result = {
                'success': True,
                'browser_type': browser_type,
                'headless': headless,
                'viewport': {'width': kwargs.get('viewport_width', 1920), 'height': kwargs.get('viewport_height', 1080)},
                'user_agent': user_agent,
                'slow_mo': slow_mo,
                'devtools': devtools,
                'message': f'{browser_type.title()} browser launched successfully'
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

    async def _close_browser(self):
        global playwright_instance, browser, context, page
        if page:
            await page.close()
            page = None
        if context:
            await context.close()
            context = None
        if browser:
            await browser.close()
            browser = None

class GetBrowserStatusTool(BaseTool):
    name: str = "get_browser_status"
    description: str = "Check current browser status"
    args_schema: Type[BaseModel] = BaseModel

    def _run(self) -> str:
        return asyncio.run(self._arun())

    async def _arun(self) -> str:
        global playwright_instance, browser, context, page

        try:
            status = {
                'playwright_initialized': playwright_instance is not None,
                'browser_launched': browser is not None,
                'context_created': context is not None,
                'page_available': page is not None,
                'current_url': page.url if page else None,
                'current_title': await page.title() if page else None
            }

            if browser:
                status['browser_type'] = browser.__class__.__name__.replace('Browser', '').lower()

            result = {
                'success': True,
                'status': status
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class NewPageTool(BaseTool):
    name: str = "new_page"
    description: str = "Create a new page/tab in the current browser context"
    args_schema: Type[BaseModel] = NewPageInput

    def _run(self, url: Optional[str] = None) -> str:
        return asyncio.run(self._arun(url=url))

    async def _arun(self, url: Optional[str] = None) -> str:
        global context, page

        try:
            if not context:
                # If no context exists, use ensure_browser to create everything
                await ensure_browser()
            else:
                # Create new page in existing context
                page = await context.new_page()

            if url:
                await page.goto(url, wait_until='domcontentloaded')

            result = {
                'success': True,
                'url': page.url if url else 'about:blank',
                'title': await page.title(),
                'message': 'New page created successfully'
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class NavigateTool(BaseTool):
    name: str = "navigate"
    description: str = "Navigate to a URL"
    args_schema: Type[BaseModel] = NavigateInput

    def _run(self, url: str) -> str:
        return asyncio.run(self._arun(url=url))

    async def _arun(self, url: str) -> str:
        try:
            current_page = await ensure_browser()
            response = await current_page.goto(url, wait_until='domcontentloaded')

            result = {
                'success': True,
                'url': current_page.url,
                'title': await current_page.title(),
                'status': response.status if response else None
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class TakeScreenshotTool(BaseTool):
    name: str = "take_screenshot"
    description: str = "Take a screenshot of the page or specific element"
    args_schema: Type[BaseModel] = TakeScreenshotInput

    def _run(self, full_page: bool = False, element_selector: Optional[str] = None, format: str = 'png') -> str:
        return asyncio.run(self._arun(full_page=full_page, element_selector=element_selector, format=format))

    async def _arun(self, full_page: bool = False, element_selector: Optional[str] = None, format: str = 'png') -> str:
        try:
            current_page = await ensure_browser()

            screenshot_options = {
                'type': format,
                'full_page': full_page
            }

            if element_selector:
                element = current_page.locator(element_selector)
                await element.wait_for(state='visible', timeout=5000)
                screenshot_bytes = await element.screenshot(**screenshot_options)
            else:
                screenshot_bytes = await current_page.screenshot(**screenshot_options)

            # Save screenshot to temporary file for easier handling
            with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as tmp_file:
                tmp_file.write(screenshot_bytes)
                screenshot_path = tmp_file.name

            result = {
                'success': True,
                'format': format,
                'element_selector': element_selector,
                'full_page': full_page,
                'screenshot_path': screenshot_path,
                'message': f'Screenshot saved to {screenshot_path}'
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class TakeMarkedScreenshotTool(BaseTool):
    name: str = "take_marked_screenshot"
    description: str = "Take a screenshot with elements marked/highlighted"
    args_schema: Type[BaseModel] = TakeMarkedScreenshotInput

    def _run(self, selectors: List[str], full_page: bool = False, format: str = 'png') -> str:
        return asyncio.run(self._arun(selectors=selectors, full_page=full_page, format=format))

    async def _arun(self, selectors: List[str], full_page: bool = False, format: str = 'png') -> str:
        try:
            current_page = await ensure_browser()

            # Add highlighting styles
            highlight_script = """
            (selectors) => {
                const style = document.createElement('style');
                style.innerHTML = `
                    .mcp-highlight {
                        outline: 3px solid red !important;
                        outline-offset: 2px !important;
                        background-color: rgba(255, 0, 0, 0.1) !important;
                    }
                `;
                document.head.appendChild(style);

                const marked_elements = [];
                selectors.forEach((selector, index) => {
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(el => {
                        el.classList.add('mcp-highlight');
                        el.setAttribute('data-mcp-index', index.toString());
                        marked_elements.push({
                            selector: selector,
                            index: index,
                            tag: el.tagName.toLowerCase(),
                            text: el.innerText?.substring(0, 100) || '',
                            rect: el.getBoundingClientRect()
                        });
                    });
                });

                return marked_elements;
            }
            """

            marked_elements = await current_page.evaluate(highlight_script, selectors)

            # Take screenshot
            screenshot_bytes = await current_page.screenshot(
                type=format,
                full_page=full_page
            )

            # Remove highlighting
            await current_page.evaluate("""
            () => {
                document.querySelectorAll('.mcp-highlight').forEach(el => {
                    el.classList.remove('mcp-highlight');
                    el.removeAttribute('data-mcp-index');
                });
                const style = document.querySelector('style');
                if (style && style.innerHTML.includes('mcp-highlight')) {
                    style.remove();
                }
            }
            """)

            # Save screenshot to temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{format}', delete=False) as tmp_file:
                tmp_file.write(screenshot_bytes)
                screenshot_path = tmp_file.name

            result = {
                'success': True,
                'format': format,
                'marked_elements': marked_elements,
                'selectors': selectors,
                'full_page': full_page,
                'screenshot_path': screenshot_path,
                'message': f'Marked screenshot saved to {screenshot_path}'
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class GetElementDataTool(BaseTool):
    name: str = "get_element_data"
    description: str = "Get data about elements matching the selector, including parent information"
    args_schema: Type[BaseModel] = GetElementDataInput

    def _run(self, selector: str, include_parent: bool = True) -> str:
        return asyncio.run(self._arun(selector=selector, include_parent=include_parent))

    async def _arun(self, selector: str, include_parent: bool = True) -> str:
        try:
            current_page = await ensure_browser()

            element_data_script = """
            (params) => {
                const { selector, includeParent } = params;
                const elements = document.querySelectorAll(selector);
                return Array.from(elements).map((el, index) => {
                    const rect = el.getBoundingClientRect();

                    // Get parent information if requested
                    let parentInfo = null;
                    if (includeParent && el.parentElement) {
                        const parent = el.parentElement;
                        const parentRect = parent.getBoundingClientRect();
                        parentInfo = {
                            tag: parent.tagName.toLowerCase(),
                            text: parent.innerText?.substring(0, 200) || '',
                            id: parent.id || '',
                            className: parent.className || '',
                            attributes: Array.from(parent.attributes).reduce((acc, attr) => {
                                acc[attr.name] = attr.value;
                                return acc;
                            }, {}),
                            rect: {
                                x: parentRect.x,
                                y: parentRect.y,
                                width: parentRect.width,
                                height: parentRect.height,
                                top: parentRect.top,
                                right: parentRect.right,
                                bottom: parentRect.bottom,
                                left: parentRect.left
                            },
                            visible: parentRect.width > 0 && parentRect.height > 0 &&
                                    window.getComputedStyle(parent).visibility !== 'hidden' &&
                                    window.getComputedStyle(parent).display !== 'none'
                        };
                    }

                    return {
                        index: index,
                        tag: el.tagName.toLowerCase(),
                        text: el.innerText?.substring(0, 200) || '',
                        value: el.value || '',
                        href: el.href || '',
                        src: el.src || '',
                        id: el.id || '',
                        className: el.className || '',
                        attributes: Array.from(el.attributes).reduce((acc, attr) => {
                            acc[attr.name] = attr.value;
                            return acc;
                        }, {}),
                        rect: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height,
                            top: rect.top,
                            right: rect.right,
                            bottom: rect.bottom,
                            left: rect.left
                        },
                        visible: rect.width > 0 && rect.height > 0 &&
                                window.getComputedStyle(el).visibility !== 'hidden' &&
                                window.getComputedStyle(el).display !== 'none',
                        parent: parentInfo
                    };
                });
            }
            """

            elements = await current_page.evaluate(element_data_script, {"selector": selector, "includeParent": include_parent})

            result = {
                'success': True,
                'selector': selector,
                'count': len(elements),
                'elements': elements,
                'includes_parent_info': include_parent
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class ClickTool(BaseTool):
    name: str = "click"
    description: str = "Click on an element"
    args_schema: Type[BaseModel] = ClickInput

    def _run(self, selector: str, index: int = 0, button: str = 'left', delay: int = 0, force: bool = False) -> str:
        return asyncio.run(self._arun(selector=selector, index=index, button=button, delay=delay, force=force))

    async def _arun(self, selector: str, index: int = 0, button: str = 'left', delay: int = 0, force: bool = False) -> str:
        try:
            current_page = await ensure_browser()
            elements = current_page.locator(selector)
            element = elements.nth(index)

            await element.wait_for(state='visible', timeout=5000)

            click_options = {
                'button': button,
                'delay': delay,
                'force': force
            }

            await element.click(**click_options)

            result = {
                'success': True,
                'selector': selector,
                'index': index,
                'button': button
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class InputTextTool(BaseTool):
    name: str = "input_text"
    description: str = "Input text into an element"
    args_schema: Type[BaseModel] = InputTextInput

    def _run(self, selector: str, text: str, index: int = 0, clear: bool = True, delay: int = 0) -> str:
        return asyncio.run(self._arun(selector=selector, text=text, index=index, clear=clear, delay=delay))

    async def _arun(self, selector: str, text: str, index: int = 0, clear: bool = True, delay: int = 0) -> str:
        try:
            current_page = await ensure_browser()
            elements = current_page.locator(selector)
            element = elements.nth(index)

            await element.wait_for(state='visible', timeout=5000)

            if clear:
                await element.clear()

            if delay > 0:
                await element.type(text, delay=delay)
            else:
                await element.fill(text)

            result = {
                'success': True,
                'selector': selector,
                'index': index,
                'text': text,
                'cleared': clear
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class KeyPressTool(BaseTool):
    name: str = "key_press"
    description: str = "Press a key, optionally on a specific element"
    args_schema: Type[BaseModel] = KeyPressInput

    def _run(self, key: str, selector: Optional[str] = None, index: int = 0) -> str:
        return asyncio.run(self._arun(key=key, selector=selector, index=index))

    async def _arun(self, key: str, selector: Optional[str] = None, index: int = 0) -> str:
        try:
            current_page = await ensure_browser()

            if selector:
                elements = current_page.locator(selector)
                element = elements.nth(index)
                await element.wait_for(state='visible', timeout=5000)
                await element.press(key)
            else:
                await current_page.keyboard.press(key)

            result = {
                'success': True,
                'key': key,
                'selector': selector,
                'index': index
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class ScrollTool(BaseTool):
    name: str = "scroll"
    description: str = "Scroll the page or a specific element"
    args_schema: Type[BaseModel] = ScrollInput

    def _run(self, direction: str = 'down', amount: int = 3, selector: Optional[str] = None, index: int = 0) -> str:
        return asyncio.run(self._arun(direction=direction, amount=amount, selector=selector, index=index))

    async def _arun(self, direction: str = 'down', amount: int = 3, selector: Optional[str] = None, index: int = 0) -> str:
        try:
            current_page = await ensure_browser()

            scroll_directions = {
                'down': 'ArrowDown',
                'up': 'ArrowUp',
                'left': 'ArrowLeft',
                'right': 'ArrowRight',
                'page_down': 'PageDown',
                'page_up': 'PageUp',
                'home': 'Home',
                'end': 'End'
            }

            key = scroll_directions.get(direction, 'ArrowDown')

            if selector:
                elements = current_page.locator(selector)
                element = elements.nth(index)
                await element.wait_for(state='visible', timeout=5000)
                await element.focus()
                for _ in range(amount):
                    await element.press(key)
            else:
                for _ in range(amount):
                    await current_page.keyboard.press(key)

            result = {
                'success': True,
                'direction': direction,
                'amount': amount,
                'selector': selector,
                'index': index
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class TapTool(BaseTool):
    name: str = "tap"
    description: str = "Tap on an element (mobile-style interaction)"
    args_schema: Type[BaseModel] = TapInput

    def _run(self, selector: str, index: int = 0, force: bool = False) -> str:
        return asyncio.run(self._arun(selector=selector, index=index, force=force))

    async def _arun(self, selector: str, index: int = 0, force: bool = False) -> str:
        try:
            current_page = await ensure_browser()
            elements = current_page.locator(selector)
            element = elements.nth(index)

            await element.wait_for(state='visible', timeout=5000)
            await element.tap(force=force)

            result = {
                'success': True,
                'selector': selector,
                'index': index,
                'force': force
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class WaitForElementTool(BaseTool):
    name: str = "wait_for_element"
    description: str = "Wait for an element to reach a specific state"
    args_schema: Type[BaseModel] = WaitForElementInput

    def _run(self, selector: str, state: str = 'visible', timeout: int = 5000) -> str:
        return asyncio.run(self._arun(selector=selector, state=state, timeout=timeout))

    async def _arun(self, selector: str, state: str = 'visible', timeout: int = 5000) -> str:
        try:
            current_page = await ensure_browser()
            element = current_page.locator(selector)

            await element.wait_for(state=state, timeout=timeout)

            result = {
                'success': True,
                'selector': selector,
                'state': state,
                'timeout': timeout
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class GetPageInfoTool(BaseTool):
    name: str = "get_page_info"
    description: str = "Get current page information"
    args_schema: Type[BaseModel] = BaseModel

    def _run(self) -> str:
        return asyncio.run(self._arun())

    async def _arun(self) -> str:
        try:
            current_page = await ensure_browser()

            result = {
                'success': True,
                'url': current_page.url,
                'title': await current_page.title(),
                'viewport': current_page.viewport_size,
                'user_agent': await current_page.evaluate('navigator.userAgent')
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

class CloseBrowserTool(BaseTool):
    name: str = "close_browser"
    description: str = "Close the browser and clean up resources"
    args_schema: Type[BaseModel] = BaseModel

    def _run(self) -> str:
        return asyncio.run(self._arun())

    async def _arun(self) -> str:
        global playwright_instance, browser, context, page

        try:
            if page:
                await page.close()
                page = None

            if context:
                await context.close()
                context = None

            if browser:
                await browser.close()
                browser = None

            if playwright_instance:
                await playwright_instance.stop()
                playwright_instance = None

            result = {'success': True, 'message': 'Browser closed successfully'}
            return json.dumps(result, indent=2)
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            return json.dumps(error_result, indent=2)

# Convenience function to get all tools
def get_playwright_tools() -> List[BaseTool]:
    """Get all Playwright tools for use with LangChain agents"""
    return [
        LaunchBrowserTool(),
        GetBrowserStatusTool(),
        NewPageTool(),
        NavigateTool(),
        TakeScreenshotTool(),
        TakeMarkedScreenshotTool(),
        GetElementDataTool(),
        ClickTool(),
        InputTextTool(),
        KeyPressTool(),
        ScrollTool(),
        TapTool(),
        WaitForElementTool(),
        GetPageInfoTool(),
        CloseBrowserTool()
    ]

# Example usage with LangChain agent
if __name__ == "__main__":
    # Example of how to use these tools with a LangChain agent
    from langchain.agents import initialize_agent, AgentType
    from langchain_openai import ChatOpenAI

    # Initialize LLM (you'll need to set up your API key)
    llm = ChatOpenAI(temperature=0)

    # Get all Playwright tools
    tools = get_playwright_tools()

    # Initialize agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Example usage
    try:
        # Launch browser and navigate to a website
        response = agent.run("Launch a browser, navigate to https://example.com, and take a screenshot")
        print(response)
    finally:
        # Clean up
        close_tool = CloseBrowserTool()
        close_tool.run()