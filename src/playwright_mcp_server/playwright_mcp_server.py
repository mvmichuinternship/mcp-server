#!/usr/bin/env python3
"""
Playwright MCP Server - Browser automation tools using Official MCP SDK
Provides screenshot, element interaction, and navigation capabilities
"""

import asyncio
import base64
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import tempfile
import os

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Locator
from mcp.server.fastmcp import FastMCP
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import mcp.server.stdio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global browser state
playwright_instance = None
browser: Optional[Browser] = None
context: Optional[BrowserContext] = None
page: Optional[Page] = None

# Initialize MCP Server
mcp = FastMCP("playwright-automation-server")

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

@mcp.tool()
async def launch_browser(
    headless: bool = False,
    browser_type: str = 'chromium',
    viewport_width: int = 1920,
    viewport_height: int = 1080,
    user_agent: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    slow_mo: int = 0,
    devtools: bool = False
) -> List[TextContent]:
    """Launch browser with custom configuration"""
    global playwright_instance, browser, context, page

    try:
        # Close existing browser if running
        if browser:
            await close_browser()

        # Start playwright
        if not playwright_instance:
            playwright_instance = await async_playwright().start()

        # Prepare browser args
        browser_args = ['--no-sandbox', '--disable-dev-shm-usage']
        if extra_args:
            browser_args.extend(extra_args)

        # Launch browser based on type
        if browser_type in ['chrome', 'google-chrome', 'chromium']:
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
        if not user_agent:
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

        # Create context
        context = await browser.new_context(
            viewport={'width': viewport_width, 'height': viewport_height},
            user_agent=user_agent
        )

        # Create page
        page = await context.new_page()

        result = {
            'success': True,
            'browser_type': browser_type,
            'headless': headless,
            'viewport': {'width': viewport_width, 'height': viewport_height},
            'user_agent': user_agent,
            'slow_mo': slow_mo,
            'devtools': devtools,
            'message': f'{browser_type.title()} browser launched successfully'
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def get_browser_status() -> List[TextContent]:
    """Check current browser status"""
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

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def new_page(url: Optional[str] = None) -> List[TextContent]:
    """Create a new page/tab in the current browser context"""
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

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def navigate(url: str) -> List[TextContent]:
    """Navigate to a URL"""
    try:
        current_page = await ensure_browser()
        response = await current_page.goto(url, wait_until='domcontentloaded')

        result = {
            'success': True,
            'url': current_page.url,
            'title': await current_page.title(),
            'status': response.status if response else None
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def take_screenshot(
    full_page: bool = False,
    element_selector: Optional[str] = None,
    format: str = 'png'
) -> List[Union[TextContent, ImageContent]]:
    """Take a screenshot of the page or specific element"""
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

        # Convert to base64 for transport
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        result = {
            'success': True,
            'format': format,
            'element_selector': element_selector,
            'full_page': full_page
        }

        return [
            TextContent(type="text", text=json.dumps(result, indent=2)),
            ImageContent(
                type="image",
                data=screenshot_b64,
                mimeType=f"image/{format}"
            )
        ]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def take_marked_screenshot(
    selectors: List[str],
    full_page: bool = False,
    format: str = 'png'
) -> List[Union[TextContent, ImageContent]]:
    """Take a screenshot with elements marked/highlighted"""
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

        screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')

        result = {
            'success': True,
            'format': format,
            'marked_elements': marked_elements,
            'selectors': selectors,
            'full_page': full_page
        }

        return [
            TextContent(type="text", text=json.dumps(result, indent=2)),
            ImageContent(
                type="image",
                data=screenshot_b64,
                mimeType=f"image/{format}"
            )
        ]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

# @mcp.tool()
# async def get_element_data(selector: str) -> List[TextContent]:
#     """Get data about elements matching the selector"""
#     try:
#         current_page = await ensure_browser()

#         element_data_script = """
#         (selector) => {
#             const elements = document.querySelectorAll(selector);
#             return Array.from(elements).map((el, index) => {
#                 const rect = el.getBoundingClientRect();
#                 return {
#                     index: index,
#                     tag: el.tagName.toLowerCase(),
#                     text: el.innerText?.substring(0, 200) || '',
#                     value: el.value || '',
#                     href: el.href || '',
#                     src: el.src || '',
#                     id: el.id || '',
#                     className: el.className || '',
#                     attributes: Array.from(el.attributes).reduce((acc, attr) => {
#                         acc[attr.name] = attr.value;
#                         return acc;
#                     }, {}),
#                     rect: {
#                         x: rect.x,
#                         y: rect.y,
#                         width: rect.width,
#                         height: rect.height,
#                         top: rect.top,
#                         right: rect.right,
#                         bottom: rect.bottom,
#                         left: rect.left
#                     },
#                     visible: rect.width > 0 && rect.height > 0 &&
#                             window.getComputedStyle(el).visibility !== 'hidden' &&
#                             window.getComputedStyle(el).display !== 'none'
#                 };
#             });
#         }
#         """

#         elements = await current_page.evaluate(element_data_script, selector)

#         result = {
#             'success': True,
#             'selector': selector,
#             'count': len(elements),
#             'elements': elements
#         }
#         return [TextContent(type="text", text=json.dumps(result, indent=2))]
#     except Exception as e:
#         error_result = {'success': False, 'error': str(e)}
#         return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def get_element_data(selector: str, include_parent: bool = True) -> List[TextContent]:
    """Get data about elements matching the selector, including parent information"""
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
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def click(
    selector: str,
    index: int = 0,
    button: str = 'left',
    delay: int = 0,
    force: bool = False
) -> List[TextContent]:
    """Click on an element"""
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
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def input_text(
    selector: str,
    text: str,
    index: int = 0,
    clear: bool = True,
    delay: int = 0
) -> List[TextContent]:
    """Input text into an element"""
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
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def key_press(
    key: str,
    selector: Optional[str] = None,
    index: int = 0
) -> List[TextContent]:
    """Press a key, optionally on a specific element"""
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
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def scroll(
    direction: str = 'down',
    amount: int = 3,
    selector: Optional[str] = None,
    index: int = 0
) -> List[TextContent]:
    """Scroll the page or a specific element"""
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
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def tap(
    selector: str,
    index: int = 0,
    force: bool = False
) -> List[TextContent]:
    """Tap on an element (mobile-style interaction)"""
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
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def wait_for_element(
    selector: str,
    state: str = 'visible',
    timeout: int = 5000
) -> List[TextContent]:
    """Wait for an element to reach a specific state"""
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
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def get_page_info() -> List[TextContent]:
    """Get current page information"""
    try:
        current_page = await ensure_browser()

        result = {
            'success': True,
            'url': current_page.url,
            'title': await current_page.title(),
            'viewport': current_page.viewport_size,
            'user_agent': await current_page.evaluate('navigator.userAgent')
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def find_elements_by_text(
    text: str,
    tag: str = None,
    partial: bool = True
) -> List[TextContent]:
    """Find elements by their text content"""
    try:
        current_page = await ensure_browser()

        script = """
        (params) => {
            const { text, tag, partial } = params;
            let elements;

            if (tag) {
                elements = document.querySelectorAll(tag);
            } else {
                elements = document.querySelectorAll('*');
            }

            const matches = [];
            Array.from(elements).forEach((el, index) => {
                const elementText = el.innerText || el.textContent || '';
                const matches_text = partial ?
                    elementText.toLowerCase().includes(text.toLowerCase()) :
                    elementText.toLowerCase() === text.toLowerCase();

                if (matches_text && elementText.trim() !== '') {
                    const rect = el.getBoundingClientRect();
                    matches.push({
                        index: index,
                        tag: el.tagName.toLowerCase(),
                        text: elementText.substring(0, 200),
                        id: el.id || '',
                        className: el.className || '',
                        selector: el.id ? `#${el.id}` :
                                 el.className ? `.${el.className.split(' ')[0]}` :
                                 el.tagName.toLowerCase(),
                        rect: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        },
                        visible: rect.width > 0 && rect.height > 0 &&
                                window.getComputedStyle(el).visibility !== 'hidden' &&
                                window.getComputedStyle(el).display !== 'none'
                    });
                }
            });

            return matches.slice(0, 20); // Limit to first 20 matches
        }
        """

        matches = await current_page.evaluate(script, {
            "text": text,
            "tag": tag,
            "partial": partial
        })

        result = {
            'success': True,
            'search_text': text,
            'tag_filter': tag,
            'partial_match': partial,
            'matches_found': len(matches),
            'elements': matches
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def find_input_fields() -> List[TextContent]:
    """Find all input fields on the page"""
    try:
        current_page = await ensure_browser()

        script = """
        () => {
            const inputs = document.querySelectorAll('input, textarea, select, [contenteditable]');
            return Array.from(inputs).map((el, index) => {
                const rect = el.getBoundingClientRect();
                return {
                    index: index,
                    tag: el.tagName.toLowerCase(),
                    type: el.type || 'text',
                    placeholder: el.placeholder || '',
                    name: el.name || '',
                    id: el.id || '',
                    className: el.className || '',
                    value: el.value || '',
                    selector: el.id ? `#${el.id}` :
                             el.name ? `[name="${el.name}"]` :
                             el.className ? `.${el.className.split(' ')[0]}` :
                             `${el.tagName.toLowerCase()}`,
                    rect: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    },
                    visible: rect.width > 0 && rect.height > 0 &&
                            window.getComputedStyle(el).visibility !== 'hidden' &&
                            window.getComputedStyle(el).display !== 'none',
                    aria_label: el.getAttribute('aria-label') || '',
                    title: el.title || ''
                };
            });
        }
        """

        inputs = await current_page.evaluate(script)

        result = {
            'success': True,
            'input_fields_found': len(inputs),
            'elements': inputs
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]

@mcp.tool()
async def smart_click(
    text: str = None,
    selector: str = None,
    index: int = 0,
    button: str = 'left',
    timeout: int = 10000
) -> List[TextContent]:
    """Smart click that can find elements by text or selector with better error handling"""
    try:
        current_page = await ensure_browser()

        element = None
        used_selector = selector

        if text and not selector:
            # Find by text
            script = """
            (text) => {
                const elements = document.querySelectorAll('*');
                for (const el of elements) {
                    if (el.innerText && el.innerText.toLowerCase().includes(text.toLowerCase())) {
                        // Prefer clickable elements
                        if (el.tagName.toLowerCase() in ['button', 'a', 'input'] ||
                            el.onclick ||
                            el.getAttribute('role') === 'button' ||
                            el.style.cursor === 'pointer') {
                            return el;
                        }
                    }
                }
                // Fallback to any element with the text
                for (const el of elements) {
                    if (el.innerText && el.innerText.toLowerCase().includes(text.toLowerCase())) {
                        return el;
                    }
                }
                return null;
            }
            """

            target_element = await current_page.evaluate(script, text)
            if target_element:
                # Create a locator for the found element
                element = current_page.locator(f'text="{text}"').first
                used_selector = f'text="{text}"'
            else:
                return [TextContent(type="text", text=json.dumps({
                    'success': False,
                    'error': f'No element found containing text: {text}'
                }, indent=2))]

        elif selector:
            element = current_page.locator(selector).nth(index)

        if not element:
            return [TextContent(type="text", text=json.dumps({
                'success': False,
                'error': 'No element specified. Provide either text or selector.'
            }, indent=2))]

        # Wait for element and click with better error handling
        try:
            await element.wait_for(state='visible', timeout=timeout)
            await element.click(button=button)

            result = {
                'success': True,
                'selector': used_selector,
                'text': text,
                'index': index,
                'button': button,
                'method': 'text_search' if text and not selector else 'selector'
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as click_error:
            # Try to provide helpful debugging info
            page_title = await current_page.title()
            page_url = current_page.url

            # Check if element exists but is not visible/clickable
            element_count = await element.count()

            debug_info = {
                'success': False,
                'error': str(click_error),
                'debug': {
                    'page_title': page_title,
                    'page_url': page_url,
                    'selector': used_selector,
                    'element_count': element_count,
                    'timeout_used': timeout
                }
            }

            return [TextContent(type="text", text=json.dumps(debug_info, indent=2))]

    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


@mcp.tool()
async def close_browser() -> List[TextContent]:
    """Close the browser and clean up resources"""
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
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_result = {'success': False, 'error': str(e)}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]



if __name__ == "__main__":
    # Run the FastMCP server
    asyncio.run(mcp.run())