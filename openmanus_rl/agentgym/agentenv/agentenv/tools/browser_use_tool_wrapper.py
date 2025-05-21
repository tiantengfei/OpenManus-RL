# openmanus_rl/agentgym/agentenv/agentenv/tools/browser_use_tool_wrapper.py
import asyncio
import json
from typing import Optional, Dict, Any, Literal

from .base import BaseTool

# Attempt to import browser_use. If not available, the tool will fail at runtime if used.
try:
    from browser_use import Browser, BrowserConfig, BrowserContext
    # It seems BrowserContext might not be directly exposed or needed for basic usage with 'with browser.new_context()'.
    # The original OpenManus tool used BrowserContext explicitly.
    # For this wrapper, we might simplify if direct context management isn't immediately necessary for basic actions.
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    # Define dummy classes if browser_use is not installed, so the file can be imported.
    class Browser: pass
    class BrowserConfig: pass
    print("[BrowserUseToolWrapper] WARNING: `browser-use` library not found. This tool will not be functional.")

# Custom Exception for Tool Errors
class ToolError(Exception):
    pass

_BROWSER_DESCRIPTION = """
A tool to interact with web pages.
Supported actions:
- 'go_to_url': Navigates to a given URL.
- 'get_page_content': Returns the text content of the current page.
- 'scroll_page': Scrolls the current page up or down.
"""

class BrowserUseToolWrapper(BaseTool):
    name: str = "web_browser" # Renamed for clarity from generic "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["go_to_url", "get_page_content", "scroll_page"],
                "description": "The browser action to perform.",
            },
            "url": {"type": "string", "description": "URL for 'go_to_url' action."},
            "scroll_direction": {
                "type": "string",
                "enum": ["up", "down"],
                "description": "Direction for 'scroll_page' action.",
                "default": "down",
            },
            "scroll_amount_pixels": { # Added for clarity, original used scroll_amount
                "type": "integer",
                "description": "Pixels to scroll for 'scroll_page' action. Default is viewport height.",
                "default": 0, # 0 could mean full page scroll, or use a sensible default like 500
            },
        },
        "required": ["action"],
    }

    def __init__(self, browser_config_args: Optional[Dict[str, Any]] = None):
        super().__init__()
        if not BROWSER_USE_AVAILABLE:
            # This instance will not be functional.
            # Attempts to use it will likely raise further errors if methods are called.
            self._browser: Optional[Browser] = None
            self._context: Optional[Any] = None # browser_use.BrowserContext type if available
            print("[BrowserUseToolWrapper] Instance created, but `browser-use` lib is missing. Tool is non-functional.")
            return

        # Default config: Run headless. For non-headless, this needs to be configurable.
        # Original OpenManus had complex config loading. Here, simplify for initial port.
        config_values = browser_config_args if browser_config_args else {}
        # Sensible defaults if not provided
        config_values.setdefault("headless", True) 
        config_values.setdefault("disable_security", True) # From original, for broader compatibility

        self._browser_config = BrowserConfig(**config_values)
        self._browser: Optional[Browser] = None
        self._context: Optional[Any] = None # Placeholder for browser_use.BrowserContext
        self._current_page_object: Optional[Any] = None # Placeholder for browser_use.Page
        self._lock = asyncio.Lock() # To protect shared browser resources if needed, though each agent might get its own instance.
        print(f"[BrowserUseToolWrapper] Initialized with config: {config_values}")

    async def _ensure_browser_initialized(self):
        if not BROWSER_USE_AVAILABLE:
            raise ToolError("`browser-use` library is not available. Cannot operate browser.")
        
        async with self._lock: # Ensure only one coroutine initializes the browser
            if self._browser is None:
                print("[BrowserUseToolWrapper] Initializing new browser instance...")
                self._browser = Browser(config=self._browser_config)
                # The original tool created a persistent context.
                # For simplicity, we might create context per session or ensure it's persistent.
                # Let's try persistent context like the original.
                self._context = await self._browser.new_context()
                print("[BrowserUseToolWrapper] Browser and context initialized.")

            if self._context and not self._current_page_object: # Ensure there's always a page
                 # Get current page or create one if none (e.g. after context creation)
                if self._context.pages:
                    self._current_page_object = self._context.pages[0]
                else:
                    self._current_page_object = await self._context.new_page()
                print("[BrowserUseToolWrapper] Ensured current page object exists.")


    async def execute(
        self,
        action: Literal["go_to_url", "get_page_content", "scroll_page"],
        url: Optional[str] = None,
        scroll_direction: Literal["up", "down"] = "down",
        scroll_amount_pixels: int = 0, # 0 might mean a full page, or use a default like 500/viewport height
        **kwargs: Any,
    ) -> str:
        if not BROWSER_USE_AVAILABLE:
            return "Error: `browser-use` library is not installed or available."

        await self._ensure_browser_initialized()
        if not self._browser or not self._context or not self._current_page_object:
            return "Error: Browser could not be initialized."

        page = self._current_page_object

        try:
            if action == "go_to_url":
                if not url:
                    return "Error: URL is required for 'go_to_url' action."
                print(f"[BrowserUseToolWrapper] Navigating to URL: {url}")
                await page.goto(url, wait_until="load") # 'load' or 'domcontentloaded' or 'networkidle'
                return f"Successfully navigated to {url}. Current page title: {await page.title()}"

            elif action == "get_page_content":
                print("[BrowserUseToolWrapper] Getting page content...")
                # For simplicity, return text content. Original used markdownify.
                # Consider adding markdownify if HTML structure is important.
                content = await page.content() # HTML content
                # A very basic text extraction:
                # More sophisticated text extraction might be needed.
                # For now, let's return a snippet of HTML to show it works.
                # In a real scenario, use libraries like beautifulsoup4 or readability-lxml for cleaner text.
                # Or, if markdownify is added as a dependency:
                # import markdownify
                # text_content = markdownify.markdownify(content)
                text_content = content[:2000] # Truncate for now
                return f"Current page content (first 2000 chars of HTML):\n{text_content}"

            elif action == "scroll_page":
                print(f"[BrowserUseToolWrapper] Scrolling page {scroll_direction} by {scroll_amount_pixels}px (0=default)...")
                direction_multiplier = -1 if scroll_direction == "up" else 1
                
                # Determine actual scroll amount
                # browser_use might have a default scroll, or use viewport height.
                # For now, a fixed default if scroll_amount_pixels is 0.
                pixels_to_scroll = scroll_amount_pixels if scroll_amount_pixels != 0 else 500 # Default 500px
                
                await page.evaluate(f"window.scrollBy(0, {direction_multiplier * pixels_to_scroll})")
                return f"Scrolled {scroll_direction} by {pixels_to_scroll} pixels."
            
            else:
                return f"Error: Unknown browser action '{action}'."

        except Exception as e:
            # Attempt to capture a screenshot on error for debugging
            error_message = f"Error during browser action '{action}': {str(e)}"
            print(f"[BrowserUseToolWrapper] {error_message}")
            try:
                if page: # Ensure page object exists
                    # In browser_use, screenshots are usually taken on the page object.
                    # screenshot_bytes = await page.screenshot(full_page=False)
                    # import base64
                    # screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                    # return f"{error_message}. Screenshot (base64) might be available if implemented and returned."
                    # For now, just the error. Adding screenshot requires more handling of return types.
                    pass # Placeholder for screenshot logic
            except Exception as se:
                print(f"[BrowserUseToolWrapper] Could not take screenshot on error: {se}")
            return error_message

    async def cleanup(self):
        # This method should be called when the environment is closed.
        async with self._lock:
            if self._browser:
                print("[BrowserUseToolWrapper] Cleaning up browser instance...")
                await self._browser.close()
                self._browser = None
                self._context = None
                self._current_page_object = None
                print("[BrowserUseToolWrapper] Browser cleaned up.")

    def __del__(self):
        # Ensure cleanup if the object is garbage collected, though explicit cleanup is preferred.
        if self._browser:
            try:
                # Running async code in __del__ is tricky and generally discouraged.
                # It's better to rely on explicit cleanup call.
                # For environments that might not call cleanup, this is a fallback.
                if asyncio.get_event_loop().is_running():
                     asyncio.ensure_future(self.cleanup())
                else:
                    # This is problematic as it might try to create a new loop
                    # asyncio.run(self.cleanup())
                    print("[BrowserUseToolWrapper] __del__: Browser instance exists but event loop not running. Cannot cleanup async.")
                    pass
            except Exception as e:
                print(f"[BrowserUseToolWrapper] Error during __del__ cleanup: {e}")
