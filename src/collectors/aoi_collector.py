"""
AOI (Area of Interest) Collector for Webpages.

This module provides functionality to extract Areas of Interest from webpages
using Selenium WebDriver. It identifies interactive elements, text blocks,
images, and other UI components that are typically targets for eye-tracking
studies.
"""

import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import pandas as pd

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.common.exceptions import (
        TimeoutException,
        NoSuchElementException,
        StaleElementReferenceException,
    )
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


@dataclass
class AOIElement:
    """Represents a single Area of Interest element on a webpage."""

    name: str
    element_type: str  # button, link, image, text, input, video, etc.
    bounds: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels
    bounds_normalized: Tuple[float, float, float, float]  # Normalized 0-1
    center: Tuple[float, float]  # Center point in pixels
    center_normalized: Tuple[float, float]  # Normalized 0-1
    area: float  # Area in pixels
    area_normalized: float  # Area as fraction of viewport
    text_content: Optional[str] = None
    tag_name: str = ""
    css_selector: str = ""
    xpath: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    is_visible: bool = True
    is_interactive: bool = False
    z_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def contains_point(self, x: float, y: float, normalized: bool = True) -> bool:
        """Check if a point is within this AOI."""
        if normalized:
            x1, y1, x2, y2 = self.bounds_normalized
        else:
            x1, y1, x2, y2 = self.bounds
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class AOICollection:
    """Collection of AOI elements from a webpage."""

    url: str
    title: str
    viewport_size: Tuple[int, int]  # (width, height)
    scroll_position: Tuple[int, int]  # (x, y)
    page_size: Tuple[int, int]  # Full page (width, height)
    timestamp: str
    elements: List[AOIElement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "title": self.title,
            "viewport_size": self.viewport_size,
            "scroll_position": self.scroll_position,
            "page_size": self.page_size,
            "timestamp": self.timestamp,
            "elements": [e.to_dict() for e in self.elements],
            "metadata": self.metadata,
        }

    def get_aoi_dict(self) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Get AOIs in format compatible with TobiiAnalyzer.analyze_aoi().

        Returns:
            Dict mapping AOI names to normalized bounds (x1, y1, x2, y2).
        """
        return {elem.name: elem.bounds_normalized for elem in self.elements}

    def find_aoi_at_point(
        self, x: float, y: float, normalized: bool = True
    ) -> Optional[AOIElement]:
        """Find which AOI contains the given point."""
        for elem in self.elements:
            if elem.contains_point(x, y, normalized):
                return elem
        return None


class AOICollector:
    """
    Collects Areas of Interest (AOI) from webpages using Selenium.

    This collector identifies interactive elements, images, text blocks,
    and other UI components that are relevant for eye-tracking analysis.

    Example:
        >>> collector = AOICollector()
        >>> collector.start_browser()
        >>> collection = collector.collect_aois("https://example.com")
        >>> df = collector.get_dataframe()
        >>> collector.save_data("example_aois")
        >>> collector.stop_browser()
    """

    # Default element types to collect
    DEFAULT_SELECTORS = {
        "button": "button, [role='button'], input[type='button'], input[type='submit']",
        "link": "a[href]",
        "image": "img, [role='img'], svg",
        "input": "input:not([type='hidden']), textarea, select",
        "video": "video, [role='video']",
        "heading": "h1, h2, h3, h4, h5, h6",
        "text_block": "p, article, section, [role='article']",
        "navigation": "nav, [role='navigation'], header, footer",
        "form": "form",
        "list": "ul, ol, [role='list']",
        "menu": "[role='menu'], [role='menubar']",
        "dialog": "[role='dialog'], [role='alertdialog']",
        "card": "[class*='card'], [class*='Card']",
    }

    def __init__(
        self,
        output_dir: str = "data/raw/aoi",
        browser: str = "chrome",
        headless: bool = True,
        viewport_size: Tuple[int, int] = (1920, 1080),
        custom_selectors: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the AOI Collector.

        Args:
            output_dir: Directory to save collected AOI data.
            browser: Browser to use ("chrome" or "firefox").
            headless: Run browser in headless mode.
            viewport_size: Browser viewport size (width, height).
            custom_selectors: Additional CSS selectors to collect.
        """
        if not SELENIUM_AVAILABLE:
            raise ImportError(
                "Selenium is required for AOICollector. "
                "Install with: pip install selenium"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.browser_type = browser.lower()
        self.headless = headless
        self.viewport_size = viewport_size

        # Combine default and custom selectors
        self.selectors = self.DEFAULT_SELECTORS.copy()
        if custom_selectors:
            self.selectors.update(custom_selectors)

        # Browser instance
        self.driver: Optional[webdriver.Remote] = None

        # Collected data
        self.collections: List[AOICollection] = []
        self._lock = threading.Lock()

        # Configuration
        self.min_element_size = 10  # Minimum pixels for width/height
        self.wait_timeout = 10  # Seconds to wait for page load
        self.scroll_pause = 0.5  # Seconds between scroll actions

    def start_browser(self) -> bool:
        """
        Start the browser instance.

        Returns:
            True if browser started successfully.
        """
        try:
            if self.browser_type == "chrome":
                options = ChromeOptions()
                if self.headless:
                    options.add_argument("--headless=new")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument(
                    f"--window-size={self.viewport_size[0]},{self.viewport_size[1]}"
                )
                self.driver = webdriver.Chrome(options=options)

            elif self.browser_type == "firefox":
                options = FirefoxOptions()
                if self.headless:
                    options.add_argument("--headless")
                self.driver = webdriver.Firefox(options=options)
                self.driver.set_window_size(*self.viewport_size)

            else:
                raise ValueError(f"Unsupported browser: {self.browser_type}")

            return True

        except Exception as e:
            print(f"Failed to start browser: {e}")
            return False

    def stop_browser(self) -> None:
        """Stop and close the browser instance."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

    def navigate_to(self, url: str, wait_for_load: bool = True) -> bool:
        """
        Navigate to a URL.

        Args:
            url: The URL to navigate to.
            wait_for_load: Wait for page to fully load.

        Returns:
            True if navigation successful.
        """
        if not self.driver:
            raise RuntimeError("Browser not started. Call start_browser() first.")

        try:
            self.driver.get(url)

            if wait_for_load:
                WebDriverWait(self.driver, self.wait_timeout).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )

            return True

        except TimeoutException:
            print(f"Timeout waiting for page to load: {url}")
            return False
        except Exception as e:
            print(f"Navigation failed: {e}")
            return False

    def _get_element_info(
        self, element, element_type: str, index: int
    ) -> Optional[AOIElement]:
        """Extract information from a web element."""
        try:
            # Get element location and size
            location = element.location
            size = element.size

            # Skip elements that are too small or not visible
            if size["width"] < self.min_element_size or size["height"] < self.min_element_size:
                return None

            if not element.is_displayed():
                return None

            # Calculate bounds
            x1 = location["x"]
            y1 = location["y"]
            x2 = x1 + size["width"]
            y2 = y1 + size["height"]

            # Get viewport dimensions
            viewport_width = self.driver.execute_script("return window.innerWidth")
            viewport_height = self.driver.execute_script("return window.innerHeight")

            # Skip elements outside viewport
            if x2 < 0 or y2 < 0 or x1 > viewport_width or y1 > viewport_height:
                return None

            # Clamp to viewport
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(viewport_width, x2)
            y2 = min(viewport_height, y2)

            # Normalize coordinates
            x1_norm = x1 / viewport_width
            y1_norm = y1 / viewport_height
            x2_norm = x2 / viewport_width
            y2_norm = y2 / viewport_height

            # Calculate center and area
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            area_norm = (x2_norm - x1_norm) * (y2_norm - y1_norm)

            # Get text content
            text = element.text[:100] if element.text else None

            # Get tag and attributes
            tag_name = element.tag_name

            # Get useful attributes
            attrs = {}
            for attr in ["id", "class", "name", "href", "src", "alt", "title", "aria-label"]:
                try:
                    value = element.get_attribute(attr)
                    if value:
                        attrs[attr] = value[:200]  # Limit length
                except Exception:
                    pass

            # Generate unique name
            elem_id = attrs.get("id", "")
            elem_class = attrs.get("class", "").split()[0] if attrs.get("class") else ""
            name = f"{element_type}_{elem_id or elem_class or tag_name}_{index}"

            # Check if interactive
            interactive_tags = {"button", "a", "input", "select", "textarea"}
            is_interactive = tag_name.lower() in interactive_tags or attrs.get("onclick")

            # Get z-index
            z_index = 0
            try:
                z_str = element.value_of_css_property("z-index")
                if z_str and z_str != "auto":
                    z_index = int(z_str)
            except Exception:
                pass

            return AOIElement(
                name=name,
                element_type=element_type,
                bounds=(x1, y1, x2, y2),
                bounds_normalized=(x1_norm, y1_norm, x2_norm, y2_norm),
                center=(center_x, center_y),
                center_normalized=(center_x / viewport_width, center_y / viewport_height),
                area=area,
                area_normalized=area_norm,
                text_content=text,
                tag_name=tag_name,
                css_selector=self._generate_css_selector(element, attrs),
                xpath="",  # XPath generation is expensive, skip for now
                attributes=attrs,
                is_visible=True,
                is_interactive=is_interactive,
                z_index=z_index,
            )

        except StaleElementReferenceException:
            return None
        except Exception as e:
            print(f"Error extracting element info: {e}")
            return None

    def _generate_css_selector(self, element, attrs: Dict[str, str]) -> str:
        """Generate a CSS selector for the element."""
        tag = element.tag_name

        if attrs.get("id"):
            return f"#{attrs['id']}"

        selector = tag
        if attrs.get("class"):
            classes = attrs["class"].split()[:3]  # Limit to first 3 classes
            selector += "".join(f".{c}" for c in classes)

        return selector

    def collect_aois(
        self,
        url: Optional[str] = None,
        element_types: Optional[List[str]] = None,
        include_all_visible: bool = False,
    ) -> AOICollection:
        """
        Collect all AOIs from the current page or a new URL.

        Args:
            url: URL to collect from (navigates if provided).
            element_types: Specific element types to collect (default: all).
            include_all_visible: Include all visible elements, not just predefined types.

        Returns:
            AOICollection containing all found elements.
        """
        if not self.driver:
            raise RuntimeError("Browser not started. Call start_browser() first.")

        if url:
            if not self.navigate_to(url):
                raise RuntimeError(f"Failed to navigate to {url}")

        # Get page info
        current_url = self.driver.current_url
        title = self.driver.title
        viewport_width = self.driver.execute_script("return window.innerWidth")
        viewport_height = self.driver.execute_script("return window.innerHeight")
        scroll_x = self.driver.execute_script("return window.pageXOffset")
        scroll_y = self.driver.execute_script("return window.pageYOffset")
        page_width = self.driver.execute_script("return document.body.scrollWidth")
        page_height = self.driver.execute_script("return document.body.scrollHeight")

        elements: List[AOIElement] = []

        # Determine which selectors to use
        types_to_collect = element_types or list(self.selectors.keys())

        # Collect elements by type
        for elem_type in types_to_collect:
            if elem_type not in self.selectors:
                continue

            selector = self.selectors[elem_type]

            try:
                found_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)

                for idx, elem in enumerate(found_elements):
                    aoi = self._get_element_info(elem, elem_type, idx)
                    if aoi:
                        elements.append(aoi)

            except Exception as e:
                print(f"Error collecting {elem_type} elements: {e}")

        # Optionally collect all visible elements
        if include_all_visible:
            try:
                all_elements = self.driver.find_elements(By.XPATH, "//*")
                for idx, elem in enumerate(all_elements):
                    aoi = self._get_element_info(elem, "generic", idx)
                    if aoi and not any(e.bounds == aoi.bounds for e in elements):
                        elements.append(aoi)
            except Exception as e:
                print(f"Error collecting all visible elements: {e}")

        # Sort by z-index (higher = on top)
        elements.sort(key=lambda e: e.z_index, reverse=True)

        # Create collection
        collection = AOICollection(
            url=current_url,
            title=title,
            viewport_size=(viewport_width, viewport_height),
            scroll_position=(scroll_x, scroll_y),
            page_size=(page_width, page_height),
            timestamp=datetime.now().isoformat(),
            elements=elements,
            metadata={
                "browser": self.browser_type,
                "headless": self.headless,
                "element_types_collected": types_to_collect,
                "total_elements": len(elements),
            },
        )

        # Store collection
        with self._lock:
            self.collections.append(collection)

        return collection

    def collect_aois_full_page(
        self,
        url: Optional[str] = None,
        scroll_step: int = 500,
    ) -> List[AOICollection]:
        """
        Collect AOIs from the entire page by scrolling.

        Args:
            url: URL to collect from.
            scroll_step: Pixels to scroll each step.

        Returns:
            List of AOICollections, one per scroll position.
        """
        if url:
            if not self.navigate_to(url):
                raise RuntimeError(f"Failed to navigate to {url}")

        collections = []

        # Get page height
        page_height = self.driver.execute_script("return document.body.scrollHeight")
        viewport_height = self.driver.execute_script("return window.innerHeight")

        current_scroll = 0

        while current_scroll < page_height:
            # Scroll to position
            self.driver.execute_script(f"window.scrollTo(0, {current_scroll})")

            # Wait for content to load
            import time
            time.sleep(self.scroll_pause)

            # Collect AOIs at this position
            collection = self.collect_aois()
            collections.append(collection)

            # Move to next position
            current_scroll += scroll_step

            # Check if we've reached the bottom
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height > page_height:
                page_height = new_height

        return collections

    def take_screenshot(
        self,
        filename: Optional[str] = None,
        full_page: bool = False,
    ) -> Path:
        """
        Take a screenshot of the current page.

        Args:
            filename: Output filename (without extension).
            full_page: Capture full page (Firefox only).

        Returns:
            Path to the saved screenshot.
        """
        if not self.driver:
            raise RuntimeError("Browser not started.")

        if filename is None:
            domain = urlparse(self.driver.current_url).netloc.replace(".", "_")
            filename = f"screenshot_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        filepath = self.output_dir / f"{filename}.png"

        if full_page and self.browser_type == "firefox":
            self.driver.save_full_page_screenshot(str(filepath))
        else:
            self.driver.save_screenshot(str(filepath))

        return filepath

    def get_dataframe(self, collection: Optional[AOICollection] = None) -> pd.DataFrame:
        """
        Convert AOI collection(s) to a pandas DataFrame.

        Args:
            collection: Specific collection to convert (default: all collections).

        Returns:
            DataFrame with all AOI elements.
        """
        if collection:
            collections = [collection]
        else:
            collections = self.collections

        if not collections:
            return pd.DataFrame()

        rows = []
        for coll in collections:
            for elem in coll.elements:
                row = elem.to_dict()
                row["url"] = coll.url
                row["page_title"] = coll.title
                row["viewport_width"] = coll.viewport_size[0]
                row["viewport_height"] = coll.viewport_size[1]
                row["collection_timestamp"] = coll.timestamp

                # Flatten bounds tuples
                row["x1"], row["y1"], row["x2"], row["y2"] = row.pop("bounds")
                row["x1_norm"], row["y1_norm"], row["x2_norm"], row["y2_norm"] = row.pop("bounds_normalized")
                row["center_x"], row["center_y"] = row.pop("center")
                row["center_x_norm"], row["center_y_norm"] = row.pop("center_normalized")

                # Convert attributes dict to JSON string
                row["attributes"] = json.dumps(row["attributes"])

                rows.append(row)

        return pd.DataFrame(rows)

    def save_data(
        self,
        filename: Optional[str] = None,
        format: str = "csv",
        include_json: bool = True,
    ) -> Dict[str, Path]:
        """
        Save collected AOI data to disk.

        Args:
            filename: Base filename (without extension).
            format: Output format ("csv" or "parquet").
            include_json: Also save as JSON with full structure.

        Returns:
            Dict mapping format names to file paths.
        """
        if not self.collections:
            raise ValueError("No AOI data collected. Call collect_aois() first.")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aoi_collection_{timestamp}"

        saved_files = {}

        # Save DataFrame format
        df = self.get_dataframe()

        if format == "csv":
            filepath = self.output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)
            saved_files["csv"] = filepath
        elif format == "parquet":
            filepath = self.output_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
            saved_files["parquet"] = filepath

        # Save JSON format with full structure
        if include_json:
            json_path = self.output_dir / f"{filename}.json"
            with open(json_path, "w") as f:
                json.dump(
                    [coll.to_dict() for coll in self.collections],
                    f,
                    indent=2,
                )
            saved_files["json"] = json_path

        return saved_files

    def clear_data(self) -> None:
        """Clear all collected data."""
        with self._lock:
            self.collections.clear()

    def get_aoi_dict_for_analyzer(
        self,
        collection_index: int = -1,
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Get AOIs in format compatible with TobiiAnalyzer.analyze_aoi().

        Args:
            collection_index: Index of collection to use (default: last).

        Returns:
            Dict mapping AOI names to normalized bounds.
        """
        if not self.collections:
            return {}

        return self.collections[collection_index].get_aoi_dict()

    def __enter__(self):
        """Context manager entry."""
        self.start_browser()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_browser()
        return False


def collect_webpage_aois(
    url: str,
    output_dir: str = "data/raw/aoi",
    save: bool = True,
    screenshot: bool = True,
    headless: bool = True,
) -> Tuple[AOICollection, Optional[pd.DataFrame]]:
    """
    Convenience function to collect AOIs from a single webpage.

    Args:
        url: The webpage URL to analyze.
        output_dir: Directory to save data.
        save: Whether to save data to disk.
        screenshot: Whether to take a screenshot.
        headless: Run browser in headless mode.

    Returns:
        Tuple of (AOICollection, DataFrame or None).

    Example:
        >>> collection, df = collect_webpage_aois("https://example.com")
        >>> print(f"Found {len(collection.elements)} AOIs")
        >>> aois = collection.get_aoi_dict()  # For use with TobiiAnalyzer
    """
    with AOICollector(output_dir=output_dir, headless=headless) as collector:
        collection = collector.collect_aois(url)

        df = None
        if save:
            domain = urlparse(url).netloc.replace(".", "_")
            collector.save_data(f"aoi_{domain}")
            df = collector.get_dataframe()

        if screenshot:
            collector.take_screenshot()

        return collection, df
