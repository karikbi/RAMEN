#!/usr/bin/env python3
"""
Web Scraping Wallpaper Sources

Fetches wallpapers from websites without APIs using HTML scraping.
Supports: 4KWallpapers.com, WallpaperCat.com

Features:
- Batch extraction: Get all wallpapers from a page grid in one request
- Parallel/alternate scraping: Interleave sources to avoid rate limiting
- Integration with deduplication system

Author: RAMEN Pipeline
Version: 1.0.0
"""

import asyncio
import aiohttp
import logging
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
from urllib.parse import urljoin, urlparse

if TYPE_CHECKING:
    from dedup_manager import DuplicateChecker

try:
    from bs4 import BeautifulSoup
    HAS_BEAUTIFULSOUP = True
except ImportError:
    HAS_BEAUTIFULSOUP = False
    BeautifulSoup = None


logger = logging.getLogger("wallpaper_curator")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ScrapingSourceConfig:
    """Configuration for a web scraping source."""
    name: str
    url: str
    fetch_count: int = 50
    enabled: bool = True


@dataclass
class WebScrapingConfig:
    """Configuration for all web scraping sources."""

    # 4KWallpapers.com sources
    fourk_sources: list[ScrapingSourceConfig] = field(default_factory=lambda: [
        ScrapingSourceConfig("minimalism", "https://4kwallpapers.com/minimalism-wallpapers/", 50),
    ])

    # WallpaperCat.com sources
    wallpapercat_sources: list[ScrapingSourceConfig] = field(default_factory=lambda: [
        ScrapingSourceConfig("studio-ghibli", "https://wallpapercat.com/studio-ghibli-wallpapers", 50),
        ScrapingSourceConfig("desktop", "https://wallpapercat.com/desktop", 50),
        ScrapingSourceConfig("wind-rises", "https://wallpapercat.com/the-wind-rises-wallpapers", 28),
    ])

    # Rate limiting - delay between page requests to same domain
    request_delay: float = 2.0

    # User agent for requests
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


# =============================================================================
# 4KWALLPAPERS.COM FETCHER
# =============================================================================

class FourKWallpapersFetcher:
    """
    Fetches wallpapers from 4kwallpapers.com.

    HTML Structure:
    - Download links: <a href="/images/wallpapers/filename.ext" target="_blank">Download</a>
    - Load more button: <button id="load-more-button">Load more</button> (requires JS)
    - Main content is under <h1>, avoid "Featured Collection" section

    Since the site uses JavaScript for "Load more", we use pagination via ?page=N instead.
    Each page contains ~24 wallpapers in a grid.
    """

    BASE_URL = "https://4kwallpapers.com"

    def __init__(self, config: WebScrapingConfig):
        self.config = config
        self.headers = {
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://4kwallpapers.com/",
        }

    async def fetch_page(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> list[dict[str, Any]]:
        """
        Fetch a single page and extract ALL wallpaper download URLs from the grid.

        The download links have format:
        <a href="/images/wallpapers/black-panther-minimal-art-black-background-3840x2160-2762.png" target="_blank">Download</a>

        We extract content under the main h1 header and avoid "Featured Collection" section.
        """
        wallpapers = []

        try:
            async with session.get(url, headers=self.headers, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"4KWallpapers page returned {response.status}: {url}")
                    return []

                html = await response.text()

                if HAS_BEAUTIFULSOUP:
                    wallpapers = self._extract_with_beautifulsoup(html, url)
                else:
                    wallpapers = self._extract_with_regex(html, url)

                logger.debug(f"4KWallpapers: Extracted {len(wallpapers)} wallpapers from {url}")

        except asyncio.TimeoutError:
            logger.warning(f"4KWallpapers timeout: {url}")
        except Exception as e:
            logger.error(f"4KWallpapers fetch error: {e}")

        return wallpapers

    def _extract_with_beautifulsoup(self, html: str, source_url: str) -> list[dict[str, Any]]:
        """Extract wallpapers using BeautifulSoup for accurate parsing."""
        wallpapers = []
        soup = BeautifulSoup(html, "lxml" if "lxml" in str(type(BeautifulSoup)) else "html.parser")

        # Find all download links - they have href starting with /images/wallpapers/
        # and contain "Download" text
        download_links = soup.find_all("a", href=re.compile(r"^/images/wallpapers/.*\.(jpg|jpeg|png|webp)$", re.IGNORECASE))

        found_urls = set()
        for link in download_links:
            # Skip if inside "Featured Collection" section
            # Check parent elements for "featured" class or section
            parent = link.find_parent(class_=re.compile(r"featured", re.IGNORECASE))
            if parent:
                continue

            href = link.get("href", "")
            if not href or href in found_urls:
                continue

            found_urls.add(href)

            # Build full URL from relative path
            img_url = f"{self.BASE_URL}{href}"

            # Extract ID and title from filename
            # Pattern: /images/wallpapers/black-panther-minimal-art-black-background-3840x2160-2762.png
            filename = href.split("/")[-1]
            id_match = re.search(r"-(\d+)\.\w+$", filename)
            wallpaper_id = id_match.group(1) if id_match else hashlib.md5(href.encode()).hexdigest()[:8]

            # Extract resolution from filename if present (e.g., 3840x2160)
            resolution_match = re.search(r"-(\d{3,5}x\d{3,5})-", filename)
            resolution = resolution_match.group(1) if resolution_match else None

            # Build title from filename (remove resolution and ID parts)
            title_part = filename.rsplit(".", 1)[0]  # Remove extension
            title_part = re.sub(r"-\d+$", "", title_part)  # Remove ID
            title_part = re.sub(r"-\d{3,5}x\d{3,5}", "", title_part)  # Remove resolution
            title = title_part.replace("-", " ").strip().title()

            wallpapers.append({
                "id": wallpaper_id,
                "url": img_url,
                "title": title or "Wallpaper",
                "resolution": resolution,
                "source_page": source_url,
            })

        return wallpapers

    def _extract_with_regex(self, html: str, source_url: str) -> list[dict[str, Any]]:
        """Fallback regex extraction if BeautifulSoup not available."""
        wallpapers = []

        # Find download links with relative URLs: href="/images/wallpapers/..."
        # Pattern matches: href="/images/wallpapers/filename.ext"
        pattern = re.compile(
            r'href="(/images/wallpapers/([^"]+\.(jpg|jpeg|png|webp)))"[^>]*>Download',
            re.IGNORECASE
        )

        found_urls = set()
        for match in pattern.finditer(html):
            href = match.group(1)
            if href in found_urls:
                continue
            found_urls.add(href)

            filename = match.group(2)
            img_url = f"{self.BASE_URL}{href}"

            # Extract ID from filename
            id_match = re.search(r"-(\d+)\.\w+$", filename)
            wallpaper_id = id_match.group(1) if id_match else hashlib.md5(href.encode()).hexdigest()[:8]

            # Extract resolution
            resolution_match = re.search(r"-(\d{3,5}x\d{3,5})-", filename)
            resolution = resolution_match.group(1) if resolution_match else None

            # Build title
            title_part = filename.rsplit(".", 1)[0]
            title_part = re.sub(r"-\d+$", "", title_part)
            title_part = re.sub(r"-\d{3,5}x\d{3,5}", "", title_part)
            title = title_part.replace("-", " ").strip().title()

            wallpapers.append({
                "id": wallpaper_id,
                "url": img_url,
                "title": title or "Wallpaper",
                "resolution": resolution,
                "source_page": source_url,
            })

        return wallpapers

    async def fetch_source(
        self,
        session: aiohttp.ClientSession,
        source: ScrapingSourceConfig,
        dedup_checker: Optional["DuplicateChecker"] = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch wallpapers from a single 4KWallpapers source (e.g., minimalism category).
        Paginates through pages until fetch_count is reached.
        """
        all_wallpapers = []
        page = 1
        max_pages = 20  # Safety limit

        logger.info(f"4KWallpapers: Fetching from '{source.name}' (target: {source.fetch_count})")

        while len(all_wallpapers) < source.fetch_count and page <= max_pages:
            # Build paginated URL
            if page == 1:
                page_url = source.url
            else:
                page_url = f"{source.url}?page={page}"

            page_wallpapers = await self.fetch_page(session, page_url)

            if not page_wallpapers:
                logger.debug(f"4KWallpapers: No more wallpapers on page {page}")
                break

            # Filter duplicates
            skipped = 0
            for wp in page_wallpapers:
                if len(all_wallpapers) >= source.fetch_count:
                    break

                full_id = f"4kwallpapers_{wp['id']}"

                if dedup_checker:
                    is_dup_id, _ = dedup_checker.check_id(full_id)
                    is_dup_url, _ = dedup_checker.check_url(wp["url"])
                    if is_dup_id or is_dup_url:
                        skipped += 1
                        continue

                wp["full_id"] = full_id
                all_wallpapers.append(wp)

            logger.info(f"  Page {page}: {len(page_wallpapers)} found, {skipped} duplicates skipped. Total: {len(all_wallpapers)}")

            page += 1
            await asyncio.sleep(self.config.request_delay)

        return all_wallpapers

    async def fetch_all(
        self,
        session: aiohttp.ClientSession,
        dedup_checker: Optional["DuplicateChecker"] = None,
    ) -> list[dict[str, Any]]:
        """Fetch from all configured 4KWallpapers sources."""
        all_wallpapers = []

        for source in self.config.fourk_sources:
            if not source.enabled:
                continue

            wallpapers = await self.fetch_source(session, source, dedup_checker)
            all_wallpapers.extend(wallpapers)

        logger.info(f"4KWallpapers: Total {len(all_wallpapers)} wallpapers fetched")
        return all_wallpapers


# =============================================================================
# WALLPAPERCAT.COM FETCHER
# =============================================================================

class WallpaperCatFetcher:
    """
    Fetches wallpapers from wallpapercat.com.

    HTML Structure:
    - Wallpaper links: <a class="ui fluid image image_popup_trigger" data-id="ID" href="/w/full/path/to/image.jpg">
    - Full resolution images at: https://wallpapercat.com/w/full/{path}.jpg
    - Collections show all wallpapers on one page (no pagination needed)
    """

    BASE_URL = "https://wallpapercat.com"

    def __init__(self, config: WebScrapingConfig):
        self.config = config
        self.headers = {
            "User-Agent": config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://wallpapercat.com/",
        }

    async def fetch_page(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> list[dict[str, Any]]:
        """
        Fetch a collection page and extract ALL wallpaper URLs from the grid.

        Wallpaper links have format:
        <a class="ui fluid image image_popup_trigger ads_popup" data-id="5823589" 
           href="/w/full/c/7/e/5823589-2920x1640-desktop-hd-boy-programmer-wallpaper-image.jpg">

        Returns list of wallpaper info dicts.
        """
        wallpapers = []

        try:
            async with session.get(url, headers=self.headers, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"WallpaperCat page returned {response.status}: {url}")
                    return []

                html = await response.text()

                # Extract collection name from URL for context
                collection_name = urlparse(url).path.strip("/").replace("-wallpapers", "").replace("-", " ").title()

                if HAS_BEAUTIFULSOUP:
                    wallpapers = self._extract_with_beautifulsoup(html, url, collection_name)
                else:
                    wallpapers = self._extract_with_regex(html, url, collection_name)

                logger.debug(f"WallpaperCat: Extracted {len(wallpapers)} wallpapers from {url}")

        except asyncio.TimeoutError:
            logger.warning(f"WallpaperCat timeout: {url}")
        except Exception as e:
            logger.error(f"WallpaperCat fetch error: {e}")

        return wallpapers

    def _extract_with_beautifulsoup(self, html: str, source_url: str, collection_name: str) -> list[dict[str, Any]]:
        """Extract wallpapers using BeautifulSoup for accurate parsing."""
        wallpapers = []
        soup = BeautifulSoup(html, "lxml" if "lxml" in str(type(BeautifulSoup)) else "html.parser")

        found_urls = set()

        # Method 1: Find anchor links with class "image_popup_trigger" (desktop page style)
        # Pattern: <a class="... image_popup_trigger ..." data-id="ID" href="/w/full/...">
        wallpaper_links = soup.find_all("a", class_=re.compile(r"image_popup_trigger", re.IGNORECASE))

        for link in wallpaper_links:
            href = link.get("href", "")
            data_id = link.get("data-id", "")

            if not href or not href.startswith("/w/full/"):
                continue

            if href in found_urls:
                continue
            found_urls.add(href)

            wp = self._parse_wallpaper_url(href, data_id, collection_name, source_url)
            if wp:
                wallpapers.append(wp)

        # Method 2: Find img tags with src="/w/full/..." (collection page style like Studio Ghibli)
        # This pattern is used when images are displayed directly in img tags
        img_tags = soup.find_all("img", src=re.compile(r"^/w/full/", re.IGNORECASE))

        for img in img_tags:
            src = img.get("src", "")
            if not src or not src.startswith("/w/full/"):
                continue

            if src in found_urls:
                continue
            found_urls.add(src)

            # Try to get ID from parent anchor tag's data-id if available
            parent_anchor = img.find_parent("a")
            data_id = parent_anchor.get("data-id", "") if parent_anchor else ""

            wp = self._parse_wallpaper_url(src, data_id, collection_name, source_url)
            if wp:
                wallpapers.append(wp)

        return wallpapers

    def _parse_wallpaper_url(self, url_path: str, data_id: str, collection_name: str, source_url: str) -> Optional[dict[str, Any]]:
        """Parse a wallpaper URL path into a wallpaper dict."""
        # Build full URL
        img_url = f"{self.BASE_URL}{url_path}"

        # Use data-id as the wallpaper ID, fallback to extraction from URL
        if data_id:
            wallpaper_id = data_id
        else:
            # Extract ID from filename (e.g., 5823589 from 5823589-2920x1640-...)
            # or from path like /w/full/7/7/4/1198914-...
            filename = url_path.split("/")[-1]
            id_match = re.search(r"^(\d+)", filename)
            wallpaper_id = id_match.group(1) if id_match else hashlib.md5(url_path.encode()).hexdigest()[:8]

        # Extract resolution and title from filename
        filename = url_path.split("/")[-1]
        resolution = None
        title = collection_name

        if "-" in filename:
            # Pattern: 5823589-2920x1640-desktop-hd-boy-programmer-wallpaper-image.jpg
            parts = filename.rsplit(".", 1)[0].split("-")

            # Extract resolution (e.g., 2920x1640)
            for part in parts:
                if re.match(r"^\d{3,5}x\d{3,5}$", part):
                    resolution = part
                    break

            # Build title - skip ID, resolution, and common filter words
            skip_words = {"4k", "hd", "desktop", "wallpaper", "background", "photo", "image"}
            title_parts = []
            for part in parts[1:]:  # Skip first part (ID)
                if re.match(r"^\d{3,5}x\d{3,5}$", part):
                    continue  # Skip resolution
                if part.lower() in skip_words:
                    continue
                title_parts.append(part)

            if title_parts:
                title = " ".join(title_parts).title()

        return {
            "id": wallpaper_id,
            "url": img_url,
            "title": title or "Wallpaper",
            "resolution": resolution,
            "collection": collection_name,
            "source_page": source_url,
        }

    def _extract_with_regex(self, html: str, source_url: str, collection_name: str) -> list[dict[str, Any]]:
        """Fallback regex extraction if BeautifulSoup not available."""
        wallpapers = []
        found_urls = set()

        # Method 1: Pattern for anchor tags with image_popup_trigger class
        # <a class="... image_popup_trigger ..." data-id="ID" href="/w/full/path.jpg">
        anchor_pattern = re.compile(
            r'<a\s+[^>]*class="[^"]*image_popup_trigger[^"]*"[^>]*data-id="(\d+)"[^>]*href="(/w/full/[^"]+)"',
            re.IGNORECASE | re.DOTALL
        )

        for match in anchor_pattern.finditer(html):
            data_id = match.group(1)
            href = match.group(2)

            if href in found_urls:
                continue
            found_urls.add(href)

            img_url = f"{self.BASE_URL}{href}"
            filename = href.split("/")[-1]
            resolution_match = re.search(r"-(\d{3,5}x\d{3,5})-", filename)
            resolution = resolution_match.group(1) if resolution_match else None

            wallpapers.append({
                "id": data_id,
                "url": img_url,
                "title": collection_name,
                "resolution": resolution,
                "collection": collection_name,
                "source_page": source_url,
            })

        # Method 2: Pattern for img tags with src="/w/full/..." (collection pages)
        # <img src="/w/full/7/7/4/1198914-2560x1440-desktop-hd-studio-ghibli-wallpaper-photo.jpg">
        img_pattern = re.compile(
            r'<img\s+[^>]*src="(/w/full/[^"]+\.(jpg|jpeg|png|webp))"',
            re.IGNORECASE
        )

        for match in img_pattern.finditer(html):
            src = match.group(1)

            if src in found_urls:
                continue
            found_urls.add(src)

            img_url = f"{self.BASE_URL}{src}"
            filename = src.split("/")[-1]

            # Extract ID from filename
            id_match = re.search(r"^(\d+)", filename)
            wallpaper_id = id_match.group(1) if id_match else hashlib.md5(src.encode()).hexdigest()[:8]

            resolution_match = re.search(r"-(\d{3,5}x\d{3,5})-", filename)
            resolution = resolution_match.group(1) if resolution_match else None

            wallpapers.append({
                "id": wallpaper_id,
                "url": img_url,
                "title": collection_name,
                "resolution": resolution,
                "collection": collection_name,
                "source_page": source_url,
            })

        return wallpapers

    async def fetch_source(
        self,
        session: aiohttp.ClientSession,
        source: ScrapingSourceConfig,
        dedup_checker: Optional["DuplicateChecker"] = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch wallpapers from a single WallpaperCat collection.
        WallpaperCat shows all wallpapers on one page (no pagination needed).
        """
        logger.info(f"WallpaperCat: Fetching from '{source.name}' (target: {source.fetch_count})")

        page_wallpapers = await self.fetch_page(session, source.url)

        # Filter duplicates
        all_wallpapers = []
        skipped = 0

        for wp in page_wallpapers:
            if len(all_wallpapers) >= source.fetch_count:
                break

            full_id = f"wallpapercat_{wp['id']}"

            if dedup_checker:
                is_dup_id, _ = dedup_checker.check_id(full_id)
                is_dup_url, _ = dedup_checker.check_url(wp["url"])
                if is_dup_id or is_dup_url:
                    skipped += 1
                    continue

            wp["full_id"] = full_id
            all_wallpapers.append(wp)

        logger.info(f"  Found {len(page_wallpapers)}, {skipped} duplicates skipped. Using: {len(all_wallpapers)}")

        return all_wallpapers

    async def fetch_all(
        self,
        session: aiohttp.ClientSession,
        dedup_checker: Optional["DuplicateChecker"] = None,
    ) -> list[dict[str, Any]]:
        """Fetch from all configured WallpaperCat sources."""
        all_wallpapers = []

        for source in self.config.wallpapercat_sources:
            if not source.enabled:
                continue

            wallpapers = await self.fetch_source(session, source, dedup_checker)
            all_wallpapers.extend(wallpapers)

            # Brief delay between collections
            await asyncio.sleep(self.config.request_delay)

        logger.info(f"WallpaperCat: Total {len(all_wallpapers)} wallpapers fetched")
        return all_wallpapers


# =============================================================================
# PARALLEL/ALTERNATE SCRAPING COORDINATOR
# =============================================================================

class WebScrapingCoordinator:
    """
    Coordinates scraping from multiple sources in parallel/alternating fashion.

    Benefits:
    - If one source is rate-limited, we're already fetching from another
    - Maximizes throughput while being respectful to each source
    - Batch extraction: each page request yields many wallpapers
    """

    def __init__(self, config: Optional[WebScrapingConfig] = None):
        self.config = config or WebScrapingConfig()
        self.fourk_fetcher = FourKWallpapersFetcher(self.config)
        self.wallpapercat_fetcher = WallpaperCatFetcher(self.config)

    async def fetch_all_parallel(
        self,
        session: aiohttp.ClientSession,
        dedup_checker: Optional["DuplicateChecker"] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Fetch from all sources in parallel.

        Returns dict with keys '4kwallpapers' and 'wallpapercat', each containing
        a list of wallpaper dicts.
        """
        if not HAS_BEAUTIFULSOUP:
            logger.warning("BeautifulSoup4 not installed. Web scraping will use regex fallback.")

        # Run both fetchers concurrently
        fourk_task = asyncio.create_task(
            self.fourk_fetcher.fetch_all(session, dedup_checker)
        )
        wallpapercat_task = asyncio.create_task(
            self.wallpapercat_fetcher.fetch_all(session, dedup_checker)
        )

        fourk_results, wallpapercat_results = await asyncio.gather(
            fourk_task, wallpapercat_task,
            return_exceptions=True
        )

        results = {
            "4kwallpapers": fourk_results if not isinstance(fourk_results, Exception) else [],
            "wallpapercat": wallpapercat_results if not isinstance(wallpapercat_results, Exception) else [],
        }

        if isinstance(fourk_results, Exception):
            logger.error(f"4KWallpapers fetch failed: {fourk_results}")
        if isinstance(wallpapercat_results, Exception):
            logger.error(f"WallpaperCat fetch failed: {wallpapercat_results}")

        total = len(results["4kwallpapers"]) + len(results["wallpapercat"])
        logger.info(f"Web scraping complete: {total} total wallpapers")
        logger.info(f"  - 4KWallpapers: {len(results['4kwallpapers'])}")
        logger.info(f"  - WallpaperCat: {len(results['wallpapercat'])}")

        return results

    async def fetch_interleaved(
        self,
        session: aiohttp.ClientSession,
        dedup_checker: Optional["DuplicateChecker"] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Fetch from sources in an interleaved fashion.

        Alternates between sources to avoid rate limiting any single source.
        This is useful when parallel fetching might overwhelm a single source.
        """
        results = {
            "4kwallpapers": [],
            "wallpapercat": [],
        }

        # Collect all sources with their fetchers
        all_sources = []

        for source in self.config.fourk_sources:
            if source.enabled:
                all_sources.append(("4kwallpapers", source, self.fourk_fetcher))

        for source in self.config.wallpapercat_sources:
            if source.enabled:
                all_sources.append(("wallpapercat", source, self.wallpapercat_fetcher))

        # Interleave: fetch one source at a time, alternating
        for source_type, source, fetcher in all_sources:
            try:
                wallpapers = await fetcher.fetch_source(session, source, dedup_checker)
                results[source_type].extend(wallpapers)
            except Exception as e:
                logger.error(f"Error fetching {source_type}/{source.name}: {e}")

            # Brief pause before switching to next source
            await asyncio.sleep(1.0)

        total = len(results["4kwallpapers"]) + len(results["wallpapercat"])
        logger.info(f"Web scraping (interleaved) complete: {total} total wallpapers")

        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    if not HAS_BEAUTIFULSOUP:
        logger.warning(
            "BeautifulSoup4 not installed. Install with: pip install beautifulsoup4 lxml\n"
            "Web scraping will fall back to regex extraction (less reliable)."
        )
        return False
    return True


# =============================================================================
# STANDALONE TEST
# =============================================================================

async def _test_scraping():
    """Test the web scrapers."""
    logging.basicConfig(level=logging.INFO)

    config = WebScrapingConfig()
    coordinator = WebScrapingCoordinator(config)

    connector = aiohttp.TCPConnector(limit=5)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        results = await coordinator.fetch_all_parallel(session)

        print("\n=== TEST RESULTS ===")
        print(f"4KWallpapers: {len(results['4kwallpapers'])} wallpapers")
        if results["4kwallpapers"]:
            print(f"  Sample: {results['4kwallpapers'][0]}")

        print(f"WallpaperCat: {len(results['wallpapercat'])} wallpapers")
        if results["wallpapercat"]:
            print(f"  Sample: {results['wallpapercat'][0]}")


if __name__ == "__main__":
    asyncio.run(_test_scraping())
