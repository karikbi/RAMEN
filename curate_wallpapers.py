#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Part 1: Setup and Fetching

Automated fetching of wallpaper candidates from Reddit, Unsplash, and Pexels APIs.
Part of the RAMEN (Refined Automated Media Embedding Network) pipeline.

Author: RAMEN Pipeline
Version: 1.0.0
"""

import asyncio
import aiohttp
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from dedup_manager import DuplicateChecker
from urllib.parse import urlparse

# Central config
try:
    from config_loader import get_config as get_central_config
    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SubredditConfig:
    """Configuration for a subreddit source."""
    name: str
    fetch_count: int
    min_upvotes: int


@dataclass
class Config:
    """Central configuration for the wallpaper curation pipeline."""
    
    # API Credentials (loaded from environment)
    reddit_client_id: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_ID", ""))
    reddit_client_secret: str = field(default_factory=lambda: os.getenv("REDDIT_CLIENT_SECRET", ""))
    
    # Reddit no longer requires keys for this public JSON method
    # User-Agent must be descriptive per Reddit's guidelines
    reddit_user_agent: str = field(default_factory=lambda: os.getenv(
        "REDDIT_USER_AGENT", 
        "python:RAMEN-Wallpaper-Curator:v1.0.0 (by /u/wallpaper_curator)"
    ))
    unsplash_access_key: str = field(default_factory=lambda: os.getenv("UNSPLASH_ACCESS_KEY", ""))
    pexels_api_key: str = field(default_factory=lambda: os.getenv("PEXELS_API_KEY", ""))
    
    # R2 Storage (for future use)
    r2_endpoint: str = field(default_factory=lambda: os.getenv("R2_ENDPOINT", ""))
    r2_access_key: str = field(default_factory=lambda: os.getenv("R2_ACCESS_KEY", ""))
    r2_secret_key: str = field(default_factory=lambda: os.getenv("R2_SECRET_KEY", ""))
    r2_bucket_name: str = field(default_factory=lambda: os.getenv("R2_BUCKET_NAME", ""))
    
    # Quality thresholds (read from central config.yaml)
    quality_threshold: float = field(default_factory=lambda: 
        get_central_config().get('quality.threshold', 0.40) if HAS_CONFIG_LOADER else 0.40
    )
    
    # Subreddit configurations - lowered upvote requirements for more candidates
    subreddits: list[SubredditConfig] = field(default_factory=lambda: [
        SubredditConfig("wallpapers", 100, 100),       # Was 1000 - got 0
        SubredditConfig("EarthPorn", 80, 500),         # Working well
        SubredditConfig("Amoledbackgrounds", 60, 50),  # Was 300 - got 6
        SubredditConfig("spaceporn", 50, 200),         # Working well
        SubredditConfig("CityPorn", 50, 200),          # Working well
        SubredditConfig("SkyPorn", 40, 100),           # Was 500 - got 16
        SubredditConfig("MinimalWallpaper", 40, 20),   # Was 200 - got 0
        SubredditConfig("ImaginaryLandscapes", 40, 100), # Was 500 - got 4
    ])
    
    # Candidate counts per source - increased for more wallpapers
    unsplash_count: int = 100  # Increased from 30
    pexels_count: int = 80     # Increased from 20
    
    # Wallpaper-friendly search queries (read from config.yaml)
    unsplash_search_query: str = field(default_factory=lambda:
        get_central_config().get('sources.unsplash.search_query', 
                                 'landscape nature scenery abstract mountains ocean sky') 
        if HAS_CONFIG_LOADER else 'landscape nature scenery abstract mountains ocean sky'
    )
    pexels_search_query: str = field(default_factory=lambda:
        get_central_config().get('sources.pexels.search_query',
                                 'landscape nature scenery abstract mountains ocean sky')
        if HAS_CONFIG_LOADER else 'landscape nature scenery abstract mountains ocean sky'
    )
    
    # Image orientation filter (landscape/portrait/squarish)
    image_orientation: str = field(default_factory=lambda:
        get_central_config().get('sources.unsplash.orientation', 'landscape')
        if HAS_CONFIG_LOADER else 'landscape'
    )
    
    # Directories
    temp_dir: Path = field(default_factory=lambda: Path("./temp"))
    candidates_dir: Path = field(default_factory=lambda: Path("./temp/candidates"))
    approved_dir: Path = field(default_factory=lambda: Path("./temp/approved"))
    
    # Retry settings
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    
    # Rate limiting
    request_delay: float = 2.0  # seconds between API requests
    
    def validate(self) -> list[str]:
        """Validate required configuration values."""
        errors = []
        # Reddit keys removed from validation
        if not self.unsplash_access_key:
            errors.append("UNSPLASH_ACCESS_KEY not set")
        if not self.pexels_api_key:
            errors.append("PEXELS_API_KEY not set")
        return errors


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class CandidateWallpaper:
    """Represents a wallpaper candidate fetched from any source."""
    id: str
    source: str  # "reddit", "unsplash", "pexels"
    filepath: Optional[Path]
    url: str
    title: str
    artist: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"CandidateWallpaper(id={self.id}, source={self.source}, artist={self.artist})"


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging with timestamps and proper formatting."""
    logger = logging.getLogger("wallpaper_curator")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"curation_{timestamp}.log")
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logging()


# =============================================================================
# UTILITIES
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds (multiplied exponentially).
        exceptions: Tuple of exception types to catch and retry.
    
    Returns:
        Decorated function with retry logic.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {e}")
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def create_directories(config: Config) -> None:
    """Create necessary temp directories for the pipeline."""
    config.temp_dir.mkdir(exist_ok=True)
    config.candidates_dir.mkdir(parents=True, exist_ok=True)
    config.approved_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directories: {config.candidates_dir}, {config.approved_dir}")


def get_file_extension(url: str, content_type: Optional[str] = None) -> str:
    """Extract file extension from URL or content type."""
    # Try URL first
    parsed = urlparse(url)
    path = parsed.path.lower()
    
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        if path.endswith(ext):
            return ext
    
    # Try content type
    if content_type:
        content_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        return content_map.get(content_type.split(";")[0].strip(), ".jpg")
    
    return ".jpg"  # Default

# =============================================================================
# REDDIT API FETCHING
# =============================================================================

class RedditFetcher:
    """Fetches wallpaper candidates from Reddit using OAuth API.
    
    Reddit requires OAuth authentication for API access. This uses the
    'application only' OAuth flow which doesn't require user interaction.
    
    Required environment variables:
        - REDDIT_CLIENT_ID: Reddit app client ID
        - REDDIT_CLIENT_SECRET: Reddit app client secret
    
    Create an app at: https://www.reddit.com/prefs/apps
    """
    
    AUTH_URL = "https://www.reddit.com/api/v1/access_token"
    API_URL = "https://oauth.reddit.com"
    
    def __init__(self, config: Config):
        self.config = config
        self.client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.access_token: Optional[str] = None
        self.user_agent = config.reddit_user_agent
    
    async def _get_access_token(self, session: aiohttp.ClientSession) -> Optional[str]:
        """Get OAuth access token using application-only flow."""
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit credentials not set (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)")
            return None
        
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        headers = {"User-Agent": self.user_agent}
        data = {"grant_type": "client_credentials"}
        
        try:
            async with session.post(
                self.AUTH_URL, 
                auth=auth, 
                headers=headers, 
                data=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Reddit OAuth failed: {response.status} - {error_text}")
                    return None
                
                result = await response.json()
                token = result.get("access_token")
                if token:
                    logger.info("Successfully obtained Reddit OAuth token")
                return token
        except Exception as e:
            logger.error(f"Reddit OAuth request failed: {e}")
            return None
    
    def _extract_image_url(self, post_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract direct image URL from a Reddit post data dict.
        Handles: direct images, imgur links, Reddit galleries.
        """
        url = post_data.get("url", "").lower()
        
        # Direct image links
        if any(url.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
            return post_data["url"]
        
        # Reddit hosted images
        if "i.redd.it" in url:
            return post_data["url"]
        
        # Reddit galleries
        if post_data.get("is_gallery"):
            try:
                media_metadata = post_data.get("media_metadata", {})
                if media_metadata:
                    # Get the first valid image from the gallery
                    for item_id, item in media_metadata.items():
                        if item.get("status") == "valid" and "s" in item:
                            img_url = item["s"].get("u", "")
                            # Unescape URL
                            return img_url.replace("&amp;", "&")
            except Exception as e:
                logger.debug(f"Failed to parse gallery for {post_data.get('id')}: {e}")
            return None
        
        # Imgur single images
        if "imgur.com" in url and not any(x in url for x in ["/a/", "/gallery/"]):
            imgur_id_match = re.search(r"imgur\.com/(\w+)", post_data["url"])
            if imgur_id_match:
                imgur_id = imgur_id_match.group(1)
                return f"https://i.imgur.com/{imgur_id}.jpg"
        
        # Preview fallback (lower quality but reliable)
        preview = post_data.get("preview", {})
        if preview:
            try:
                images = preview.get("images", [])
                if images:
                    source = images[0].get("source", {})
                    preview_url = source.get("url", "")
                    return preview_url.replace("&amp;", "&")
            except Exception as e:
                logger.debug(f"Failed to get preview for {post_data.get('id')}: {e}")
        
        return None
    
    async def fetch_subreddit(
        self, 
        session: aiohttp.ClientSession, 
        subreddit_config: SubredditConfig,
        dedup_checker: Optional["DuplicateChecker"] = None
    ) -> list[dict[str, Any]]:
        """
        Fetch top posts from a subreddit using OAuth API with pagination (Deep Fetch).
        """
        posts = []
        after_token = None
        max_pages = 10  # Safety limit to prevent infinite loops
        
        # Get OAuth token if not already available
        if not self.access_token:
            self.access_token = await self._get_access_token(session)
            if not self.access_token:
                logger.warning(f"Skipping r/{subreddit_config.name} - no OAuth token")
                return []
        
        url = f"{self.API_URL}/r/{subreddit_config.name}/top"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": self.user_agent
        }
        
        logger.info(
            f"Fetching from r/{subreddit_config.name}: "
            f"Target {subreddit_config.fetch_count} NEW posts"
        )
        
        for page in range(max_pages):
            # Stop if we have enough candidates
            if len(posts) >= subreddit_config.fetch_count:
                break
                
            params = {
                "t": "month",
                "limit": 100,
            }
            if after_token:
                params["after"] = after_token
            
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 401:
                        # Token expired loop handling could be here, but simplifying for now
                        logger.warning("Reddit OAuth token expired")
                        break
                    elif response.status == 429:
                        logger.warning(f"Rate limited by Reddit on r/{subreddit_config.name}")
                        break
                    elif response.status != 200:
                        logger.error(f"Reddit error {response.status}")
                        break
                    
                    data = await response.json()
                    response_data = data.get("data", {})
                    children = response_data.get("children", [])
                    after_token = response_data.get("after")
                    
                    if not children:
                        break
                    
                    skipped_dupes = 0
                    
                    for child in children:
                        post = child.get("data", {})
                        
                        # Dedup Check BEFORE processing
                        post_id = post.get("id")
                        full_id = f"reddit_{post_id}"
                        
                        if dedup_checker:
                            is_dup, _ = dedup_checker.check_id(full_id)
                            if is_dup:
                                skipped_dupes += 1
                                continue

                        # Skip if below upvote threshold
                        if post.get("score", 0) < subreddit_config.min_upvotes:
                            continue
                        
                        # Skip deleted/removed/NSFW
                        if (post.get("removed_by_category") or 
                            post.get("selftext") == "[deleted]" or 
                            post.get("over_18")):
                            continue
                        
                        # Extract image URL
                        image_url = self._extract_image_url(post)
                        if not image_url:
                            continue
                            
                        # Check URL duplicate if possible
                        if dedup_checker:
                            is_dup, _ = dedup_checker.check_url(image_url)
                            if is_dup:
                                skipped_dupes += 1
                                continue
                        
                        posts.append({
                            "id": post_id,
                            "title": post.get("title"),
                            "subreddit": subreddit_config.name,
                            "upvotes": post.get("score"),
                            "author": post.get("author", "[deleted]"),
                            "post_url": f"https://reddit.com{post.get('permalink')}",
                            "image_url": image_url,
                            "created_utc": post.get("created_utc"),
                        })
                        
                        if len(posts) >= subreddit_config.fetch_count:
                            break
                    
                    logger.info(f"  Page {page+1}: Found {len(children)} posts, {skipped_dupes} duplicates skipped. Total new: {len(posts)}")
                    
                    if not after_token:
                        break
                        
                    # Be nice to Reddit
                    await asyncio.sleep(1.0)
            
            except Exception as e:
                logger.error(f"Error fetching page {page} from r/{subreddit_config.name}: {e}")
                break
        
        return posts
    
    async def _fetch_via_rss(self, session: aiohttp.ClientSession, subreddit_config: SubredditConfig) -> list[dict[str, Any]]:
        """
        Fallback: Fetch posts using public RSS feed (no auth required).
        
        RSS feeds have limitations:
        - Only top 25 posts per request
        - No upvote filtering (done client-side)
        - Less metadata available
        """
        posts = []
        rss_url = f"https://www.reddit.com/r/{subreddit_config.name}/top.rss"
        params = {"t": "month", "limit": 100}
        
        headers = {"User-Agent": self.user_agent}
        
        try:
            async with session.get(rss_url, headers=headers, params=params) as response:
                if response.status != 200:
                    logger.warning(f"RSS feed failed for r/{subreddit_config.name}: {response.status}")
                    return []
                
                text = await response.text()
                
                # Simple XML parsing for RSS entries
                import xml.etree.ElementTree as ET
                root = ET.fromstring(text)
                
                # Handle Atom namespace
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                entries = root.findall('.//atom:entry', ns)
                
                for entry in entries:
                    try:
                        title = entry.find('atom:title', ns)
                        link = entry.find('atom:link', ns)
                        author = entry.find('atom:author/atom:name', ns)
                        content = entry.find('atom:content', ns)
                        entry_id = entry.find('atom:id', ns)
                        
                        if content is not None:
                            content_text = content.text or ""
                            # Extract image URL from content
                            img_match = re.search(r'href="(https://[^"]+\.(?:jpg|jpeg|png|webp|gif))"', content_text, re.IGNORECASE)
                            if not img_match:
                                img_match = re.search(r'src="(https://[^"]+\.(?:jpg|jpeg|png|webp|gif))"', content_text, re.IGNORECASE)
                            if not img_match:
                                # Try i.redd.it links
                                img_match = re.search(r'href="(https://i\.redd\.it/[^"]+)"', content_text)
                            
                            if img_match:
                                image_url = img_match.group(1)
                                post_id = entry_id.text.split('/')[-1] if entry_id is not None else str(len(posts))
                                
                                posts.append({
                                    "id": post_id,
                                    "title": title.text if title is not None else "Untitled",
                                    "subreddit": subreddit_config.name,
                                    "upvotes": 0,  # Not available in RSS
                                    "author": author.text.replace("/u/", "") if author is not None else "[unknown]",
                                    "post_url": link.get('href') if link is not None else "",
                                    "image_url": image_url,
                                    "created_utc": None,
                                })
                                
                                if len(posts) >= subreddit_config.fetch_count:
                                    break
                    except Exception as e:
                        logger.debug(f"Failed to parse RSS entry: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"RSS fetch error for r/{subreddit_config.name}: {e}")
        
        logger.info(f"Fetched {len(posts)} posts from r/{subreddit_config.name} via RSS")
        return posts
    
    async def fetch_all(
        self, 
        session: aiohttp.ClientSession,
        dedup_checker: Optional["DuplicateChecker"] = None
    ) -> list[dict[str, Any]]:
        """Fetch from all configured subreddits. Uses OAuth if available, RSS as fallback."""
        all_posts = []
        use_rss = False
        
        # Check if we have OAuth credentials
        if not self.client_id or not self.client_secret:
            logger.warning("No Reddit OAuth credentials - using RSS feeds (no upvote filtering)")
            use_rss = True
        
        for subreddit_config in self.config.subreddits:
            if use_rss:
                posts = await self._fetch_via_rss(session, subreddit_config)
            else:
                posts = await self.fetch_subreddit(session, subreddit_config, dedup_checker)
                # If OAuth fails, switch to RSS for remaining subreddits
                if len(posts) == 0 and not self.access_token:
                    logger.warning("OAuth failed - switching to RSS feeds for remaining subreddits")
                    use_rss = True
                    posts = await self._fetch_via_rss(session, subreddit_config)
            
            all_posts.extend(posts)
            # Be nice to Reddit's servers
            await asyncio.sleep(self.config.request_delay)
        
        return all_posts


# =============================================================================
# UNSPLASH API FETCHING
# =============================================================================

class UnsplashFetcher:
    """Fetches wallpaper candidates from Unsplash curated collections."""
    
    BASE_URL = "https://api.unsplash.com"
    
    def __init__(self, config: Config):
        self.config = config
        self.headers = {
            "Authorization": f"Client-ID {config.unsplash_access_key}",
            "Accept-Version": "v1",
        }
    
    async def fetch_curated_photos(
        self, 
        session: aiohttp.ClientSession,
        dedup_checker: Optional["DuplicateChecker"] = None
    ) -> list[dict[str, Any]]:
        """
        Fetch wallpaper-friendly photos from Unsplash using search endpoint.
        Uses landscape orientation and nature/scenery queries to avoid portraits.
        """
        photos = []
        
        # Use search endpoint with wallpaper-friendly queries
        try:
            # We will iterate until we hit the count or run out of pages
            page = 1
            max_pages = 10
            
            while len(photos) < self.config.unsplash_count and page <= max_pages:
                search_url = f"{self.BASE_URL}/search/photos"
                params = {
                    "query": self.config.unsplash_search_query,
                    "per_page": 30,  # Fetch a batch
                    "orientation": self.config.image_orientation,  # landscape/portrait/squarish
                    "page": page
                }
                
                logger.info(f"Fetching Unsplash page {page} (query: '{self.config.unsplash_search_query}', orientation: {self.config.image_orientation})...")
                
                async with session.get(search_url, headers=self.headers, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch Unsplash photos: {response.status}")
                        break
                        
                    data = await response.json()
                    batch = data.get("results", [])  # Search results are under "results" key
                    if not batch:
                        break
                    
                    skipped = 0
                    for photo in batch:
                        # Dedup check
                        photo_id = photo.get("id")
                        full_id = f"unsplash_{photo_id}"
                        
                        if dedup_checker:
                            is_dup, _ = dedup_checker.check_id(full_id)
                            if is_dup:
                                skipped += 1
                                continue
                        
                        photos.append(self._parse_photo(photo))
                        if len(photos) >= self.config.unsplash_count:
                            break
                    
                    logger.info(f"  Unsplash page {page}: {len(batch)} items, {skipped} skipped. Total: {len(photos)}")
                    page += 1
                    await asyncio.sleep(self.config.request_delay)
        
        except Exception as e:
            logger.error(f"Error fetching from Unsplash: {e}")
        
        logger.info(f"Fetched {len(photos)} wallpaper-friendly photos from Unsplash")
        return photos[:self.config.unsplash_count]
    
    def _parse_photo(self, photo: dict[str, Any]) -> dict[str, Any]:
        """Parse a photo response into our standard format."""
        user = photo.get("user", {})
        urls = photo.get("urls", {})
        
        return {
            "id": photo.get("id"),
            "photographer": user.get("name", "Unknown"),
            "photographer_url": user.get("links", {}).get("html", ""),
            "photo_url": urls.get("full") or urls.get("raw", ""),
            "download_url": photo.get("links", {}).get("download_location", ""),
            "thumb_url": urls.get("thumb", ""),
            "description": photo.get("description") or photo.get("alt_description", ""),
            "width": photo.get("width"),
            "height": photo.get("height"),
            "license": "Unsplash License",
            "unsplash_url": photo.get("links", {}).get("html", ""),
        }


# =============================================================================
# PEXELS API FETCHING
# =============================================================================

class PexelsFetcher:
    """Fetches wallpaper candidates from Pexels curated endpoint."""
    
    BASE_URL = "https://api.pexels.com/v1"
    
    def __init__(self, config: Config):
        self.config = config
        self.headers = {
            "Authorization": config.pexels_api_key,
        }
    
    async def fetch_curated_photos(
        self, 
        session: aiohttp.ClientSession,
        dedup_checker: Optional["DuplicateChecker"] = None
    ) -> list[dict[str, Any]]:
        """
        Fetch wallpaper-friendly photos from Pexels using search endpoint.
        Uses landscape orientation and nature/scenery queries to avoid portraits.
        """
        photos = []
        page = 1
        max_pages = 10
        
        try:
            while len(photos) < self.config.pexels_count and page <= max_pages:
                search_url = f"{self.BASE_URL}/search"
                params = {
                    "query": self.config.pexels_search_query,
                    "per_page": 40,
                    "orientation": self.config.image_orientation,  # landscape/portrait/square
                    "page": page
                }
                
                logger.info(f"Fetching Pexels page {page} (query: '{self.config.pexels_search_query}', orientation: {self.config.image_orientation})...")
                
                async with session.get(search_url, headers=self.headers, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch from Pexels: {response.status}")
                        break
                        
                    data = await response.json()
                    batch = data.get("photos", [])
                    if not batch:
                        break
                        
                    skipped = 0
                    for photo in batch:
                        photo_id = str(photo.get("id"))
                        full_id = f"pexels_{photo_id}"
                        
                        if dedup_checker:
                            is_dup, _ = dedup_checker.check_id(full_id)
                            if is_dup:
                                skipped += 1
                                continue
                        
                        photos.append(self._parse_photo(photo))
                        if len(photos) >= self.config.pexels_count:
                            break
                    
                    logger.info(f"  Pexels page {page}: {len(batch)} items, {skipped} skipped. Total: {len(photos)}")
                    page += 1
                    await asyncio.sleep(self.config.request_delay)
        
        except Exception as e:
            logger.error(f"Error fetching from Pexels: {e}")
        
        logger.info(f"Fetched {len(photos)} wallpaper-friendly photos from Pexels")
        return photos
    
    def _parse_photo(self, photo: dict[str, Any]) -> dict[str, Any]:
        """Parse a photo response into our standard format."""
        src = photo.get("src", {})
        
        return {
            "id": str(photo.get("id")),
            "photographer": photo.get("photographer", "Unknown"),
            "photographer_url": photo.get("photographer_url", ""),
            "photo_url": src.get("original") or src.get("large2x", ""),
            "pexels_url": photo.get("url", ""),
            "width": photo.get("width"),
            "height": photo.get("height"),
            "license": "Pexels License",
        }


# =============================================================================
# IMAGE DOWNLOADER
# =============================================================================

class ImageDownloader:
    """Handles downloading images with retry logic and error handling."""
    
    def __init__(self, config: Config):
        self.config = config
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, exceptions=(aiohttp.ClientError, asyncio.TimeoutError))
    async def download_image(
        self,
        session: aiohttp.ClientSession,
        url: str,
        filepath: Path,
    ) -> bool:
        """
        Download an image from URL to the specified filepath.
        
        Args:
            session: aiohttp client session.
            url: Direct URL to the image.
            filepath: Destination path for the downloaded image.
        
        Returns:
            True if download successful, False otherwise.
        """
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with session.get(url, timeout=timeout) as response:
            if response.status != 200:
                logger.debug(f"Failed to download {url}: HTTP {response.status}")
                return False
            
            content_type = response.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                logger.debug(f"Not an image: {url} (Content-Type: {content_type})")
                return False
            
            # Ensure proper extension
            ext = get_file_extension(url, content_type)
            if not filepath.suffix:
                filepath = filepath.with_suffix(ext)
            
            # Download in chunks
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
            
            logger.debug(f"Downloaded: {filepath.name}")
            return True


# =============================================================================
# MAIN PIPELINE
# =============================================================================

async def fetch_candidates(
    config: Config,
    test_mode: bool = False,
    dedup_checker: Optional["DuplicateChecker"] = None
) -> list[CandidateWallpaper]:
    """Fetch candidates from all sources."""
    
    # In test mode, reduce fetch counts drastically
    if test_mode:
        logger.info("🧪 TEST MODE: Using reduced fetch counts")
        # Configure test limits
        for sub in config.subreddits:
            sub.fetch_count = 5
        config.unsplash_count = 2
        config.pexels_count = 3
    
    candidates = []
    downloader = ImageDownloader(config)

    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=300)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 1. Reddit
        if config.reddit_client_id and config.reddit_client_secret:
            logger.info("\n📱 Fetching from Reddit...")
            reddit_fetcher = RedditFetcher(config)
            reddit_posts = await reddit_fetcher.fetch_all(session, dedup_checker)
            logger.info(f"Processing {len(reddit_posts)} Reddit posts...")
            
            for post in reddit_posts:
                try:
                    ext = get_file_extension(post["image_url"])
                    filename = f"reddit_{post['subreddit']}_{post['id']}{ext}"
                    filepath = config.candidates_dir / filename
                    
                    success = await downloader.download_image(session, post["image_url"], filepath)
                    
                    if success:
                        candidates.append(CandidateWallpaper(
                            id=f"reddit_{post['id']}",
                            source="reddit",
                            filepath=filepath,
                            url=post["image_url"],
                            title=post["title"],
                            artist=post["author"],
                            metadata={
                                "subreddit": post["subreddit"],
                                "upvotes": post["upvotes"],
                                "post_url": post["post_url"],
                                "created_utc": post["created_utc"],
                            },
                        ))
                    await asyncio.sleep(config.request_delay)
                except Exception as e:
                    logger.warning(f"Failed to process Reddit post {post['id']}: {e}")
                    continue
        else:
            logger.warning("Skipping Reddit: missing API keys")

        # 2. Unsplash
        if config.unsplash_access_key:
            logger.info("\n📷 Fetching from Unsplash...")
            unsplash_fetcher = UnsplashFetcher(config)
            unsplash_photos = await unsplash_fetcher.fetch_curated_photos(session, dedup_checker)
            logger.info(f"Processing {len(unsplash_photos)} Unsplash photos...")
            
            for photo in unsplash_photos:
                try:
                    ext = get_file_extension(photo["photo_url"])
                    filename = f"unsplash_{photo['id']}{ext}"
                    filepath = config.candidates_dir / filename
                    
                    success = await downloader.download_image(session, photo["photo_url"], filepath)
                    
                    if success:
                        candidates.append(CandidateWallpaper(
                            id=f"unsplash_{photo['id']}",
                            source="unsplash",
                            filepath=filepath,
                            url=photo["photo_url"],
                            title=photo.get("description", "Untitled"),
                            artist=photo["photographer"],
                            metadata={
                                "photographer_url": photo["photographer_url"],
                                "unsplash_url": photo.get("unsplash_url", ""),
                                "download_url": photo.get("download_url", ""),
                                "width": photo.get("width"),
                                "height": photo.get("height"),
                                "license": photo["license"],
                            },
                        ))
                    await asyncio.sleep(config.request_delay)
                except Exception as e:
                    logger.warning(f"Failed to process Unsplash photo {photo['id']}: {e}")
                    continue
        else:
            logger.warning("Skipping Unsplash: missing API key")

        # 3. Pexels
        if config.pexels_api_key:
            logger.info("\n🖼️ Fetching from Pexels...")
            pexels_fetcher = PexelsFetcher(config)
            pexels_photos = await pexels_fetcher.fetch_curated_photos(session, dedup_checker)
            logger.info(f"Processing {len(pexels_photos)} Pexels photos...")
            
            for photo in pexels_photos:
                try:
                    ext = get_file_extension(photo["photo_url"])
                    filename = f"pexels_{photo['id']}{ext}"
                    filepath = config.candidates_dir / filename
                    
                    success = await downloader.download_image(session, photo["photo_url"], filepath)
                    
                    if success:
                        candidates.append(CandidateWallpaper(
                            id=f"pexels_{photo['id']}",
                            source="pexels",
                            filepath=filepath,
                            url=photo["photo_url"],
                            title=f"Photo by {photo['photographer']}",
                            artist=photo["photographer"],
                            metadata={
                                "photographer_url": photo["photographer_url"],
                                "pexels_url": photo.get("pexels_url", ""),
                                "width": photo.get("width"),
                                "height": photo.get("height"),
                                "license": photo["license"],
                            },
                        ))
                    await asyncio.sleep(config.request_delay)
                except Exception as e:
                    logger.warning(f"Failed to process Pexels photo {photo['id']}: {e}")
                    continue
        else:
            logger.warning("Skipping Pexels: missing API key")
    
    return candidates


async def main(test_mode: bool = False, dedup_checker: Optional["DuplicateChecker"] = None) -> list[CandidateWallpaper]:
    """
    Main entry point for the wallpaper curation pipeline Part 1.
    
    Args:
        test_mode: If True, fetch only 10 wallpapers (5 Reddit, 2 Unsplash, 3 Pexels)
        dedup_checker: An optional DuplicateChecker instance to prevent fetching duplicates.
    
    Returns:
        List of all CandidateWallpaper objects fetched and downloaded.
    """
    logger.info("=" * 60)
    logger.info("RAMEN Wallpaper Curation Pipeline - Part 1: Fetching")
    logger.info("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.warning("Configuration warnings:")
        for error in errors:
            logger.warning(f"  - {error}")
        logger.warning("Some sources may not be fetched due to missing API keys.")
    
    # Create directories
    create_directories(config)
    
    
    # 1. Fetch candidates
    all_candidates = await fetch_candidates(config, test_mode, dedup_checker)
    
    # Summary logging (count by source)
    reddit_count = sum(1 for c in all_candidates if c.source == 'reddit')
    unsplash_count = sum(1 for c in all_candidates if c.source == 'unsplash')
    pexels_count = sum(1 for c in all_candidates if c.source == 'pexels')
    
    logger.info("\n" + "=" * 60)
    logger.info("FETCHING COMPLETE - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Reddit:   {reddit_count} candidates")
    logger.info(f"  Unsplash: {unsplash_count} candidates")
    logger.info(f"  Pexels:   {pexels_count} candidates")
    logger.info("-" * 60)
    logger.info(f"  TOTAL:    {len(all_candidates)} candidates downloaded")
    logger.info(f"  Location: {config.candidates_dir.absolute()}")
    logger.info("=" * 60)
    
    return all_candidates


if __name__ == "__main__":
    candidates = asyncio.run(main())
    print(f"\nPart 1 complete. Fetched {len(candidates)} wallpaper candidates.")
