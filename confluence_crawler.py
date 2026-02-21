"""
confluence_crawler.py
Recursively crawls all pages under a base URL.
Extracts clean text, title, and URL for each page.
"""

import httpx
import time
import json
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from collections import deque


def clean_text(soup: BeautifulSoup) -> str:
    """Remove nav, footer, scripts, styles and return clean text."""
    # Remove unwanted tags
    for tag in soup.find_all(["script", "style", "nav", "footer", "header",
                               "noscript", "iframe", "img", "svg"]):
        tag.decompose()

    # Get text, normalize whitespace
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_page_title(soup: BeautifulSoup) -> str:
    """Extract page title."""
    if soup.title:
        return soup.title.get_text().strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text().strip()
    return "Untitled"


def extract_links(soup: BeautifulSoup, current_url: str, base_url: str) -> list[str]:
    """Extract all links that are under the base URL."""
    links = []
    base_parsed = urlparse(base_url)

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # Resolve relative URLs
        full_url = urljoin(current_url, href)
        # Remove fragments
        full_url = full_url.split("#")[0]
        # Remove trailing slash for consistency
        full_url = full_url.rstrip("/")

        parsed = urlparse(full_url)

        # Must be same domain + path starts with base path
        if (parsed.netloc == base_parsed.netloc and
                parsed.path.startswith(base_parsed.path) and
                full_url not in links):
            links.append(full_url)

    return links


def crawl(base_url: str, max_pages: int = 200, delay: float = 0.5) -> list[dict]:
    """
    Recursively crawl all pages under base_url.

    Returns list of dicts:
    {
        "url": str,
        "title": str,
        "text": str
    }
    """
    base_url = base_url.rstrip("/")
    visited = set()
    queue = deque([base_url])
    results = []

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RAG-Crawler/1.0)"
    }

    print(f"Starting crawl from: {base_url}")
    print(f"Max pages: {max_pages}")

    with httpx.Client(headers=headers, follow_redirects=True, timeout=15) as client:
        while queue and len(results) < max_pages:
            url = queue.popleft()

            if url in visited:
                continue
            visited.add(url)

            try:
                print(f"[{len(results)+1}] Fetching: {url}")
                response = client.get(url)

                if response.status_code != 200:
                    print(f"  ⚠ Status {response.status_code}, skipping")
                    continue

                content_type = response.headers.get("content-type", "")
                if "html" not in content_type:
                    print(f"  ⚠ Not HTML ({content_type}), skipping")
                    continue

                soup = BeautifulSoup(response.text, "html.parser")
                title = get_page_title(soup)
                text = clean_text(soup)

                if len(text) < 100:
                    print(f"  ⚠ Too little content, skipping")
                    continue

                results.append({
                    "url": url,
                    "title": title,
                    "text": text
                })
                print(f"  ✓ '{title}' ({len(text)} chars)")

                # Find and queue new links
                new_links = extract_links(soup, url, base_url)
                for link in new_links:
                    if link not in visited:
                        queue.append(link)

                print(f"  → Found {len(new_links)} new links, queue size: {len(queue)}")

                time.sleep(delay)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

    print(f"\n✅ Crawl complete! {len(results)} pages crawled.")
    return results


def save_crawl_results(results: list[dict], output_file: str = "crawl_results.json"):
    """Save crawl results to JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved {len(results)} pages to {output_file}")


if __name__ == "__main__":
    BASE_URL = "https://www.atlassian.com/software/confluence/resources"
    results = crawl(BASE_URL, max_pages=50, delay=0.5)
    save_crawl_results(results, "crawl_results.json")
