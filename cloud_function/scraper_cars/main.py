# cloud_function/main.py
# Per-listing scraper: saves ALL visible text from each car listing page.
import os, io, time, datetime as dt, requests, re, csv
from typing import List
from bs4 import BeautifulSoup
from google.cloud import storage
from flask import Request, jsonify

# ---- Config (overridable via env vars in deploy.yml) ----
BUCKET_NAME        = os.environ["BUCKET_NAME"]
BASE_SITE          = os.environ.get("BASE_SITE", "https://newhaven.craigslist.org")
# Comma-separated list of sites to scrape (e.g., "newhaven,hartford,boston")
SITES_TO_SCRAPE    = os.environ.get("SITES_TO_SCRAPE", "newhaven")
SEARCH_PATH        = os.environ.get("SEARCH_PATH", "/search/cta")   # cars+trucks
MAX_PAGES          = int(os.environ.get("MAX_PAGES", "1"))          # search pages to scan
MAX_ITEMS_PER_RUN  = int(os.environ.get("MAX_ITEMS_PER_RUN", "50")) # safety cap per run (per site)
DELAY_SECS         = float(os.environ.get("DELAY_SECS", "1.0"))     # polite delay between requests
USER_AGENT         = os.environ.get("USER_AGENT", "UConn-OPIM-Student-Scraper/1.0")

HDRS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"}

# -- Helpers -------------------------------------------------------------------

def _site_to_url(site: str) -> str:
    """Convert a city name or URL to a full Craigslist base URL.
       Examples: 'newhaven' -> 'https://newhaven.craigslist.org'
                 'https://hartford.craigslist.org' -> 'https://hartford.craigslist.org'
    """
    site = site.strip()
    if site.startswith("http"):
        return site
    return f"https://{site}.craigslist.org"

def _page_url(base: str, path: str, page: int) -> str:
    # Craigslist uses s=<offset> with 120 results/page
    if page == 0:
        return f"{base}{path}?hasPic=1&srchType=T"
    return f"{base}{path}?hasPic=1&srchType=T&s={page*120}"

import re
POST_PAGE_RE = re.compile(r"/(\d+)\.html?$")

def _extract_listing_links(html: str, base_site: str = None) -> list[str]:
    """Return absolute URLs to individual listings from a search results page.
       Handles classic/new layouts and falls back to regex if needed.

       Args:
           html: HTML content of the search results page
           base_site: Base URL for the site (e.g., 'https://newhaven.craigslist.org')
    """
    if base_site is None:
        base_site = BASE_SITE

    soup = BeautifulSoup(html, "html.parser")
    links = set()

    # Classic layout
    for a in soup.select("a.result-title, a.result-title.hdrlnk"):
        href = a.get("href")
        if href: links.add(href)

    # Newer layout (cl-static / cl-search)
    for a in soup.select("li.cl-search-result a.titlestring"):
        href = a.get("href")
        if href: links.add(href)

    # Fallback: any anchor that looks like a posting
    for a in soup.select("li.cl-search-result a, .result-row a, a[href$='.html']"):
        href = a.get("href")
        if href and POST_PAGE_RE.search(href):
            links.add(href)

    # Final fallback: regex scan of raw HTML
    # matches absolute or relative post URLs ending with /<post_id>.html
    for m in re.findall(r'href="([^"]+?/\d+\.html)"', html):
        links.add(m)

    # Normalize to absolute
    abs_links = []
    for href in links:
        if href.startswith("//"):
            abs_links.append(f"https:{href}")
        elif href.startswith("/"):
            abs_links.append(f"{base_site}{href}")
        else:
            abs_links.append(href)

    # keep only post pages (…/<post_id>.html)
    abs_links = [u for u in abs_links if POST_PAGE_RE.search(u)]
    return abs_links


POST_ID_RE = re.compile(r"/(\d+)\.html?$")

def _post_id_from_url(url: str) -> str:
    m = POST_ID_RE.search(url)
    return m.group(1) if m else ""

def _visible_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    raw = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln and not ln.isspace()]
    dedup = []
    for ln in lines:
        if not dedup or ln != dedup[-1]:
            dedup.append(ln)
    return "\n".join(dedup) + "\n"

def _upload_text(bucket: str, object_name: str, text: str):
    storage.Client().bucket(bucket).blob(object_name)\
        .upload_from_string(text, content_type="text/plain")

def _upload_csv(bucket: str, object_name: str, rows: List[dict], header: List[str]):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
    storage.Client().bucket(bucket).blob(object_name)\
        .upload_from_string(buf.getvalue(), content_type="text/csv")

# -- Entry point ----------------------------------------------------------------

def entrypoint(request: Request):
    """HTTP GET. Optional query overrides:
       ?pages=2&max=40&sites=newhaven,hartford,boston&path=/search/cta

       New multi-site support:
       - Use 'sites' parameter with comma-separated city names (e.g., sites=newhaven,hartford,boston)
       - Or use SITES_TO_SCRAPE environment variable
       - Each site is scraped independently and results are combined
    """
    pages = min(MAX_PAGES, int(request.args.get("pages", MAX_PAGES)))
    max_items = min(MAX_ITEMS_PER_RUN, int(request.args.get("max", MAX_ITEMS_PER_RUN)))
    path = request.args.get("path", SEARCH_PATH)

    # Get sites to scrape (comma-separated)
    sites_str = request.args.get("sites", SITES_TO_SCRAPE)
    sites = [s.strip() for s in sites_str.split(",") if s.strip()]

    # Fallback to BASE_SITE if no sites specified
    if not sites:
        sites = [BASE_SITE]

    # Convert site names to full URLs
    site_urls = [_site_to_url(site) for site in sites]

    # 1) Build run folder: YYYYMMDDHHMMSS (UTC)
    run_id = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    run_prefix = f"scrapes/{run_id}"

    # Track results per site
    all_index_rows = []
    site_stats = []
    global_seen_pids = set()  # Prevent duplicates across sites

    # 2) Loop through each site
    for site_url in site_urls:
        site_name = site_url.replace("https://", "").replace(".craigslist.org", "")

        # Collect listing links from search pages for this site
        listing_urls = []
        for p in range(pages):
            url = _page_url(site_url, path, p)
            try:
                r = requests.get(url, headers=HDRS, timeout=25)
                r.raise_for_status()
                listing_urls.extend(_extract_listing_links(r.text, site_url))
                if p < pages - 1:
                    time.sleep(DELAY_SECS)
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                continue

        # 3) Deduplicate + cap to max_items per site
        seen = set()
        urls = []
        for u in listing_urls:
            pid = _post_id_from_url(u)
            if pid and pid not in seen and pid not in global_seen_pids:
                seen.add(pid)
                global_seen_pids.add(pid)
                urls.append((pid, u))
            if len(urls) >= max_items:
                break

        # 4) Fetch each listing page and write one TXT per listing
        site_index_rows = []
        for i, (pid, u) in enumerate(urls, start=1):
            try:
                r = requests.get(u, headers=HDRS, timeout=25)
                r.raise_for_status()
                text = _visible_text_from_html(r.text)
                obj = f"{run_prefix}/{site_name}/{pid}.txt"
                _upload_text(BUCKET_NAME, obj, text)
                site_index_rows.append({
                    "site": site_name,
                    "post_id": pid,
                    "url": u,
                    "object": obj
                })
                if i < len(urls):
                    time.sleep(DELAY_SECS)
            except Exception as e:
                # record failure in index for transparency
                site_index_rows.append({
                    "site": site_name,
                    "post_id": pid,
                    "url": u,
                    "object": "",
                    "error": str(e)
                })

        all_index_rows.extend(site_index_rows)

        # Track stats for this site
        site_stats.append({
            "site": site_name,
            "candidates_found": len(listing_urls),
            "items_attempted": len(urls),
            "items_saved": len([r for r in site_index_rows if r.get("object")])
        })

    # 5) Write a combined index.csv for the entire run
    if all_index_rows:
        header = sorted(all_index_rows[0].keys())
        _upload_csv(BUCKET_NAME, f"{run_prefix}/index.csv", all_index_rows, header)

    return jsonify({
        "ok": True,
        "run_id": run_id,
        "sites_scraped": len(site_urls),
        "pages_per_site": pages,
        "total_items_attempted": len(all_index_rows),
        "saved_prefix": run_prefix,
        "site_details": site_stats
    })
