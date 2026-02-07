# Multi-Site Craigslist Scraper - Usage Guide

The scraper has been extended to support scraping car listings from multiple Craigslist areas in a single run.

## Features

- **Multi-site scraping**: Scrape from multiple Craigslist cities in one request
- **Automatic deduplication**: Prevents duplicate listings across sites
- **Site tracking**: Each listing is tagged with its source site
- **Organized storage**: Listings are stored in site-specific folders

## Usage

### Method 1: Query Parameters (HTTP Request)

When calling the cloud function via HTTP, use the `sites` parameter with comma-separated city names:

```
GET https://your-function-url/?sites=newhaven,hartford,boston&pages=2&max=50
```

**Parameters:**
- `sites`: Comma-separated list of Craigslist city names (e.g., `newhaven,hartford,boston,providence`)
- `pages`: Number of search pages to scrape per site (default: 1)
- `max`: Maximum number of listings per site (default: 50)
- `path`: Search path (default: `/search/cta` for cars+trucks)

### Method 2: Environment Variables

Set the `SITES_TO_SCRAPE` environment variable in your deployment configuration:

```yaml
env_variables:
  SITES_TO_SCRAPE: "newhaven,hartford,boston,providence,newyork"
  MAX_PAGES: "2"
  MAX_ITEMS_PER_RUN: "50"
```

## City Names

You can use either:
- **Short names**: `newhaven`, `hartford`, `boston`, `newyork`
- **Full URLs**: `https://newhaven.craigslist.org`

Common Craigslist cities include:
- **Connecticut**: `newhaven`, `hartford`, `newlondon`
- **Massachusetts**: `boston`, `worcester`, `westernmass`
- **New York**: `newyork`, `albany`, `buffalo`, `rochester`
- **Rhode Island**: `providence`
- **Vermont**: `vermont`

## Output Structure

Results are organized by site in Google Cloud Storage:

```
scrapes/
└── 20260207123456/              # Run timestamp
    ├── index.csv                # Combined index for all sites
    ├── newhaven/
    │   ├── 1234567.txt
    │   └── 1234568.txt
    ├── hartford/
    │   ├── 7654321.txt
    │   └── 7654322.txt
    └── boston/
        ├── 8888888.txt
        └── 9999999.txt
```

## Response Format

The function returns JSON with detailed statistics:

```json
{
  "ok": true,
  "run_id": "20260207123456",
  "sites_scraped": 3,
  "pages_per_site": 2,
  "total_items_attempted": 145,
  "saved_prefix": "scrapes/20260207123456",
  "site_details": [
    {
      "site": "newhaven",
      "candidates_found": 67,
      "items_attempted": 50,
      "items_saved": 50
    },
    {
      "site": "hartford",
      "candidates_found": 53,
      "items_attempted": 50,
      "items_saved": 49
    },
    {
      "site": "boston",
      "candidates_found": 102,
      "items_attempted": 45,
      "items_saved": 45
    }
  ]
}
```

## Example Usage

### Scrape from Connecticut cities only
```
?sites=newhaven,hartford,newlondon&pages=1&max=30
```

### Scrape from entire Northeast region
```
?sites=newhaven,hartford,boston,providence,newyork,albany&pages=2&max=50
```

### Scrape with custom search parameters
```
?sites=boston,newyork&pages=3&max=100&path=/search/cta
```

## Notes

- The scraper respects Craigslist's rate limits with configurable delays between requests
- Duplicate listings (same post ID) are automatically filtered across sites
- Each site is scraped independently, so failures in one site don't affect others
- The `MAX_ITEMS_PER_RUN` limit applies per site, not globally
