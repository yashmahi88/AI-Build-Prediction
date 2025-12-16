"""Yocto documentation scraping service"""  # Module docstring describing this file fetches and chunks official Yocto Project documentation from the web
import logging  # Standard Python logging library for tracking scraping operations
import time  # Time library for tracking scrape intervals and timestamps
from typing import List  # Type hint for function return types: List for arrays


import requests  # HTTP library for fetching web pages
from bs4 import BeautifulSoup  # HTML parsing library for extracting text from web pages
from langchain_core.documents import Document  # LangChain's Document class representing text chunks with metadata


from app.core.config import get_settings  # Function to load application configuration (scraping settings)
from app.utils.text_processing import create_confluence_splitter  # Utility to create text splitter for chunking documentation



logger = logging.getLogger(__name__)  # Create logger instance for this module to output scraping-related logs
settings = get_settings()  # Load application settings once at module level (cached for reuse)



YOCTO_DOC_URLS = [  # List of official Yocto Project documentation URLs to scrape
    "https://docs.yoctoproject.org/current/ref-manual/",  # Reference manual (comprehensive variable reference, classes, tasks)
    "https://docs.yoctoproject.org/current/dev-manual/",  # Developer manual (guides for building and customizing)
    "https://docs.yoctoproject.org/current/bitbake-user-manual/",  # BitBake user manual (build system documentation)
    "https://docs.yoctoproject.org/migration-guides/index.html",  # Migration guides (version upgrade information)
]




_last_fetch_ts: float = 0.0  # Global variable tracking timestamp of last successful scrape (0.0 means never scraped)



def _should_scrape() -> bool:  # Helper function to determine if scraping should occur based on config and time interval
    """Check config + interval to decide if we scrape Yocto docs."""  # Docstring explaining this checks if scraping is enabled and enough time has passed
    global _last_fetch_ts  # Declare that we'll read (and potentially use) the global timestamp variable


    # Match field names in Settings (config.py)
    if not getattr(settings, "yocto_doc_scrape_enabled", False):  # Check if scraping is enabled in config (getattr provides default False if setting doesn't exist)
        logger.info("Yocto doc scraping disabled via config")  # Log that scraping is turned off
        return False  # Don't scrape if disabled in configuration


    interval_hours = getattr(settings, "yocto_doc_update_interval_hours", 24)  # Get configured scrape interval in hours (default 24 hours if not set)
    interval_sec = interval_hours * 3600  # Convert interval from hours to seconds (3600 seconds per hour)


    if _last_fetch_ts == 0:  # Check if we've never scraped before (initial run)
        return True  # Always scrape on first run regardless of interval


    return (time.time() - _last_fetch_ts) > interval_sec  # Return True if enough time has passed since last scrape (current time minus last scrape time exceeds interval)



def scrape_yocto_docs() -> List[Document]:  # Main function to fetch official Yocto documentation from web and return as chunked Document objects
    """Fetch official Yocto docs from the web and return as Documents."""  # Docstring explaining this scrapes, cleans, and chunks Yocto documentation
    global _last_fetch_ts  # Declare that we'll modify the global timestamp variable


    if not _should_scrape():  # Check if scraping should be skipped (disabled or interval not elapsed)
        logger.info("Yocto docs scrape skipped (interval not elapsed or disabled)")  # Log why scraping was skipped
        return []  # Return empty list without scraping


    logger.info("Scraping Yocto documentation...")  # Log start of scraping operation
    splitter = create_confluence_splitter()  # Create text splitter instance for chunking documentation (uses RecursiveCharacterTextSplitter with appropriate chunk size)
    documents: List[Document] = []  # Initialize empty list to collect all document chunks from all URLs


    headers = {"User-Agent": "Mozilla/5.0 (Yocto-Build-Analyzer)"}  # Set User-Agent header to identify our scraper (some sites block requests without User-Agent)


    for url in YOCTO_DOC_URLS:  # Loop through each Yocto documentation URL
        try:  # Wrap scraping in try-except to handle network errors and continue with other URLs
            resp = requests.get(url, timeout=30, headers=headers)  # Fetch web page with 30-second timeout and custom headers (prevents hanging on slow servers)
            resp.raise_for_status()  # Raise exception if HTTP status code indicates error (4xx or 5xx)


            soup = BeautifulSoup(resp.text, "html.parser")  # Parse HTML response into BeautifulSoup object for easy navigation and text extraction
            for tag in soup(["script", "style", "nav", "footer", "header"]):  # Find all script, style, nav, footer, and header tags (non-content elements)
                tag.decompose()  # Remove these tags from the parsed HTML tree (cleans up navigation, styling, and JavaScript)


            text = soup.get_text(separator="\n", strip=True)  # Extract all text content from cleaned HTML (separator="\n" adds newlines between elements, strip=True removes extra whitespace)
            chunks = splitter.split_text(text)  # Split extracted text into smaller chunks using configured splitter (makes chunks suitable for embedding and retrieval)


            for chunk in chunks:  # Loop through each text chunk
                documents.append(  # Add new Document object to documents list
                    Document(  # Create LangChain Document object
                        page_content=chunk,  # Store text chunk as page content
                        metadata={  # Store metadata dictionary with source information
                            "source": f"yocto-docs:{url}",  # Source identifier with URL (for citation and filtering)
                            "confluenceurl": url,  # Original URL (for linking back to source)
                            "doctype": "yocto-official-docs",  # Document type classifier (distinguishes from user uploads or Jenkins logs)
                            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp when this was scraped (formatted as human-readable date/time string)
                        },
                    )
                )


            logger.info(f"OK Yocto doc processed {url} -> {len(chunks)} chunks")  # Log successful processing with URL and chunk count


        except Exception as e:  # Catch any errors during fetching or parsing this URL
            logger.warning(f"WARN Yocto doc fetch failed {url}: {e}")  # Log warning with URL and error details (but continue with other URLs)


    _last_fetch_ts = time.time()  # Update global timestamp to current time (marks successful scrape completion)
    logger.info(f"OK Added {len(documents)} Yocto documentation chunks")  # Log total number of document chunks collected from all URLs
    return documents  # Return complete list of Document objects ready for vectorstore ingestion
