#!/usr/bin/env python3
"""
sppu_scraper.py

Table-aware scraper for sppuquestionpapers.com.

Features:
- Build dept/semester URL from user input
- Inspect all <table> elements on the page
- Heuristically match table headers to requested subject and pattern
- Extract direct PDF links from matched tables
- Simple CLI for testing: prints matches and writes sppu_result.json

Drop this file into your project and call:
    python sppu_scraper.py --department "Computer Engineering" --semester "Sem 5" --subject "Database Management Systems" --pattern 2019
"""

import re
import json
import argparse
import logging
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urljoin
import difflib

# ---------- Config ----------
USER_AGENT = {"User-Agent": "Mozilla/5.0 (compatible; SPPUScraper/1.0)"}
REQUEST_TIMEOUT = 15  # seconds
LOG_LEVEL = logging.INFO

# Minimal department slug map: extend this as needed
DEPT_SLUGS = {
    "computer engineering": "computer-engineering",
    "cse": "computer-engineering",
    "computer engg": "computer-engineering",
    "mechanical engineering": "mechanical-engineering",
    "mechanical": "mechanical-engineering",
    "information technology": "information-technology",
    "it": "information-technology",
    "civil engineering": "civil-engineering",
    "electrical engineering": "electrical-engineering",
    "electronics and telecommunication": "electronics-telecommunication",
    "entc": "electronics-telecommunication",
}

# ---------- Logging ----------
logger = logging.getLogger("sppu_scraper")
logger.setLevel(LOG_LEVEL)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(handler)


# ---------- Helpers ----------
def normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'[^a-z0-9 ]+', '', text.lower()).strip()


def semester_to_slug(sem_text: str) -> str:
    digits = ''.join(ch for ch in (sem_text or "") if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse semester from '{sem_text}'")
    return f"semester-{digits}"


def build_sppu_url(department: str, semester: str) -> str:
    dept_norm = normalize(department)
    dept_slug = DEPT_SLUGS.get(dept_norm)
    if not dept_slug:
        # fallback: convert spaces to hyphens, remove extra chars
        dept_slug = dept_norm.replace(" ", "-")
        logger.debug(f"No canonical slug found; falling back to '{dept_slug}'")
    sem_slug = semester_to_slug(semester)
    url = f"https://sppuquestionpapers.com/be/{dept_slug}/{sem_slug}/"
    logger.debug(f"Built main URL: {url}")
    return url


# ---------- Table header extraction and scoring ----------
def _get_table_header_text(table: Tag) -> str:
    """Try several places for the human-visible header for a table."""
    # 1) caption
    cap = table.find("caption")
    if cap and cap.get_text(strip=True):
        return cap.get_text(" ", strip=True)

    # 2) thead or first th
    thead = table.find("thead")
    if thead:
        txt = thead.get_text(" ", strip=True)
        if txt:
            return txt
    first_th = table.find("th")
    if first_th and first_th.get_text(strip=True):
        txt = first_th.get_text(" ", strip=True)
        # ignore generic column headers
        if not re.match(r'year|month|link', txt, flags=re.I):
            return txt

    # 3) previous siblings (look back a few tags)
    prev = table.previous_sibling
    steps = 0
    while prev and steps < 6:
        if isinstance(prev, Tag):
            if prev.name in ("h1", "h2", "h3", "h4", "strong", "div", "p"):
                t = prev.get_text(" ", strip=True)
                if t and len(t) > 3:
                    return t
        prev = prev.previous_sibling
        steps += 1

    # 4) look for previous heading in parent
    parent = table.parent
    if parent:
        heading = parent.find_previous(lambda tag: tag.name in ("h1", "h2", "h3", "h4"))
        if heading and heading.get_text(strip=True):
            return heading.get_text(" ", strip=True)

    return ""


def _score_table_match(header_text: str, subject_query: str, pattern_query: str = None) -> float:
    """Return a heuristic score [0..1] how well a table header matches the subject+pattern."""
    if not header_text:
        return 0.0
    h = normalize(header_text)
    s = normalize(subject_query)

    # fuzzy similarity
    subj_score = difflib.SequenceMatcher(None, s, h).ratio()
    if s and s in h:
        subj_score = max(subj_score, 0.85)

    # pattern handling (year)
    pat_score = 0.0
    if pattern_query:
        year_match = re.search(r'((?:19|20)\d{2})', str(pattern_query))
        if year_match:
            y = year_match.group(1)
            if y in h:
                pat_score = 1.0
        else:
            if normalize(pattern_query) in h:
                pat_score = 1.0

    if pattern_query:
        return subj_score * (0.7 + 0.3 * pat_score)
    return subj_score


# ---------- Find matching tables ----------
def find_subject_tables(main_page_url: str, subject: str, pattern: str = None, min_score: float = 0.45):
    """
    Return list of (table_tag, header_text, score) that match the subject (and pattern if given).
    """
    logger.info(f"Fetching main page: {main_page_url}")
    resp = requests.get(main_page_url, headers=USER_AGENT, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    tables = soup.find_all("table")
    logger.debug(f"Found {len(tables)} tables on page")
    matches = []
    for t in tables:
        header_text = _get_table_header_text(t)
        score = _score_table_match(header_text, subject, pattern)
        logger.debug(f"Table header [{header_text}] => score {score:.3f}")
        if score >= min_score:
            matches.append((t, header_text, score))

    matches.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"Matched {len(matches)} table(s) with score >= {min_score}")
    return matches


# ---------- Extract PDFs from a table ----------
def extract_pdfs_from_table(table_tag: Tag, base_url: str) -> List[str]:
    """
    Given a <table> Tag, extract all PDF links found in its rows (column 'Link' expected).
    Returns absolute URLs.
    """
    pdfs: List[str] = []
    for a in table_tag.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            pdfs.append(urljoin(base_url, href))
        else:
            # if anchor text looks like 'Download' but href is not a pdf, skip for speed.
            # (Optionally follow the link in a HEAD/GET to confirm, but avoid heavy IO here.)
            continue

    # dedupe preserving order
    seen = set()
    out = []
    for u in pdfs:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# ---------- Top-level pipeline ----------
def get_papers_from_sppu_tables(department: str, semester: str, subject: str, pattern: str = None,
                                min_score: float = 0.45):
    """
    Main function:
    - builds dept/semester URL
    - finds best matching subject table(s)
    - extracts pdf links from matched tables
    Returns: (pdf_list, list_of_matched_headers, main_page_url, matched_table_urls)
    """
    main_url = build_sppu_url(department, semester)
    matches = find_subject_tables(main_url, subject, pattern, min_score=min_score)

    if not matches:
        return [], [], main_url, []

    all_pdfs = []
    matched_headers = []
    matched_table_urls = []
    for table_tag, header_text, score in matches:
        pdfs = extract_pdfs_from_table(table_tag, main_url)
        if pdfs:
            all_pdfs.extend(pdfs)
            matched_headers.append(header_text)
            matched_table_urls.append(main_url)

    # dedupe PDFs while preserving order
    seen = set()
    unique_pdfs = []
    for p in all_pdfs:
        if p not in seen:
            seen.add(p)
            unique_pdfs.append(p)

    return unique_pdfs, matched_headers, main_url, matched_table_urls


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="SPPU question paper scraper (table-aware).")
    parser.add_argument("--department", "-d", required=True, help="Department (e.g., 'Computer Engineering')")
    parser.add_argument("--semester", "-s", required=True, help="Semester (e.g., 'Sem 5' or '5')")
    parser.add_argument("--subject", "-j", required=True, help="Subject name (e.g., 'Database Management Systems')")
    parser.add_argument("--pattern", "-p", required=False, help="Pattern/year (e.g., 2019)", default=None)
    parser.add_argument("--min-score", type=float, default=0.45, help="Minimum header match score (0..1)")
    parser.add_argument("--out", "-o", default="sppu_result.json", help="JSON output file (summary)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        pdfs, headers, main_page, table_pages = get_papers_from_sppu_tables(
            args.department, args.semester, args.subject, args.pattern, min_score=args.min_score
        )
    except Exception as e:
        logger.error(f"Error while scraping: {e}")
        return

    result = {
        "department": args.department,
        "semester": args.semester,
        "subject": args.subject,
        "pattern": args.pattern,
        "main_page": main_page,
        "matched_headers": headers,
        "matched_table_pages": table_pages,
        "pdf_count": len(pdfs),
        "pdfs": pdfs,
    }

    # pretty print results
    logger.info("=== SCRAPER RESULT ===")
    logger.info(f"Main page searched: {main_page}")
    logger.info(f"Matched headers: {headers}")
    logger.info(f"PDFs found: {len(pdfs)}")
    for i, p in enumerate(pdfs[:50], start=1):
        logger.info(f"{i:02d}. {p}")

    # write JSON
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote result to {args.out}")
    except Exception as e:
        logger.warning(f"Failed to write JSON output: {e}")

    # quick user message
    if not pdfs:
        logger.info("No PDF links found for the matched tables. Consider lowering --min-score or verifying department/subject spelling.")
    else:
        logger.info("Done.")


if __name__ == "__main__":
    main()
