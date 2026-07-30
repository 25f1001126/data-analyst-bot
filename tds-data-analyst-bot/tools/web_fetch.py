"""Fetch a URL (HTML/CSV/JSON) and hand back truncated text for the LLM to read."""
import requests
from bs4 import BeautifulSoup

TIMEOUT_SECONDS = 20
MAX_CHARS = 12000


def _looks_like_html(content_type: str, text: str) -> bool:
    return "html" in content_type.lower() or text.lstrip().lower().startswith(("<!doctype", "<html"))


def fetch_url(url: str) -> dict:
    try:
        resp = requests.get(url, timeout=TIMEOUT_SECONDS, headers={"User-Agent": "data-analyst-bot/1.0"})
        resp.raise_for_status()
        raw_text = resp.text
        content_type = resp.headers.get("Content-Type", "")

        if _looks_like_html(content_type, raw_text):
            soup = BeautifulSoup(raw_text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
        else:
            text = raw_text  # CSV/JSON/plain text: no HTML boilerplate to strip

        return {
            "success": True,
            "status_code": resp.status_code,
            "content": text[:MAX_CHARS],
            "truncated": len(text) > MAX_CHARS,
        }
    except requests.RequestException as e:
        return {"success": False, "error": str(e)}
