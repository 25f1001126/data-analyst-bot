"""Fetch a URL (HTML/CSV/JSON) and hand back truncated text for the LLM to read."""
import requests

TIMEOUT_SECONDS = 20
MAX_CHARS = 8000


def fetch_url(url: str) -> dict:
    try:
        resp = requests.get(url, timeout=TIMEOUT_SECONDS, headers={"User-Agent": "data-analyst-bot/1.0"})
        resp.raise_for_status()
        text = resp.text
        return {
            "success": True,
            "status_code": resp.status_code,
            "content": text[:MAX_CHARS],
            "truncated": len(text) > MAX_CHARS,
        }
    except requests.RequestException as e:
        return {"success": False, "error": str(e)}
