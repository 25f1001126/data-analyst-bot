import json
import os
import re

from llm_client import call_llm
from tools.python_exec import run_python
from tools.web_fetch import fetch_url
from logger import write_log_entry

# The URL your log file is actually reachable at once deployed. We inject this
# ourselves into every answer rather than trusting the model to remember it,
# so a single env var always controls the (correct) log_url in every reply.
PUBLIC_LOG_URL = os.environ.get("PUBLIC_LOG_URL", "http://localhost:8080/run.jsonl")

SYSTEM_PROMPT = """You are a rigorous data-analyst agent answering questions about public datasets \
(MOSPI, data.gov.in, PIB, and similar Indian government statistics, or any dataset described inline \
in the question).

Rules:
1. Never guess a number or fact you have not verified. Use the `python_exec` tool to compute anything \
   quantitative, and the `web_fetch` tool to pull real data from URLs (CSV/HTML/JSON) when the question \
   points at a public dataset instead of embedding the data inline.
2. If the question embeds data directly in the message, use python_exec to parse and compute from that \
   data rather than eyeballing it.
3. Work efficiently: use as few tool calls as you need, but never skip verification for anything numeric.
4. Your FINAL reply must be EXACTLY one JSON object and NOTHING else: no markdown fences, no explanation, \
   no leading/trailing text. Match the exact key names and value shape the question's example JSON shows. \
   You do not need to fill in "log_url" yourself -- leave it as an empty string if unsure; it will be \
   corrected automatically before sending.
5. Only pandas, numpy, requests, json, re, and math/statistics are guaranteed to be installed in \
   python_exec. If code you run raises ModuleNotFoundError or ImportError, do NOT retry the same \
   import -- switch to a different approach immediately (e.g. use pandas.read_html on a URL's raw text \
   instead of BeautifulSoup, or just use built-in string parsing). Only fall back on a well-established fact from memory if every web_fetch attempt has failed (network \
   error, 404, etc.) -- if a fetch SUCCEEDED, you must extract the real number from that fetched content \
   (use python_exec to search/parse it, e.g. regex on digit groups near the relevant label, or \
   pandas.read_html on tables) rather than substituting a number from memory, even one you are confident \
   about. Numbers from memory can be subtly wrong; numbers extracted from a source you just fetched are \
   verified.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Run Python code (pandas, numpy, requests available) and get stdout/stderr back. Use for any computation.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source code to execute"}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL (HTML/CSV/JSON) and return its text content.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]

TOOL_IMPL = {
    "python_exec": lambda args: run_python(args["code"]),
    "web_fetch": lambda args: fetch_url(args["url"]),
}

MAX_TOOL_ROUNDS = 8


def _extract_json(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    candidate = match.group(0) if match else text
    json.loads(candidate)  # raises json.JSONDecodeError if invalid
    return candidate


def _finalize(clean_json: str) -> str:
    """Force the correct log_url into the answer regardless of what the model wrote."""
    obj = json.loads(clean_json)
    if isinstance(obj, dict):
        obj["log_url"] = PUBLIC_LOG_URL
    return json.dumps(obj)


def run_agent(chat_id, user_text, history):
    """
    history: list of {"role": "user"|"assistant", "content": str} from prior
    turns in this chat (system prompt is added here, not stored in history).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": user_text}
    ]
    tool_trace = []
    last_provider = None

    for _ in range(MAX_TOOL_ROUNDS):
        resp, provider = call_llm(messages, TOOLS)
        last_provider = provider
        choice = resp.choices[0]
        msg = choice.message

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                impl = TOOL_IMPL.get(name)
                result = impl(args) if impl else {"error": f"unknown tool {name}"}
                tool_trace.append({"tool": name, "args": args, "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)[:4000],
                })
            continue

        try:
            clean_json = _extract_json(msg.content)
            final_answer = _finalize(clean_json)
        except (json.JSONDecodeError, AttributeError):
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({
                "role": "user",
                "content": "Your last reply was not a single valid JSON object. "
                            "Reply again with ONLY the JSON object, no other text.",
            })
            continue

        write_log_entry(chat_id, user_text, last_provider, tool_trace, final_answer)
        return final_answer

    fallback = json.dumps({"answer": None, "error": "max_tool_rounds_exceeded", "log_url": PUBLIC_LOG_URL})
    write_log_entry(chat_id, user_text, last_provider, tool_trace, fallback)
    return fallback
