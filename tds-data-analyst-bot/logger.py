import json
import os
from datetime import datetime, timezone

LOG_PATH = os.environ.get("LOG_PATH", "/data/run.jsonl")


def write_log_entry(chat_id, user_message, provider, tool_trace, final_answer_json):
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "chat_id": chat_id,
        "user_message": user_message,
        "provider": provider,
        "tool_trace": tool_trace,
        "final_answer": final_answer_json,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
