import asyncio
import os
import threading

from flask import Flask, send_file
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

from agent import run_agent
from logger import LOG_PATH

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
PORT = int(os.environ.get("PORT", 8080))
MAX_HISTORY_MESSAGES = 10  # keep last few turns per chat, for multi-turn questions

histories = {}  # chat_id -> list[{"role": "user"|"assistant", "content": str}]

# --- tiny public server just to expose the log file at a stable URL ---
web_app = Flask(__name__)


@web_app.route("/run.jsonl")
def serve_log():
    if not os.path.exists(LOG_PATH):
        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
        open(LOG_PATH, "a").close()
    return send_file(LOG_PATH, mimetype="application/x-ndjson")


@web_app.route("/")
def health():
    return "ok"


def run_web_server():
    web_app.run(host="0.0.0.0", port=PORT)


# --- telegram side ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text or ""

    history = histories.setdefault(chat_id, [])
    # run_agent is blocking (subprocess + HTTP calls) -- push it off the event loop
    reply = await asyncio.to_thread(run_agent, chat_id, text, history)

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": reply})
    histories[chat_id] = history[-MAX_HISTORY_MESSAGES:]

    await update.message.reply_text(reply)


def main():
    threading.Thread(target=run_web_server, daemon=True).start()
    telegram_app = ApplicationBuilder().token(BOT_TOKEN).build()
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    telegram_app.run_polling()


if __name__ == "__main__":
    main()
