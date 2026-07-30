# Data-Analyst Telegram Bot

An LLM agent, reachable via Telegram, that answers data-analysis questions
(MOSPI and similar public datasets) with a single JSON reply plus a public
run log.

## How it works

1. `bot.py` polls Telegram for incoming text messages (no public HTTPS
   endpoint needed for Telegram itself) and also runs a tiny Flask server
   that serves `logger.LOG_PATH` at `/run.jsonl` -- that's what makes the
   `log_url` in each answer actually `wget`-able.
2. `agent.py` runs a tool-calling loop: the LLM can call `python_exec`
   (pandas/numpy in a subprocess) or `web_fetch` (pull a MOSPI/data.gov.in
   page or CSV) as many times as it needs, then must emit exactly one JSON
   object matching the shape the question asked for.
3. `llm_client.py` tries **Groq first**, retries once on rate limit, then
   **falls back to OpenRouter** (pinned free model, then OpenRouter's
   `openrouter/auto` router in case that free model got delisted -- the
   free catalog rotates).
4. Every completed question appends one line to the JSONL log
   (`logger.py`), and `agent.py` force-overwrites the `log_url` field in
   the final answer with your real deployed URL, so it's always correct
   even if the model forgets it.

## Setup

```bash
git clone <this repo>
cd tds-data-analyst-bot
cp .env.example .env   # fill in tokens/keys
pip install -r requirements.txt
python bot.py
```

Required accounts:
- Telegram bot token from [@BotFather](https://t.me/BotFather) (username must end in `bot`)
- Groq API key: https://console.groq.com
- OpenRouter API key: https://openrouter.ai/keys

## Deploying

Any host that can run a long-lived process works (Telegram uses polling,
not webhooks, so you don't need a domain for Telegram -- but you do need
one public port for the `/run.jsonl` route).

**Render / Railway / Fly.io**: deploy via the included `Dockerfile`, set
the env vars from `.env.example`, expose port `8080`, and set
`PUBLIC_LOG_URL` to whatever public URL that service assigns you (e.g.
`https://your-app.onrender.com/run.jsonl`). Use a service type that keeps
running (worker/web service with `Restart: always`), not something that
sleeps on inactivity.

**Docker anywhere:**
```bash
docker build -t data-analyst-bot .
docker run -d --env-file .env -p 8080:8080 -v $(pwd)/data:/data data-analyst-bot
```

Mount `/data` as a persistent volume so the log survives restarts.

## Testing locally

Clone the official grading pipeline
(`github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot`), point it at your bot's
username, and add your own questions to its `evals/questions.json` to dry-run
before submitting.

## Files

| File | Purpose |
|---|---|
| `bot.py` | Telegram polling + log-serving web route |
| `agent.py` | Tool-calling loop, JSON extraction/repair, log_url injection |
| `llm_client.py` | Groq -> OpenRouter fallback logic |
| `tools/python_exec.py` | Sandboxed pandas/numpy execution |
| `tools/web_fetch.py` | Fetch public dataset URLs |
| `logger.py` | JSONL run log |
