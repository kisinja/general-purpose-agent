# Agentic AI

A Streamlit workspace for running the web-search research agent with a local
Ollama model.

## Run locally

1. Install the project dependencies:

   ```powershell
   uv sync
   ```

2. Make sure Ollama is running and the configured model is available. The
   default model is `qwen3.5:2b`; pull it with `ollama pull qwen3.5:2b` if
   needed.

3. Start the app:

   ```powershell
   uv run streamlit run src/agentic_ai/app.py
   ```

4. Set `TAVILY_API_KEY=...` in a `.env` file in the project root.

The app uses the `qwen3.5:2b` Ollama model and keeps your recent questions and
answers visible during the current session. Configuration controls and raw
search details are intentionally kept out of the user interface.