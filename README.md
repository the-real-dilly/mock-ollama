# Mock Ollama (OpenAI-compatible)

Fake `/v1/chat/completions` server for testing a LangChain `ChatOpenAI` agent offline. Standard library only.

## Run

    python mock_ollama.py                # http://127.0.0.1:11434/v1
    python mock_ollama.py --port 8000 --script my_script.json

## Point your agent at it

Your code stays the same (`init_chat_model("gpt-5.5")`); set the environment first:

    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "test"    # ignored by the mock

Or: `init_chat_model("gpt-5.5", base_url="http://localhost:11434/v1", api_key="test")`.

## Script

`script.json` is a list of steps. The step played is the number of `tool` messages already in the request, so runs are deterministic. Past the end, the last step repeats (end with a `content` step).

    {"tool_calls": [{"name": "sql_db_query", "arguments": {"query": "SELECT 1;"}}]}
    {"content": "Final answer text."}

The default script follows the LangChain SQL agent tutorial (list tables, schema, query, answer). Edit the schema/query steps to match your question.

## Not supported

`/v1/responses` (used if `use_responses_api=True`), Ollama-native `/api/*` routes, embeddings.

## Tests

    pip install pytest openai
    python -m pytest
