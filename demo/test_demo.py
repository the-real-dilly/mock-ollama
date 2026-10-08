import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import mock_ollama  # noqa: E402


@pytest.fixture(scope="module")
def mock_url():
    server = mock_ollama.make_server(port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_agent_runs_against_mock(mock_url, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", mock_url)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    import sql_agent

    messages = sql_agent.ask(sql_agent.build_agent())
    tools = [m.name for m in messages if m.type == "tool"]
    assert tools == ["sql_db_list_tables", "sql_db_schema", "sql_db_query"]
    assert "347" in next(m.content for m in messages if m.name == "sql_db_query")
    assert "347" in messages[-1].content
