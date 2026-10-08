"""LangChain SQL agent (from the LangChain SQL tutorial) pointed at the mock server.

Start the mock first (`python mock_ollama.py`), then `python demo/sql_agent.py`.
With no OPENAI_API_KEY / OPENAI_BASE_URL set, it targets the mock. Set a real key
(and leave OPENAI_BASE_URL unset) to run the same agent against OpenAI.
"""
import os
import sys
from pathlib import Path

if "OPENAI_API_KEY" not in os.environ and "OPENAI_BASE_URL" not in os.environ:
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "test"

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

DB_PATH = Path(__file__).with_name("Chinook.db")
QUESTION = "How many albums are there?"


def build_agent(model_name="gpt-5.5"):
    model = init_chat_model(model_name)
    db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    tools = SQLDatabaseToolkit(db=db, llm=model).get_tools()
    system_prompt = (
        f"You are an agent for interacting with a SQL database ({db.dialect}). "
        "List the tables, check the relevant schema, then run a query to answer. "
        "Never run DML statements (INSERT, UPDATE, DELETE, DROP)."
    )
    return create_agent(model, tools, system_prompt=system_prompt)


def ask(agent, question=QUESTION):
    return agent.invoke({"messages": [{"role": "user", "content": question}]})["messages"]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or QUESTION
    for m in ask(build_agent(), question):
        m.pretty_print()
