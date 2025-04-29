# 🧠 SQL Agent Small Demo with LangChain & Gemini 2.0

This notebook demonstrates how to build a simple yet powerful **SQL agent** using the LangChain framework with **Google's Gemini 2.0** model on **Vertex AI**, capable of answering natural language questions using a SQLite database.

> 🔗 [View in Google Colab](https://colab.research.google.com/drive/1SEhD9F_J_02oQZOC31fIuEn3oX6Rbmpw)

---

## 📦 Setup

Install the required packages:

```bash
pip install pydantic==2.9.2
pip install langchain-community
pip install -U langchain-google-genai
pip install langchain_google_vertexai
pip install psycopg2
pip install --upgrade google-genai
pip install --upgrade --quiet google-cloud-aiplatform
```

---

## ☁️ Vertex AI Authentication

Authenticate with your Google Cloud account:

```python
from google.colab import auth
auth.authenticate_user()
```

Set your project and region:

```python
import vertexai
vertexai.init(project="your-project-id", location="europe-west1")
```

---

## 🤖 Load the Gemini 2.0 Model

This demo uses the lightweight but capable `gemini-2.0-flash-001` model:

```python
from langchain_google_vertexai import ChatVertexAI

vertex_llm = ChatVertexAI(
    model="gemini-2.0-flash-001",
    temperature=0,
    max_tokens=2048,
    max_retries=16
)
```

---

## 🛠️ Database Setup

The SQLite database used in this example is `bicycles.db`. You can load it like this:

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///bicycles.db")
db.run("SELECT * FROM categories LIMIT 3;")
```

---

## 🧠 Configure the SQL Agent

This function sets up the agent with:

- Default SQL tools (`query_checker`, `query_sql_db`, etc.)
- A clean system prompt
- A tool-calling multi-action agent
- A verbose LangChain agent executor

```python
from langchain.agents import create_tool_calling_agent
from langchain.agents.agent import AgentExecutor, RunnableMultiActionAgent
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

def configure_database_agent(db, llm):
    database_tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

    prompt = (
        "You are an AI SQL agent specializing in generating and executing "
        "**syntactically correct SQL queries** in SQLite language.\n"
        "Return a final natural language sentence answer given the executed answer of the SQL query.\n"
        "### Response Format\n"
        "Your response must only contain the final natural language sentence, without any explanation or reasoning."
    )

    database_prompt = ChatPromptTemplate.from_messages([
        prompt,
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    runnable = create_tool_calling_agent(llm, database_tools, database_prompt)
    agent = RunnableMultiActionAgent(runnable=runnable, input_keys_arg=["input"], return_keys_arg=["output"])

    database_agent = AgentExecutor(
        agent=agent,
        tools=database_tools,
        verbose=True,
        max_iterations=40,
        max_execution_time=40
    )
    print("Database Agent created!")
    return database_agent
```

Then instantiate it:

```python
database_agent = configure_database_agent(db, vertex_llm)
```

---

## 💬 Example Usage

Ask a natural language question and get an answer powered by SQL and Gemini:

```python
question = "How many bicycles are sold?"
response = database_agent.invoke(question)
print(response['output'])
```

---

## 🧪 Features

- ✅ Uses **LangChain's tool calling agent**
- ✅ Connects to a **local SQLite database**
- ✅ Uses **Vertex AI's Gemini 2.0 Flash** for low latency
- ✅ Translates **natural language to SQL queries**
- ✅ Responds with **natural language only** (not raw SQL)
- ✅ Fully extendable with custom tools and prompts

---

## 🔧 Potential Extensions

You can expand this notebook further by:

- Adding **few-shot examples** in the prompt
- Swapping in **PostgreSQL / MySQL** using SQLAlchemy
- Building **multi-agent systems** using LangChain’s router agents
- Implementing custom tools for **table explanations**, **column descriptions**, or **data summaries**
- Integrating this with a **Streamlit / Gradio UI**

---

## ⚠️ Notes

- This demo uses all **default SQL tools** from LangChain.
- **No prompt tuning** or fine-tuning is applied.
- Make sure your database structure is understandable (clear table/column names) to help the agent succeed.

---

## 🧾 License

This project is for **educational and prototype** use. Please validate all generated SQL in production systems.
