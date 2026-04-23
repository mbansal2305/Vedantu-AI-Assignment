# Vedantu Student Learning Assistant

An AI-powered, context-aware Student Learning Assistant for Vedantu. The app uses Streamlit, LangChain, OpenAI, and a locally persisted ChromaDB vector store to recommend study material and weekly preparation plans for a default logged-in student: Arjun (`student_id=S123`).

## Features

- Personalized answers using Arjun's profile, weak topics, strong topics, and performance scores.
- RAG over `data/study_materials.json` with OpenAI `text-embedding-3-small`.
- Local ChromaDB persistence in `./chroma_db` so materials are embedded once and reused.
- LangChain tools for study-material search and upcoming-test filtering.
- `ConversationBufferWindowMemory(k=5)` for short follow-up context.
- Streamlit chat UI with `st.chat_message`, sidebar API key input, and non-streaming responses.

## Repository Structure

```text
data/
  student_profile.json
  performance_history.json
  study_materials.json
  upcoming_tests.json
src/
  agent.py
  database.py
  tools.py
app.py
requirements.txt
README.md
```

## Setup

Use Python 3.10+. On Windows, Python 3.10 or 3.11 is recommended because older ChromaDB releases may need local C++ build tools on Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Enter your OpenAI API key in the Streamlit sidebar. On the first query, the app creates `./chroma_db`; later runs reuse that persisted index.

## Sample Queries

- `I am weak in Algebra. What should I do next?`
- `What should I study this week?`
- `Which topic should I prioritize first?`
- `I have a Maths test coming up. Help me prepare.`
- `How do I prepare for my upcoming test.`
- `Do I have any upcoming test?`



## Architecture

`src/database.py` loads `study_materials.json`, converts each material into a LangChain `Document`, embeds it with `text-embedding-3-small`, and stores it in ChromaDB. Before indexing, it checks for `./chroma_db/chroma.sqlite3`; if the database exists, it loads the persisted collection instead of calling the embeddings API again. Please note that I have added a little more information in the `study_materials.json` to enhance the retrieval and provide llm reponse with material or video urls.

`src/tools.py` defines two LangChain tools:

- `search_study_materials`: semantic retrieval over Vedantu study materials.
- `get_upcoming_tests`: structured filtering over `upcoming_tests.json`.

`src/agent.py` builds the LangChain OpenAI-tools agent. It injects Arjun's profile and performance context into a `ChatPromptTemplate` system message, then uses tool calling for study material retrieval and test-aware planning.

`app.py` handles the Streamlit UI, OpenAI API key input, Chroma initialization, chat state, and memory.

## Key Decisions

### Why ChromaDB?

ChromaDB provides local persistent storage. The app creates embeddings once and saves them to `./chroma_db`, which reduces repeated OpenAI embedding cost and makes later app startup faster.

### Why System Prompt Injection for Profiles?

The student profile is stable context, not something the model should rediscover through a tool. Injecting Arjun's strong topics, weak topics, daily study time, and scores into the system prompt improves accuracy and latency. It prevents hallucinated profile details and avoids a needless reasoning loop just to fetch known student information.

### Tool-Usage Strategy

The assignment requires tool usage, so the app combines structured context with retrieval:

- The system prompt carries immutable profile and performance context.
- `get_upcoming_tests` handles structured date/topic logic.
- `search_study_materials` handles unstructured semantic retrieval and returns relevant links.

This keeps the assistant personalized while ensuring recommendations are grounded in the available Vedantu materials.

## Verification Notes

To verify reasoning, ask: `I am weak in Algebra. What should I do next?`

The response should mention Algebra as one of Arjun's weak topics, acknowledge the low Mathematics score, and include retrieved Algebra study material links. For `What should I study this week?`, the agent should call the upcoming-test tool, notice the Mathematics test, prioritize Algebra and Quadratic Equations, and retrieve matching materials.

## Limitations and Next Improvements

- The dataset is intentionally small, so retrieval quality is bounded by the fixture content.
- The app supports one default student. A production version should add login and per-student collections or metadata filtering.
- Test-date filtering uses the local runtime date. For production, inject a consistent timezone-aware clock.
- Add automated tests with mocked OpenAI calls for tool selection and prompt-context checks.
