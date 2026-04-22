import json
from pathlib import Path

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.database import DATA_DIR
from src.tools import TOOLS


DEFAULT_STUDENT_ID = "S123"


def _load_json(filename: str) -> dict:
    with (Path(DATA_DIR) / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_student_context(student_id: str = DEFAULT_STUDENT_ID) -> dict:
    print(f"Loading context for student ID: {student_id}")
    profile = _load_json("student_profile.json")
    performance = _load_json("performance_history.json")

    if profile.get("student_id") != student_id:
        raise ValueError(f"Student {student_id} was not found in student_profile.json.")
    if performance.get("student_id") != student_id:
        raise ValueError(f"Student {student_id} was not found in performance_history.json.")

    return {"profile": profile, "performance": performance}


def format_student_context(context: dict) -> str:
    profile = context["profile"]
    performance = context["performance"]
    scores = performance.get("subject_performance", [])
    score_lines = [
        f"- {item['subject']}: {item['overall_score_percentage']}%"
        for item in scores
    ]

    return "\n".join(
        [
            f"Student ID: {profile['student_id']}",
            f"Name: {profile['name']}",
            f"Grade/Board: Grade {profile['grade']} {profile['board']}",
            f"Target exam: {profile['target_exam']}",
            f"Daily study time: {profile['daily_study_time_minutes']} minutes",
            f"Strong topics: {', '.join(profile.get('strong_topics', []))}",
            f"Weak topics: {', '.join(profile.get('weak_topics', []))}",
            "Scores:",
            *score_lines,
        ]
    )


def create_memory() -> ConversationBufferWindowMemory:
    return ConversationBufferWindowMemory(
        k=5,
        return_messages=True,
        memory_key="chat_history",
        input_key="input",
        output_key="output",
    )


def create_agent_executor(
    api_key: str,
    memory: ConversationBufferWindowMemory,
    model: str = "gpt-4o",
) -> AgentExecutor:
    print("Creating agent executor with model:", model)
    student_context = format_student_context(load_student_context())
    llm = ChatOpenAI(model=model, temperature=0.2, api_key=api_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "\n".join(
                    [
                        "You are Vedantu's Student Learning Assistant for Arjun.",
                        "Use the injected student context as trusted profile and performance truth.",
                        "Personalize every answer around weak topics, strong topics, scores, upcoming tests, and daily study time.",
                        "When the student asks for help, recommendations, a study plan, or test preparation, use search_study_materials to retrieve relevant Vedantu materials and include useful links.",
                        "When the student asks what to study this week or mentions an upcoming test, use get_upcoming_tests, then combine the result with weak topics and retrieved study materials.",
                        "Do not invent student details, test dates, scores, or links. If a tool has no result, say so and still give a practical next step.",
                        "Keep responses concise, supportive, and specific.",
                        "",
                        "Student context:",
                        "{student_context}",


                        "If you do not find relevant study materials or upcoming tests, inform the student that you couldn't find any matches. Do not provide generic advice like 'review weak topics' without specific materials or test information to back it up.",
                    ]
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    ).partial(student_context=student_context)

    agent = create_openai_tools_agent(llm=llm, tools=TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )
