import os

import streamlit as st

from src.agent import create_agent_executor, create_memory, load_student_context
from src.database import build_or_load_vectorstore


st.set_page_config(page_title="Vedantu Student Learning Assistant")


WELCOME_MESSAGE = {
    "role": "assistant",
    "content": "Hi Arjun, I can help you plan study time around your weak topics and upcoming tests.",
}


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [WELCOME_MESSAGE]
    if "memory" not in st.session_state:
        st.session_state.memory = create_memory()
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None
    if "indexed_api_key" not in st.session_state:
        st.session_state.indexed_api_key = None
    if "active_model" not in st.session_state:
        st.session_state.active_model = None


def _render_sidebar() -> str:
    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4o", "gpt-3.5-turbo"], index=0)

        context = load_student_context()
        profile = context["profile"]
        performance = context["performance"]["subject_performance"]

        st.divider()
        st.subheader("Logged in as")
        st.write(f"**{profile['name']}** · Grade {profile['grade']} {profile['board']}")
        st.caption(f"Daily study time: {profile['daily_study_time_minutes']} minutes")
        st.write("Weak topics")
        st.write(", ".join(profile["weak_topics"]))
        st.write("Scores")
        for item in performance:
            st.write(f"{item['subject']}: {item['overall_score_percentage']}%")

        if st.button("Reset chat"):
            st.session_state.messages = [WELCOME_MESSAGE]
            st.session_state.memory = create_memory()
            st.session_state.agent_executor = None
            st.rerun()

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        if (
            st.session_state.agent_executor is None
            or st.session_state.indexed_api_key != api_key
            or st.session_state.active_model != model
        ):
            with st.spinner("Preparing study material index..."):
                build_or_load_vectorstore()
            st.session_state.agent_executor = create_agent_executor(
                api_key=api_key,
                memory=st.session_state.memory,
                model=model,
            )
            st.session_state.indexed_api_key = api_key
            st.session_state.active_model = model

    return api_key


def main() -> None:
    _init_session_state()
    api_key = _render_sidebar()

    st.title("Student Learning Assistant")
    st.caption("Personalized RAG assistant for Arjun's Vedantu study plan")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not api_key:
        st.info("Enter your OpenAI API key in the sidebar to start.")
        return

    prompt = st.chat_input("Ask about weak topics, weekly study plans, or test preparation")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print("Invoking agent executor with prompt:", prompt)
            response = st.session_state.agent_executor.invoke({"input": prompt})
            answer = response["output"]
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
