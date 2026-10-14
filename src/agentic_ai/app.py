import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

DEFAULT_MODEL = "qwen3.5:2b"


def build_agent(
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    max_results: int,
    tavily_api_key: str,
) -> Any:
    search_tool = TavilySearchResults(
        max_results=max_results,
        api_wrapper=TavilySearchAPIWrapper(tavily_api_key=tavily_api_key),
    )
    llm = ChatOllama(
        model=model,
        temperature=temperature,
        reasoning=False,
        num_predict=max_tokens,
    )
    return create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=(
            "You are a helpful research assistant. Use web search when current "
            "or factual information would improve your answer. Summarize the "
            "evidence clearly and say when information is uncertain."
        ),
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return str(content)


def main() -> None:
    load_dotenv()
    st.set_page_config(
        page_title="Research Assistant",
        page_icon="✳",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --app-bg: #0a0f1a;
            --panel-bg: #111a29;
            --panel-border: #263449;
            --text-primary: #f1f5f9;
            --text-secondary: #a6b3c5;
            --accent: #a5b4fc;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background: var(--app-bg);
            color: var(--text-primary);
        }
        [data-testid="stHeader"] { background: var(--app-bg); }
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] { display: none; }
        .block-container {
            max-width: 780px;
            padding-top: 4rem;
            padding-bottom: 4rem;
        }
        .stMarkdown, .stMarkdown p, .stMarkdown li, label,
        [data-testid="stAlert"] {
            color: var(--text-primary);
        }
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {
            background: #0e1726;
            color: var(--text-primary);
            border-color: var(--panel-border);
            border-radius: 12px;
            font-size: 1rem;
        }
        [data-testid="stTextArea"] textarea::placeholder {
            color: #8795aa;
            opacity: 1;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel-bg);
            border-color: var(--panel-border);
            border-radius: 16px;
        }
        [data-testid="stAlert"] {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
        }
        .stFormSubmitButton button[kind="primary"] {
            min-height: 3rem;
            background: #a5b4fc;
            border-color: #a5b4fc;
            border-radius: 12px;
            color: #101827;
            font-weight: 700;
        }
        .stFormSubmitButton button[kind="primary"]:hover {
            background: #c7d2fe;
            border-color: #c7d2fe;
        }
        .eyebrow {
            color: var(--accent); font-size: .76rem; font-weight: 700;
            letter-spacing: .12em; text-transform: uppercase;
        }
        .subheading { color: var(--text-secondary) !important; }
        .answer-label {
            color: var(--accent);
            font-size: .75rem;
            font-weight: 700;
            letter-spacing: .1em;
            text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("runs", [])

    st.markdown(
        '<p class="eyebrow">A little curiosity goes a long way</p>',
        unsafe_allow_html=True,
    )
    st.title("What would you like to know?")
    st.markdown(
        '<p class="subheading">Ask a question and I’ll help you find a clear answer.</p>',
        unsafe_allow_html=True,
    )

    with st.form("agent_run_form"):
        query = st.text_area(
            "Your question",
            placeholder="Ask anything...",
            height=110,
        )
        submitted = st.form_submit_button(
            "Find an answer",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not query.strip():
            st.warning("Type a question above to get started.")
        elif not os.getenv("TAVILY_API_KEY"):
            st.error(
                "Web search isn’t set up yet. Please ask the app owner to configure it."
            )
        else:
            with st.spinner("Finding an answer..."):
                agent = build_agent(
                    model=DEFAULT_MODEL,
                    temperature=0,
                    max_tokens=300,
                    max_results=2,
                    tavily_api_key=os.environ["TAVILY_API_KEY"],
                )
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": query.strip()}]},
                    config={"recursion_limit": 25},
                )

            messages = result["messages"]
            final_answers = [
                message
                for message in messages
                if isinstance(message, AIMessage) and not message.tool_calls
            ]
            answer = _content_to_text(final_answers[-1].content)
            st.session_state["runs"].insert(
                0,
                {
                    "query": query.strip(),
                    "answer": answer,
                },
            )

    runs = st.session_state["runs"]
    if runs:
        st.markdown("### Recent runs")
        for run in runs:
            with st.container(border=True):
                st.markdown(f"**You asked**  \n{run['query']}")
                st.markdown(
                    '<p class="answer-label">Answer</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(run["answer"])


if __name__ == "__main__":
    main()
