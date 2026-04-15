import html
import os
from pathlib import Path
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:3001").rstrip("/")


def _html_body(text: str) -> str:
    return html.escape(text or "", quote=False).replace("\n", "<br/>")


def render_message_sources(citations: list, confidence: float) -> None:
    if not citations:
        return
    with st.expander(
        f"Sources ({len(citations)}) · confidence {float(confidence):.3f}",
        expanded=False,
    ):
        for i, chunk in enumerate(citations, start=1):
            col_a, col_b = st.columns([1, 8])
            with col_a:
                st.markdown(
                    f'<span class="cite-tag">[{i}]</span>',
                    unsafe_allow_html=True,
                )
            with col_b:
                st.markdown(
                    f"**{chunk.get('title', 'Unknown')}** · "
                    f"*{chunk.get('section', '')}*"
                )
                st.markdown(
                    f'<span class="score-badge">'
                    f"score {round(chunk.get('score', 0), 3)}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(chunk.get("text", ""))
            if i < len(citations):
                st.divider()


def render_chat_message(message: dict) -> None:
    if message["role"] == "user":
        body = _html_body(message.get("content", ""))
        st.markdown(
            f'''<div style="display:flex;justify-content:flex-end;margin:0.4rem 0;">
                <div style="background:#E8E8E4;border-radius:18px;padding:0.55rem 1rem;
                            max-width:72%;font-size:0.9rem;line-height:1.7;color:#1A1A1A;
                            font-family:Inter,sans-serif;">
                    {body}
                </div>
            </div>''',
            unsafe_allow_html=True,
        )
        return

    body = _html_body(message.get("content", ""))
    st.markdown(
        f'''<div style="display:flex;justify-content:flex-start;margin:0.4rem 0;">
            <div style="background:transparent;padding:0.4rem 0;max-width:100%;
                        font-size:0.9rem;line-height:1.7;color:#1A1A1A;
                        font-family:Inter,sans-serif;">
                {body}
            </div>
        </div>''',
        unsafe_allow_html=True,
    )
    cites = message.get("citations") or []
    if cites:
        render_message_sources(cites, float(message.get("confidence") or 0.0))


def _inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {
            --bg:           #FAF9F7;
            --surface:      #FFFFFF;
            --sidebar-bg:   #FFFFFF;
            --sidebar-border:#EBEBEB;
            --border:       #E5E5E2;
            --border-2:     #D4D4D0;
            --text:         #1A1A1A;
            --text-2:       #3D3D3A;
            --muted:        #8C8C89;
            --accent:       #D97757;
            --accent-soft:  #FDF0EB;
            --accent-border:#F5D1C3;
            --radius-sm:    12px;
            --sidebar-width: 240px;
            --gutter:        2rem;
            --chat-column-outer: calc(800px + 2 * var(--gutter));
            --surface-hover: #F0F0EE;
            --surface-subtle: #F5F5F3;
            --shadow-bottom-float:
                0 0.25rem 1.25rem hsl(0 0% 0% / 3.5%),
                0 0 0 0.5px color-mix(in srgb, var(--border) 55%, transparent);
        }

        html, body, .stApp {
            background: var(--bg) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif !important;
            color: var(--text) !important;
        }

        [data-testid="stAppViewContainer"] > .main {
            background: var(--bg) !important;
        }

        .block-container,
        .stMainBlockContainer {
            max-width: var(--chat-column-outer) !important;
            width: 100% !important;
            padding: 1.5rem var(--gutter) 6rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        [data-testid="stSidebar"] .block-container {
            max-width: 100% !important;
            padding: 0.75rem 0.6rem !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar-bg) !important;
            border-right: 1px solid var(--sidebar-border) !important;
        }
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: var(--sidebar-width) !important;
            max-width: var(--sidebar-width) !important;
            width: var(--sidebar-width) !important;
        }
        [data-testid="stSidebarContent"] {
            padding: 0 !important;
        }
        [data-testid="stSidebarHeader"] {
            margin-bottom: 0 !important;
            position: absolute !important;
            z-index: 10 !important;
            right: 0 !important;
            height: 50px !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.35rem !important;
        }

        h1, h2, h3 {
            font-family: 'Inter', sans-serif !important;
            color: var(--text) !important;
        }

        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 18vh 1rem 2rem;
            text-align: center;
        }
        .welcome-title {
            font-family: 'Inter', sans-serif;
            font-size: 1.85rem;
            font-weight: 500;
            color: var(--text);
            letter-spacing: -0.02em;
        }
        .welcome-icon {
            color: var(--accent);
            margin-right: 6px;
        }
        .welcome-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: center;
            margin-top: 1.25rem;
            max-width: 540px;
        }
        .welcome-chip {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 0.4rem 0.9rem;
            font-size: 0.8rem;
            color: var(--text-2);
            cursor: default;
            transition: border-color 0.15s, background 0.15s;
        }
        .welcome-chip:hover {
            border-color: var(--border-2);
            background: var(--surface-subtle);
        }

        [data-testid="stChatInput"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 20px !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: var(--border-2) !important;
        }
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] > div > div,
        [data-testid="stChatInput"] form,
        [data-testid="stChatInput"] [data-baseweb] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }
        [data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: var(--text) !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.85rem !important;
            line-height: 1.5 !important;
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--muted) !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stChatInput"] button,
        [data-testid="stChatInputSubmitButton"] button {
            border-radius: 50% !important;
            border: none !important;
            background: var(--text) !important;
            color: var(--surface) !important;
            width: 28px !important;
            height: 28px !important;
            min-width: 28px !important;
            min-height: 28px !important;
            padding: 0 !important;
        }
        [data-testid="stChatInput"] button:hover,
        [data-testid="stChatInputSubmitButton"] button:hover {
            background: var(--text-2) !important;
        }
        [data-testid="stChatInput"] button svg {
            width: 14px !important;
            height: 14px !important;
        }

        [data-testid="stBottom"] {
            background: var(--bg) !important;
            border-top: none !important;
            box-shadow: none !important;
        }
        [data-testid="stBottom"] > div {
            background: transparent !important;
        }

        [data-testid="stBottomBlockContainer"] {
            max-width: 864px !important;
            padding-inline: 2rem !important;
            margin-inline: auto !important;
        }
        [data-testid="stChatInput"] {
            box-shadow: var(--shadow-bottom-float) !important;
        }

        [data-testid="stSpinner"] {
            padding: 0.5rem 0 !important;
        }
        [data-testid="stSpinner"] > div {
            display: flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
        }
        [data-testid="stSpinner"] p {
            color: var(--muted) !important;
            font-size: 0.85rem !important;
            font-family: 'Inter', sans-serif !important;
        }
        [data-testid="stStatusWidget"],
        .stAlert {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            font-family: 'Inter', sans-serif !important;
        }

        [data-testid="stSidebar"] .stButton > button {
            background: transparent !important;
            color: var(--text-2) !important;
            border: none !important;
            border-radius: 6px !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.85rem !important;
            padding: 0.4rem 0.6rem !important;
            text-align: left !important;
            justify-content: flex-start !important;
            transition: background 0.12s ease !important;
            font-weight: 400 !important;
            min-height: 0 !important;
            line-height: 1.4 !important;
            letter-spacing: 0 !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: var(--surface-hover) !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] .stButton > button > div {
            display: flex !important;
            justify-content: flex-start !important;
            align-items: center !important;
            width: 100% !important;
        }
        [data-testid="stSidebar"] .stButton > button p {
            text-align: left !important;
            width: 100% !important;
        }
        p.sidebar-recents-heading {
            font-size: 0.875rem !important;
            font-weight: 400 !important;
            color: #6b6b66 !important;
            font-family: Inter, sans-serif !important;
            text-align: left !important;
            margin: 2rem 0 0.15rem 0.15rem !important;
            padding: 0.2rem 0.25rem 0.5rem 0 !important;
            line-height: 1.35 !important;
            position: relative !important;
            z-index: 4 !important;
        }
        .sidebar-recents-spacer {
            display: block;
            height: 10px;
            margin: 0;
            padding: 0;
        }
        [data-testid="stSidebar"] hr {
            margin: 0.4rem 0 !important;
        }

        [data-testid="stExpander"] {
            background: var(--bg) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            box-shadow: none !important;
            outline: none !important;
            overflow: hidden !important;
        }
        [data-testid="stExpander"] details {
            background: transparent !important;
            border: 0 !important;
        }
        [data-testid="stExpander"] details > summary {
            background: var(--surface) !important;
            border-radius: 0 !important;
            color: var(--muted) !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stExpander"] details[open] > summary {
            background: var(--surface-hover) !important;
            border-radius: 0 !important;
        }
        [data-testid="stExpander"] details > div {
            background: var(--surface) !important;
            padding: 1rem !important;
        }
        [data-testid="stExpander"] details:not([open]) > summary:hover {
            background: var(--surface-hover) !important;
            border-radius: 0 !important;
        }
        [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
            gap: 0.25rem !important;
        }
        [data-testid="stExpander"] [data-testid="stHorizontalBlock"] {
            padding-top: 0.75rem !important;
            gap: 0.5rem !important;
            align-items: flex-start !important;
        }
        [data-testid="stExpander"] hr {
            margin: 0.5rem 0 !important;
        }
        [data-testid="stExpander"] details:focus-within,
        [data-testid="stExpander"]:focus,
        [data-testid="stExpander"]:focus-within {
            border-color: var(--border) !important;
            outline: none !important;
            box-shadow: none !important;
        }

        .cite-tag {
            display: inline-block;
            background: var(--accent-soft);
            color: var(--accent);
            border: 1px solid var(--accent-border);
            border-radius: 6px;
            padding: 1px 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            font-weight: 500;
            margin-right: 4px;
        }
        .score-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--muted);
        }

        [data-testid="stCaptionContainer"] p,
        .stCaption {
            color: var(--muted) !important;
            font-size: 0.8rem !important;
        }

        hr { border-color: var(--border) !important; }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] { background: transparent !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=30)
def fetch_history():
    try:
        res = requests.get(f"{BACKEND_URL}/history", timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception:
        return []


st.set_page_config(
    page_title="Climate Research Assistant",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)
_inject_styles()


with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:flex-start;
                    padding:0.8rem 0.4rem 0.4rem;margin-bottom:2rem;">
            <span style="font-size:1.15rem;font-weight:600;color:#1A1A1A;
                         font-family:Inter,sans-serif;letter-spacing:-0.01em;">
                Climate RAG
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("+ New chat", key="nav_new_chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.chat_id = None
        st.rerun()

    st.markdown(
        '<p class="sidebar-recents-heading">Recents</p>'
        '<div class="sidebar-recents-spacer" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )

    hist_data = fetch_history() or []
    if hist_data:
        for entry in reversed(hist_data[-15:]):
            title = entry.get("title", "Untitled")[:38]
            if st.button(
                title,
                key=f"hist_{entry.get('chat_id', '')}",
                use_container_width=True,
            ):
                st.session_state.chat_id = entry.get("chat_id")
                st.session_state.messages = []
                for msg in entry.get("messages", []):
                    st.session_state.messages.extend(
                        [
                            {"role": "user", "content": msg.get("query", "")},
                            {
                                "role": "assistant",
                                "content": msg.get("answer", ""),
                                "meta": {
                                    "tool_calls": msg.get("tool_calls", []),
                                    "num_iterations": msg.get(
                                        "num_iterations", 0
                                    ),
                                },
                                "citations": msg.get("chunks") or [],
                                "confidence": float(msg.get("confidence") or 0.0),
                            },
                        ]
                    )
                st.session_state.pending_question = None
                st.rerun()
    else:
        st.caption("No conversations yet.")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-title">
                <span class="welcome-icon">&#10044;</span>Climate Research Assistant
            </div>
            <div class="welcome-chips">
                <span class="welcome-chip">How does the Gulf Stream affect European climate?</span>
                <span class="welcome-chip">What does the Last Interglacial tell us about sea level rise?</span>
                <span class="welcome-chip">How does solar variability influence global temperature?</span>
                <span class="welcome-chip">What is the relationship between CO2 and global warming?</span>
                <span class="welcome-chip">How do tropical cyclones respond to climate change?</span>
                <span class="welcome-chip">What causes thermohaline circulation to weaken?</span>
                <span class="welcome-chip">How do aerosols affect Earth's radiation budget?</span>
                <span class="welcome-chip">What are Dansgaard-Oeschger climate events?</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    render_chat_message(message)

pending = st.session_state.pending_question
prompt = st.chat_input(
    "Ask a question about climate science...",
    disabled=bool(pending),
)

if pending:
    with st.spinner("Searching papers and drafting your answer ..."):
        try:
            payload = {
                "question": pending,
                "top_k": 10,
                "chat_id": st.session_state.get("chat_id"),
                "chat_history": st.session_state.messages[:-1],
            }
            res = requests.post(
                f"{BACKEND_URL}/query", json=payload, timeout=60
            )
            res.raise_for_status()
            data = res.json()

            if "chat_id" in data:
                st.session_state.chat_id = data["chat_id"]

            answer = data.get("answer", "No answer provided.")
            citations = data.get("citations", [])
            confidence = data.get("confidence", 0.0)
            tool_calls = data.get("tool_calls", [])
            num_iterations = data.get("num_iterations", 0)

            assistant_msg = {
                "role": "assistant",
                "content": answer,
                "meta": {
                    "tool_calls": tool_calls,
                    "num_iterations": num_iterations,
                },
                "citations": citations,
                "confidence": confidence,
            }
            st.session_state.messages.append(assistant_msg)
            fetch_history.clear()
        except Exception as e:
            err_msg = f"Backend error: {e}"
            st.error(err_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": err_msg}
            )
    st.session_state.pending_question = None
    st.rerun()

elif prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_question = prompt
    st.rerun()
