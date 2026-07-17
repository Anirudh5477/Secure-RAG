# app.py
# Enterprise Secure RAG Chatbot — Streamlit Frontend
# Handles: Authentication, Chat UI, Backend Integration, Memory

# ── Windows UTF-8 fix ────────────────────────────────────────────────────────
# The backend files print emoji at import time. Windows defaults to cp1252
# which crashes on those characters. reconfigure() mutates the existing stream
# in place — safer than TextIOWrapper which fails when Streamlit has already
# replaced sys.stdout with its own closed-buffer logger.
import sys
import os
from dotenv import load_dotenv

# Load .env FIRST — LangSmith reads LANGCHAIN_* vars at import time
load_dotenv()

os.environ.setdefault("PYTHONIOENCODING", "utf-8")  # affects any child processes
os.environ.setdefault("PYTHONUTF8", "1")            # Python 3.7+ UTF-8 mode flag

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # Streamlit may have already replaced stdout; safe to ignore
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SecureRAG | Enterprise AI",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CUSTOM CSS  — Premium dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- Root tokens ---------- */
:root {
    --bg-deep:       #0a0d14;
    --bg-card:       rgba(255,255,255,0.04);
    --border:        rgba(255,255,255,0.08);
    --accent-blue:   #4f8ef7;
    --accent-purple: #a78bfa;
    --accent-green:  #34d399;
    --accent-red:    #f87171;
    --text-primary:  #e2e8f0;
    --text-muted:    #64748b;
    --radius:        14px;
}

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
}

/* Remove default Streamlit padding */
.block-container { padding: 1.5rem 2rem 3rem !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ---------- Login card ---------- */
.login-card {
    max-width: 440px;
    margin: 6vh auto;
    padding: 2.5rem 2.5rem 2rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    backdrop-filter: blur(16px);
    box-shadow: 0 8px 40px rgba(79,142,247,0.08);
}
.login-logo {
    font-size: 3rem;
    text-align: center;
    margin-bottom: 0.25rem;
}
.login-title {
    text-align: center;
    font-size: 1.55rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4f8ef7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.login-subtitle {
    text-align: center;
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 2rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ---------- Input fields ---------- */
input[type="text"], input[type="password"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    transition: border-color 0.2s;
}
input[type="text"]:focus, input[type="password"]:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
}

/* ---------- Buttons ---------- */
button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #4f8ef7 0%, #7c5ef7 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.85rem 1.1rem !important;
    margin-bottom: 0.5rem !important;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

/* ---------- Chat input ---------- */
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
}

/* ---------- Role badge ---------- */
.role-badge {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.role-engineering { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid #34d39930; }
.role-hr          { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid #a78bfa30; }
.role-marketing   { background: rgba(251,191,36,0.15);  color: #fbbf24; border: 1px solid #fbbf2430; }
.role-finance     { background: rgba(79,142,247,0.15);  color: #4f8ef7; border: 1px solid #4f8ef730; }
.role-general     { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid #94a3b830; }

/* ---------- Info / error banners ---------- */
.info-banner {
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}
.banner-success { background: rgba(52,211,153,0.1); border: 1px solid #34d39940; color: #34d399; }
.banner-error   { background: rgba(248,113,113,0.1); border: 1px solid #f8717140; color: #f87171; }
.banner-warn    { background: rgba(251,191,36,0.1);  border: 1px solid #fbbf2440; color: #fbbf24; }

/* ---------- Spinner overlay ---------- */
[data-testid="stSpinner"] { color: var(--accent-blue) !important; }

/* ---------- Source citation block ---------- */
.citation-block {
    margin-top: 0.6rem;
    padding: 0.5rem 0.8rem;
    border-left: 3px solid var(--accent-blue);
    background: rgba(79,142,247,0.07);
    border-radius: 0 6px 6px 0;
    font-size: 0.8rem;
    color: var(--text-muted);
}

/* ---------- Divider ---------- */
hr { border-color: var(--border) !important; }

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ---------- Sidebar user card ---------- */
.sidebar-user {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem;
    margin-bottom: 1.2rem;
    text-align: center;
}
.sidebar-avatar {
    width: 56px; height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4f8ef7, #a78bfa);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    margin: 0 auto 0.65rem;
}
.sidebar-name  { font-weight: 600; font-size: 0.95rem; margin-bottom: 0.3rem; }
.sidebar-id    { font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.6rem; }

/* ---------- Stats chips ---------- */
.stat-chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.3rem 0.6rem;
    font-size: 0.78rem;
    color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "employee_data (1).csv")

ROLE_ICON = {
    "engineering": "⚙️",
    "hr":          "👥",
    "marketing":   "📣",
    "finance":     "💰",
    "general":     "🏢",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_employee_db(path: str) -> pd.DataFrame:
    """Load and cache the employee CSV. Normalise column names defensively."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # remove stray whitespace
    return df


def validate_credentials(emp_id: str, password: str, df: pd.DataFrame):
    """
    Return (name, role) if credentials match, else (None, None).
    Comparison is case-sensitive for passwords (security requirement).
    """
    match = df[
        (df["Employer ID"].str.strip() == emp_id.strip()) &
        (df["Password"].str.strip() == password.strip())
    ]
    if not match.empty:
        row = match.iloc[0]
        return str(row["Employer Name"]).strip(), str(row["Role"]).strip().lower()
    return None, None


def role_badge_html(role: str) -> str:
    css_class = f"role-{role}" if role in ROLE_ICON else "role-general"
    icon = ROLE_ICON.get(role, "🏢")
    return f'<span class="role-badge {css_class}">{icon} {role.upper()}</span>'


def split_response(raw: str):
    """
    Separate the LLM answer from the citation line (📁 Sourced from: …).
    Returns (answer_text, citation_text_or_None).
    """
    if "\U0001f4c1 **Sourced from:**" in raw:
        parts = raw.split("\U0001f4c1 **Sourced from:**", 1)
        return parts[0].strip(), parts[1].strip()
    return raw.strip(), None


def render_bot_message(raw_response: str):
    """Render assistant response with a styled citation block below it."""
    answer, citation = split_response(raw_response)
    st.markdown(answer)
    if citation:
        st.markdown(
            f'<div class="citation-block">\U0001f4c1 <strong>Sourced from:</strong> {citation}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
if "logged_in"    not in st.session_state: st.session_state.logged_in    = False
if "user_name"    not in st.session_state: st.session_state.user_name    = ""
if "user_role"    not in st.session_state: st.session_state.user_role    = ""
if "user_id"      not in st.session_state: st.session_state.user_id      = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "msg_count"    not in st.session_state: st.session_state.msg_count    = 0
if "login_error"  not in st.session_state: st.session_state.login_error  = ""


# ─────────────────────────────────────────────────────────────────────────────
#                          LOGIN SCREEN
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:

    # Centre the login card with columns
    _, centre, _ = st.columns([1, 1.2, 1])
    with centre:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        # Branding
        st.markdown('<div class="login-logo">\U0001f510</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-title">SecureRAG Enterprise</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="login-subtitle">Role-Based AI Knowledge Platform</div>',
            unsafe_allow_html=True,
        )

        # Error banner (shown after failed attempt)
        if st.session_state.login_error:
            st.markdown(
                f'<div class="info-banner banner-error">\U0001f6ab {st.session_state.login_error}</div>',
                unsafe_allow_html=True,
            )

        # Login form
        with st.form("login_form", clear_on_submit=False):
            emp_id   = st.text_input("Employee ID", placeholder="e.g. EMP-133065")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted:
            if not emp_id or not password:
                st.session_state.login_error = "Please fill in both fields."
                st.rerun()
            else:
                try:
                    df = load_employee_db(CSV_PATH)
                    name, role = validate_credentials(emp_id, password, df)
                    if name:
                        # Success — persist to session state
                        st.session_state.logged_in    = True
                        st.session_state.user_name    = name
                        st.session_state.user_role    = role
                        st.session_state.user_id      = emp_id.strip()
                        st.session_state.login_error  = ""
                        st.session_state.chat_history = []
                        st.rerun()
                    else:
                        st.session_state.login_error = "Invalid Employee ID or Password."
                        st.rerun()
                except FileNotFoundError:
                    st.session_state.login_error = f"Employee database not found at: {CSV_PATH}"
                    st.rerun()

        # Footer note
        st.markdown(
            '<p style="text-align:center;font-size:0.73rem;color:#475569;margin-top:1.2rem;">'
            '\U0001f6e1\ufe0f All sessions are monitored &bull; PII is auto-redacted &bull; Queries are role-scoped'
            '</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Stop rendering the rest of the app while not authenticated
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#            AUTHENTICATED — Lazy-import backend
# ─────────────────────────────────────────────────────────────────────────────
# Delay import until after auth so the heavy ML models only load once the user
# is verified (avoids paying initialisation cost on the login screen).
try:
    from retrieval import secure_rag_query
    backend_ready = True
    backend_error = ""
except Exception as exc:
    backend_ready = False
    backend_error = str(exc)


# ─────────────────────────────────────────────────────────────────────────────
#                             SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    role  = st.session_state.user_role
    name  = st.session_state.user_name
    uid   = st.session_state.user_id
    icon  = ROLE_ICON.get(role, "\U0001f3e2")

    # User card
    st.markdown(f"""
    <div class="sidebar-user">
        <div class="sidebar-avatar">{icon}</div>
        <div class="sidebar-name">{name}</div>
        <div class="sidebar-id">{uid}</div>
        {role_badge_html(role)}
    </div>
    """, unsafe_allow_html=True)

    # Session stats
    st.markdown("**Session Stats**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stat-chip">\U0001f4ac {st.session_state.msg_count} msgs</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-chip">\U0001f511 {role.upper()}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Security info
    with st.expander("\U0001f6e1\ufe0f Active Security Layers", expanded=False):
        st.markdown("""
- \u2705 **Intent Guardrail** — off-topic queries blocked
- \u2705 **PII Redaction** — personal data scrubbed before search
- \u2705 **RBAC Filter** — results scoped to your department only
- \u2705 **LRU Semantic Cache** — role-isolated response caching
        """)

    st.markdown("---")

    # Clear chat
    if st.button("\U0001f5d1\ufe0f Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.msg_count   = 0
        st.rerun()

    # Logout
    if st.button("\U0001f6aa Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown(
        '<p style="font-size:0.7rem;color:#334155;text-align:center;margin-top:1rem;">'
        'SecureRAG v1.0 &bull; Enterprise Edition</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#                          MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────────────────────
role = st.session_state.user_role
name = st.session_state.user_name

# Page header
st.markdown(f"""
<div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">
    <span style="font-size:2rem;">\U0001f510</span>
    <div>
        <h1 style="margin:0;font-size:1.55rem;font-weight:700;
                   background:linear-gradient(135deg,#4f8ef7,#a78bfa);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            SecureRAG Enterprise
        </h1>
        <p style="margin:0;font-size:0.8rem;color:#64748b;">
            AI knowledge assistant &mdash; {role_badge_html(role)} access level
        </p>
    </div>
</div>
<hr style="margin-top:0.4rem;margin-bottom:1rem;">
""", unsafe_allow_html=True)

# Backend-not-ready warning
if not backend_ready:
    st.markdown(
        f'<div class="info-banner banner-error">'
        f'\u26a0\ufe0f <strong>Backend failed to load:</strong> {backend_error}'
        f'</div>',
        unsafe_allow_html=True,
    )

# Welcome message (injected once as first history item)
if not st.session_state.chat_history:
    welcome = (
        f"👋 Welcome, **{name}**! I'm your secure AI assistant.\n\n"
        f"You're authenticated as a member of the **{role.upper()}** department. "
        f"I can only access and share documents that belong to your department — "
        f"your queries are automatically scoped, PII-redacted, and intent-verified.\n\n"
        f"What would you like to know?"
    )
    st.session_state.chat_history.append({"role": "assistant", "content": welcome})

# Render existing chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_bot_message(msg["content"])
        else:
            st.markdown(msg["content"])

# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT & BACKEND CALL
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    f"Ask a question about {role.upper()} documents...",
    disabled=not backend_ready,
)

if user_input:
    # Guard against blank / whitespace-only inputs
    user_input = user_input.strip()
    if not user_input:
        st.stop()

    # Append & display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.msg_count += 1

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend & display response
        with st.chat_message("assistant"):
            with st.spinner("Searching secure database..."):
                try:
                    response = secure_rag_query(
                        user_query=user_input,
                        user_role=st.session_state.user_role,
                    )
                except Exception as exc:
                    response = (
                        f"\u26a0\ufe0f **System Error:** An unexpected error occurred while "
                        f"processing your request.\n\n`{exc}`\n\n"
                        f"Please try again or contact your system administrator."
                    )

            # Render with citation splitting
            render_bot_message(response)

    # Persist assistant response to history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.msg_count += 1

