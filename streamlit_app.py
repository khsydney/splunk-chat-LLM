# streamlit_app.py — Chat UI for the RAG backend
import os, re, sys, time, html, uuid, pathlib, subprocess
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import httpx
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
HTTPXClientInstrumentor().instrument()

APP_NAME   = "Splunk RAG"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
USER_NAME  = os.getenv("UI_USER_NAME", "nick")

st.set_page_config(page_title=APP_NAME, page_icon=":material/forum:", layout="wide",
                   initial_sidebar_state="expanded")

# ──────────────────────────────────────────────────────────────────────────────
# Styling (warm paper, serif answers, user bubbles, pill composer)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&display=swap');

:root{
  --paper:#FAF9F5; --side:#F5F4EE; --ink:#1F1E1D; --muted:#73726C; --faint:#A3A29B;
  --line:#E3E1D8; --bubble:#F0EEE6; --clay:#C96442; --card:#FFFFFF;
}
html, body, .stApp, [data-testid="stAppViewContainer"]{ background:var(--paper); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button{
  font-family:'Inter', system-ui, -apple-system, sans-serif;
}

/* Header bar: transparent, no deploy button */
[data-testid="stHeader"]{ background:var(--paper); }
[data-testid="stAppDeployButton"], [data-testid="stDecoration"]{ display:none; }

/* Centered reading column */
[data-testid="stMainBlockContainer"]{ max-width:800px; padding-top:3.2rem; padding-bottom:9rem; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"]{ background:var(--side); border-right:1px solid var(--line); }
[data-testid="stSidebarUserContent"]{ padding-top:.5rem; }
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]{
  width:100%; justify-content:flex-start; text-align:left; padding:.42rem .65rem; border-radius:.55rem;
  color:var(--ink); font-weight:400; font-size:.92rem; min-height:0;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:hover{ background:#EAE8DF; color:var(--ink); }
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] p{
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:100%;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{
  width:100%; justify-content:flex-start; background:#E8E6DC; border:none; color:var(--ink);
  font-weight:500; padding:.42rem .65rem; border-radius:.55rem; font-size:.92rem; min-height:0;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p{
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
[data-testid="stSidebar"] [data-testid="stExpander"] details{ border:none; background:transparent; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary{ padding:.35rem .65rem; font-size:.92rem; }
.side-brand{ font-family:'Source Serif 4', Georgia, serif; font-size:1.35rem; font-weight:500;
  color:var(--ink); padding:.1rem .6rem .7rem; display:flex; align-items:center; gap:.45rem; }
.side-label{ color:var(--muted); font-size:.8rem; padding:.9rem .65rem .25rem; }
.side-user{ display:flex; align-items:center; gap:.55rem; padding:.6rem .65rem; margin-top:1.2rem;
  border-top:1px solid var(--line); color:var(--ink); font-size:.92rem; }
.side-user .avatar{ width:28px; height:28px; border-radius:50%; background:#3D3D3A; color:#fff;
  display:flex; align-items:center; justify-content:center; font-size:.8rem; font-weight:600; }
.side-user .plan{ color:var(--muted); }

/* ── Top breadcrumb ─────────────────────────────────────── */
.topbar{ position:fixed; top:0; left:0; right:0; height:3rem; display:flex; align-items:center;
  justify-content:center; pointer-events:none; z-index:999990; }
.topbar .crumb{ font-size:.95rem; color:var(--ink); padding:.25rem .6rem;
  border-radius:.5rem; max-width:55vw; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.topbar .crumb .proj{ color:var(--muted); }

/* ── Messages ───────────────────────────────────────────── */
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"]{ display:none; }
[data-testid="stChatMessage"]{ background:transparent; padding:.25rem 0; gap:0; }
/* user: soft bubble on the right */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  flex-direction:row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
  background:var(--bubble); border-radius:1.1rem; padding:.65rem 1rem; max-width:80%;
  margin-left:auto !important; margin-right:0 !important; flex:0 1 auto; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p{ font-size:1rem; }
/* assistant: plain serif text, no bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] :is(p,li){
  font-family:'Source Serif 4', Georgia, serif; font-size:1.08rem; line-height:1.65; color:var(--ink); }
[data-testid="stChatMessage"] code{ background:var(--bubble); color:#B4532F; border-radius:.35rem; padding:.05rem .3rem; }
[data-testid="stChatMessage"] blockquote{ border-left:3px solid var(--line); color:var(--muted); margin:.2rem 0 .6rem; }
[data-testid="stChatMessage"] [data-testid="stExpander"] details{ border:1px solid var(--line); border-radius:.75rem; background:var(--card); }
.meta{ color:var(--faint); font-size:.8rem; margin-top:.15rem; display:flex; gap:.9rem; align-items:center; }

/* ── Welcome screen ─────────────────────────────────────── */
.welcome{ text-align:center; margin:16vh 0 1.6rem; }
.welcome h1{ font-family:'Source Serif 4', Georgia, serif; font-weight:400; font-size:2.3rem;
  color:var(--ink); margin:.2rem 0 0; padding:0; }
.welcome p{ color:var(--muted); margin-top:.5rem; }
[data-testid="stMain"] [data-testid="stBaseButton-secondary"]{
  border:1px solid var(--line); background:var(--card); color:var(--ink); border-radius:.75rem;
  font-size:.9rem; justify-content:flex-start; text-align:left; min-height:3.2rem; }
[data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover{ border-color:#CFCCC1; background:#FDFCF9; color:var(--ink); }

/* ── Composer (bottom) ──────────────────────────────────── */
[data-testid="stBottom"], [data-testid="stBottom"] > div{ background:var(--paper); }
[data-testid="stBottomBlockContainer"]{ max-width:800px; padding-bottom:.4rem; }
[data-testid="stChatInput"]{ background:var(--card); border:1px solid var(--line); border-radius:1.25rem;
  box-shadow:0 1px 2px rgba(0,0,0,.03), 0 6px 24px rgba(31,30,29,.05); }
[data-testid="stChatInput"]:focus-within{ border-color:#CFCCC1; }
[data-testid="stChatInput"] div, [data-testid="stChatInputTextArea"]{ background-color:transparent !important; }
[data-testid="stChatInputTextArea"]{ font-size:1rem; }
[data-testid="stChatInputSubmitButton"]{ background:var(--clay); border-radius:.6rem; }
[data-testid="stChatInputSubmitButton"] svg{ color:#fff; fill:#fff; }
.footnote{ display:flex; justify-content:space-between; color:var(--faint); font-size:.78rem; padding:.35rem .4rem 0; }
.footnote .model{ color:var(--muted); }

/* ── Right "Sources" card ─── */
.side-card{ position:fixed; right:1.2rem; top:3.6rem; width:290px; max-height:70vh; overflow:auto;
  background:var(--card); border:1px solid var(--line); border-radius:1rem; padding:1rem 1.1rem;
  box-shadow:0 6px 24px rgba(31,30,29,.06); z-index:999989; }
.side-card h4{ font-family:'Inter', sans-serif; font-size:.95rem; font-weight:600; margin:0 0 .55rem; padding:0; color:var(--ink); }
.side-card .row{ display:flex; gap:.55rem; align-items:flex-start; font-size:.86rem; color:var(--ink);
  padding:.32rem 0; border-bottom:1px solid #F0EEE6; }
.side-card .row:last-child{ border-bottom:none; }
.side-card .ico{ color:var(--muted); width:1rem; flex:none; }
.side-card .sub{ color:var(--muted); font-size:.8rem; }
.side-card .empty{ color:var(--faint); font-size:.85rem; }
.side-card .sep{ height:1px; background:var(--line); margin:.7rem 0; }
@media (max-width:1300px){ .side-card{ display:none; } }
/* working indicator while a request is in flight */
.working{ display:flex; align-items:center; gap:.6rem; color:var(--muted); font-size:.9rem; padding:.35rem 0; }
.working .pulse{ width:.62rem; height:.62rem; border-radius:50%; background:var(--clay);
  animation:pulse 1.1s ease-in-out infinite; }
@keyframes pulse{ 0%,100%{ opacity:.25; transform:scale(.75); } 50%{ opacity:1; transform:scale(1); } }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Conversation state — each chat has its own backend session_id (own Redis memory)
# ──────────────────────────────────────────────────────────────────────────────
WELCOME_SUGGESTIONS = [
    "Summarize the SPF Splunk proposal",
    "Compare the Cisco and WatchDog proposals",
    "What are the key AI trends for 2024?",
    "What does the SPRI AI brief say about regulation?",
]

def _new_chat() -> str:
    cid = uuid.uuid4().hex[:10]
    st.session_state.chats[cid] = {
        "title": "New chat", "sid": f"ui-{uuid.uuid4()}", "messages": [], "created": time.time(),
    }
    st.session_state.current = cid
    return cid

if "chats" not in st.session_state:
    st.session_state.chats = {}
    _new_chat()
if "composer_nonce" not in st.session_state:
    st.session_state.composer_nonce = 0

chat = st.session_state.chats[st.session_state.current]

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="side-brand">{APP_NAME}</div>', unsafe_allow_html=True)

    if st.button("New chat", icon=":material/add:", type="tertiary", key="new_chat"):
        if chat["messages"]:          # don't stack up empty chats
            _new_chat()
        st.rerun()

    query = st.text_input("Search", placeholder="Search chats", label_visibility="collapsed",
                          key="chat_search", icon=":material/search:")

    st.markdown('<div class="side-label">Recents</div>', unsafe_allow_html=True)
    recents = sorted(st.session_state.chats.items(), key=lambda kv: kv[1]["created"], reverse=True)
    shown = 0
    for cid, c in recents:
        if not c["messages"] and cid != st.session_state.current:
            continue
        if query and query.lower() not in c["title"].lower():
            continue
        is_current = cid == st.session_state.current
        if st.button(c["title"], key=f"chat_{cid}", type="secondary" if is_current else "tertiary"):
            st.session_state.current = cid
            st.rerun()
        shown += 1
    if not shown:
        st.caption("No matching chats")

    st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)

    with st.expander("Knowledge base", icon=":material/folder_open:"):
        uploaded = st.file_uploader("Add documents", type=["pdf", "md", "txt", "html", "htm"],
                                    accept_multiple_files=True, label_visibility="collapsed")
        if uploaded:
            docs_dir = pathlib.Path("data/docs")
            docs_dir.mkdir(parents=True, exist_ok=True)
            for up in uploaded:
                (docs_dir / up.name).write_bytes(up.getbuffer())
            st.success(f"Saved {len(uploaded)} file(s) to {docs_dir}")
        docs = sorted(p.name for p in pathlib.Path("data/docs").glob("*") if p.is_file())
        st.caption(f"{len(docs)} documents indexed from data/docs")
        if st.button("Rebuild index", icon=":material/refresh:", width="stretch"):
            with st.spinner("Embedding documents into Milvus…"):
                try:
                    proc = subprocess.run([sys.executable, "-m", "index.indexer"],
                                          capture_output=True, text=True, check=True)
                    st.success("Index built.")
                    if proc.stdout:
                        st.code(proc.stdout[-1500:])
                except subprocess.CalledProcessError as e:
                    st.error("Indexer failed.")
                    st.code((e.stderr or e.stdout or "")[-1500:])

    with st.expander("Settings", icon=":material/tune:"):
        api_base = st.text_input("Backend URL", value=os.getenv("API_BASE", "http://localhost:8000"))
        show_panel = st.toggle("Show side panel", value=True)
        if st.button("Test backend", icon=":material/lan:", width="stretch"):
            try:
                r = httpx.get(api_base.rstrip("/") + "/health", timeout=5)
                st.success(f"Backend OK ({r.status_code})")
            except Exception as e:
                st.error(f"{e}")
        if st.button("Clear this chat", icon=":material/delete:", width="stretch"):
            chat["messages"] = []
            chat["title"] = "New chat"
            chat["sid"] = f"ui-{uuid.uuid4()}"
            st.rerun()

    initial = html.escape(USER_NAME[:1].upper() or "U")
    st.markdown(
        f'<div class="side-user"><div class="avatar">{initial}</div>'
        f'<div>{html.escape(USER_NAME)} <span class="plan">· RAG demo</span></div></div>',
        unsafe_allow_html=True)

api_base = api_base.rstrip("/")

# ──────────────────────────────────────────────────────────────────────────────
# Backend calls (unchanged contract: /chat streams text, /chat/multimodal is multipart)
# ──────────────────────────────────────────────────────────────────────────────
def stream_text(question: str):
    with httpx.stream("POST", f"{api_base}/chat",
                      json={"question": question, "session_id": chat["sid"]}, timeout=None) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_text():
            if chunk:
                yield chunk

def stream_multimodal(question: str, audio_bytes, audio_mime, image_bytes, image_mime, image_name):
    files = {}
    if audio_bytes:
        files["audio"] = ("recording." + ("wav" if "wav" in audio_mime else "mp3"), audio_bytes, audio_mime)
    if image_bytes:
        files["image"] = (image_name or "image.png", image_bytes, image_mime)
    data = {"question": question or "", "session_id": chat["sid"]}
    with httpx.stream("POST", f"{api_base}/chat/multimodal", data=data, files=files, timeout=None) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_text():
            if chunk:
                yield chunk

def stream_with_status(chunks, label: str) -> str:
    """Stream the answer with a pulsing status line under it until the last chunk arrives.
    The label follows the backend's stages: voice/image step → doc search → writing."""
    body, status = st.empty(), st.empty()
    shown = None
    def show(text: str) -> None:
        nonlocal shown
        if text != shown:
            status.markdown(f'<div class="working"><span class="pulse"></span>{html.escape(text)}</div>',
                            unsafe_allow_html=True)
            shown = text
    show(label)
    answer = ""
    try:
        for chunk in chunks:
            answer += chunk
            body.markdown(answer)
            # Backend echoes "> 🎤 You said" / "> 🖼️ Image understood as" first; the answer follows.
            started = any(l.strip() and not l.lstrip().startswith(">") for l in answer.splitlines())
            show("Writing…" if started else "Searching your docs…")
    finally:
        status.empty()
    return answer

# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────
_CITE_RE = re.compile(r"\[([^\[\]\n]{2,160})\]")

def parse_sources(answer: str) -> list[str]:
    """Collect [source:page]-style tokens the model cites (skips the [EVAL] line)."""
    out = []
    for c in _CITE_RE.findall(answer or ""):
        if c.startswith("EVAL") or c.lower().startswith(("attached image", "stream-error")):
            continue
        if c not in out:
            out.append(c)
    return out[:30]

def _ago(ts: float) -> str:
    s = int(time.time() - ts)
    if s < 60: return "just now"
    if s < 3600: return f"{s // 60} min ago"
    return datetime.fromtimestamp(ts).strftime("%H:%M")

def render_user(m: dict):
    with st.chat_message("user"):
        if m.get("image"):
            st.image(m["image"], width=220)
        if m.get("audio"):
            st.audio(m["audio"], format=m.get("audio_mime", "audio/wav"))
        if m.get("content"):
            st.markdown(m["content"])

def render_meta(m: dict):
    bits = []
    if m.get("elapsed") is not None:
        bits.append(f"{m['elapsed']:.1f}s")
    bits.append(_ago(m.get("ts", time.time())))
    bits.append(CHAT_MODEL)
    st.markdown(f'<div class="meta">{" · ".join(html.escape(b) for b in bits)}</div>', unsafe_allow_html=True)

def render_assistant(m: dict):
    with st.chat_message("assistant"):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander(f"Sources · {len(m['sources'])}", icon=":material/description:"):
                for s in m["sources"]:
                    st.markdown(f"- {s}")
        render_meta(m)

def _title_from(text: str, answer: str, has_audio: bool, has_image: bool) -> str:
    said = re.search(r"You said:\*\*\s*(.+)", answer or "")
    base = (text or (said.group(1) if said else "") or
            ("Voice question" if has_audio else "Image question" if has_image else "New chat"))
    base = re.sub(r"\s+", " ", base).strip()
    return base[:42] + ("…" if len(base) > 42 else "")

# ──────────────────────────────────────────────────────────────────────────────
# One turn: show the user message, stream the answer, store both
# ──────────────────────────────────────────────────────────────────────────────
def run_turn(text: str, audio=None, image=None):
    audio_bytes = audio.getvalue() if audio else None
    audio_mime  = (getattr(audio, "type", None) or "audio/wav") if audio else ""
    image_bytes = image.getvalue() if image else None
    image_mime  = (getattr(image, "type", None) or "image/png") if image else ""
    image_name  = image.name if image else ""

    user_msg = {"role": "user", "content": text or "", "ts": time.time()}
    if audio_bytes: user_msg.update(audio=audio_bytes, audio_mime=audio_mime)
    if image_bytes: user_msg["image"] = image_bytes
    chat["messages"].append(user_msg)
    render_user(user_msg)

    with st.chat_message("assistant"):
        t0 = time.time()
        try:
            if audio_bytes or image_bytes:
                final = stream_with_status(stream_multimodal(text, audio_bytes, audio_mime,
                                                             image_bytes, image_mime, image_name),
                                           "Transcribing your voice…" if audio_bytes else "Reading your image…")
            else:
                final = stream_with_status(stream_text(text), "Searching your docs…")
        except Exception as e:
            st.error(f"Request failed: {e}")
            final = f"_Error: {e}_"
    final = final if isinstance(final, str) else "".join(map(str, final or []))

    chat["messages"].append({"role": "assistant", "content": final, "ts": time.time(),
                             "elapsed": time.time() - t0, "sources": parse_sources(final)})
    if chat["title"] == "New chat":
        chat["title"] = _title_from(text, final, bool(audio_bytes), bool(image_bytes))
    st.session_state.composer_nonce += 1
    st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div class="topbar"><div class="crumb"><span class="proj">{APP_NAME} /</span> '
    f'{html.escape(chat["title"])}</div></div>', unsafe_allow_html=True)

pending = st.session_state.pop("pending_prompt", None)

if not chat["messages"] and not pending:
    hour = datetime.now().hour
    greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
    st.markdown(
        f'<div class="welcome"><h1>{greet}, {html.escape(USER_NAME)}</h1>'
        '<p>Ask about your documents. Type, attach an image, or use the mic.</p></div>',
        unsafe_allow_html=True)
    cols = st.columns(2)
    for i, s in enumerate(WELCOME_SUGGESTIONS):
        if cols[i % 2].button(s, key=f"sugg_{i}", width="stretch"):
            st.session_state.pending_prompt = s
            st.rerun()
else:
    for m in chat["messages"]:
        (render_user if m["role"] == "user" else render_assistant)(m)

# Right-hand card (used-in-this-session panel)
if show_panel:
    kb = sorted(p.name for p in pathlib.Path("data/docs").glob("*") if p.is_file())
    rows = "".join(f'<div class="row"><span class="ico">▤</span><span>{html.escape(n)}</span></div>'
                   for n in kb[:12]) or '<div class="empty">No documents yet. Add some under Knowledge base.</div>'
    turns = sum(1 for m in chat["messages"] if m["role"] == "user")
    st.markdown(
        f'<div class="side-card"><h4>Knowledge base <span class="sub">· {len(kb)} docs</span></h4>{rows}<div class="sep"></div>'
        '<h4>Used in this session</h4>'
        '<div class="row"><span class="ico">◎</span><span>Milvus <span class="sub">· rag_chunks</span></span></div>'
        f'<div class="row"><span class="ico">↺</span><span>Redis memory <span class="sub">· {turns} turn(s)</span></span></div>'
        f'<div class="row"><span class="ico">◇</span><span>{html.escape(CHAT_MODEL)} <span class="sub">· answers</span></span></div>'
        '</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Composer: text + image (＋) + mic
# ──────────────────────────────────────────────────────────────────────────────
submitted = st.chat_input(
    "Reply…" if chat["messages"] else "How can I help you today?",
    accept_file=True, file_type=["png", "jpg", "jpeg"], accept_audio=True,
    key=f"composer_{st.session_state.composer_nonce}",
)
with st.bottom:
    st.markdown(f'<div class="footnote"><span>Answers come from your documents and can be wrong.</span>'
                f'<span class="model">{html.escape(CHAT_MODEL)}</span></div>', unsafe_allow_html=True)

if pending:
    run_turn(pending)
elif submitted:
    text  = (submitted.text or "").strip()
    image = (submitted.files or [None])[0]
    audio = getattr(submitted, "audio", None)
    if text or image or audio:
        run_turn(text, audio=audio, image=image)
