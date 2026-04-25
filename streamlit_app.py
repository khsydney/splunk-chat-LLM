# streamlit_app.py
import os, time, re, subprocess, pathlib
import streamlit as st
import httpx
import uuid

if "sid" not in st.session_state:
    st.session_state.sid = f"ui-{uuid.uuid4()}"

st.set_page_config(page_title="Chat RAG (Streamlit)", layout="wide")
st.title("Chat LLM with RAG – Splunk")

# --- Sidebar controls ---------------------------------------------------------
default_api = os.getenv("API_BASE", "http://localhost:8000")
api_base = st.sidebar.text_input("Backend base URL", value=default_api)
st.sidebar.caption("Your FastAPI server exposing POST /chat (streamed text/plain)")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Test backend"):
        try:
            r = httpx.get(api_base.rstrip("/") + "/docs", timeout=5)
            st.sidebar.success(f"OK {r.status_code}")
        except Exception as e:
            st.sidebar.error(f"{e}")

with col2:
    if st.button("Clear chat"):
        st.session_state.messages = [{"role":"assistant","content":"Hi! Ask me anything."}]
        st.rerun()

# Upload docs → data/docs
st.sidebar.divider()
st.sidebar.subheader("Docs (ingest)")
uploaded = st.sidebar.file_uploader(
    "Upload PDFs / MD / TXT / HTML",
    type=["pdf","md","txt","html","htm"],
    accept_multiple_files=True,
)
if uploaded:
    docs_dir = pathlib.Path("data/docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    for up in uploaded:
        dest = docs_dir / up.name
        dest.write_bytes(up.getbuffer())
    st.sidebar.success(f"Saved {len(uploaded)} file(s) to {docs_dir}")

if st.sidebar.button("Rebuild Index"):
    st.sidebar.write("Running: python -m index.indexer …")
    try:
        proc = subprocess.run(
            ["python","-m","index.indexer"],
            capture_output=True, text=True, check=True
        )
        st.sidebar.success("Index built.")
        st.sidebar.code(proc.stdout or "(no stdout)")
        if proc.stderr:
            st.sidebar.caption("stderr:")
            st.sidebar.code(proc.stderr)
    except subprocess.CalledProcessError as e:
        st.sidebar.error("Indexer failed.")
        st.sidebar.code(e.stdout)
        st.sidebar.code(e.stderr)

st.sidebar.divider()
st.sidebar.caption("Tip: keep the FastAPI backend running with uvicorn.")

# --- Chat state ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! Ask me anything. I’ll search your docs with RAG and cite sources."}
    ]

# render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --- Helper: stream from backend ---------------------------------------------
def stream_from_backend(question: str):
    """
    Yields incrementally growing text from FastAPI /chat,
    which streams text/plain chunks.
    """
    url = api_base.rstrip("/") + "/chat"
    t0 = time.time()
    with httpx.stream("POST", url, json={"question": question, "session_id": st.session_state.sid}, timeout=None) as resp:
        resp.raise_for_status()
        acc = ""
        for chunk in resp.iter_text():
            if not chunk:
                continue
            acc += chunk
            yield acc, False, None
    yield acc, True, time.time() - t0

# --- Chat input ---------------------------------------------------------------
prompt = st.chat_input("Ask a question…")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        final = ""
        elapsed = None
        try:
            for acc, done, et in stream_from_backend(prompt):
                placeholder.markdown(acc)
                final = acc
                if done:
                    elapsed = et
        except Exception as e:
            placeholder.error(f"Request failed: {e}")
            final = f"_Error: {e}_"

        if elapsed is not None:
            st.caption(f"Response time: {elapsed:.2f}s")

        # Basic source parser: collects [source:page] tokens from the answer
        cites = sorted(set(re.findall(r"\[([^\[\]\n]+)\]", final)))
        if cites:
            with st.expander("Sources (parsed from answer)"):
                for c in cites[:30]:
                    st.write("•", c)

    st.session_state.messages.append({"role":"assistant","content":final})
    st.rerun()
