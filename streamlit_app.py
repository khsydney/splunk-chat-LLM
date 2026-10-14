# streamlit_app.py
import os, time, re, subprocess, pathlib
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import httpx
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
HTTPXClientInstrumentor().instrument()
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
        if m.get("audio"):
            st.audio(m["audio"], format=m.get("audio_mime", "audio/wav"))
        if m.get("image"):
            st.image(m["image"], width=240)
        if m.get("content"):
            st.markdown(m["content"])

# --- Helper: stream from backend ---------------------------------------------
def stream_from_backend(question: str):
    """Yields raw text chunks from FastAPI /chat for use with st.write_stream."""
    url = api_base.rstrip("/") + "/chat"
    with httpx.stream("POST", url, json={"question": question, "session_id": st.session_state.sid}, timeout=None) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_text():
            if chunk:
                yield chunk

def stream_multimodal(question: str, audio_bytes: bytes | None, audio_mime: str,
                      image_bytes: bytes | None, image_mime: str, image_name: str):
    """POST /chat/multimodal (multipart) — voice and/or image + optional text."""
    url = api_base.rstrip("/") + "/chat/multimodal"
    files = {}
    if audio_bytes:
        files["audio"] = ("recording." + ("wav" if "wav" in audio_mime else "mp3"), audio_bytes, audio_mime)
    if image_bytes:
        files["image"] = (image_name or "image.png", image_bytes, image_mime)
    data = {"question": question or "", "session_id": st.session_state.sid}
    with httpx.stream("POST", url, data=data, files=files, timeout=None) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_text():
            if chunk:
                yield chunk

# --- Voice & image input (multimodal) -----------------------------------------
# Galileo logs these as attachments on the trace: audio shows as a waveform player,
# images render inline (docs.galileo.ai/concepts/logging/multimodal-observability).
if "mm_nonce" not in st.session_state:
    st.session_state.mm_nonce = 0   # bump to reset the widgets after a send

with st.expander("🎤 Voice & 🖼️ image (multimodal)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        if hasattr(st, "audio_input"):
            voice = st.audio_input("Record your question", key=f"voice_{st.session_state.mm_nonce}")
        else:
            voice = None
            st.caption("Upgrade Streamlit (>=1.39) for the microphone recorder.")
    with c2:
        img = st.file_uploader("Attach an image (png / jpg)", type=["png", "jpg", "jpeg"],
                               key=f"img_{st.session_state.mm_nonce}")
    mm_text = st.text_input("Optional text to go with it", key=f"mmtext_{st.session_state.mm_nonce}",
                            placeholder="e.g. What does this screenshot show?")
    send_mm = st.button("Send voice / image", type="primary",
                        disabled=not (voice or img), key=f"send_{st.session_state.mm_nonce}")

if send_mm and (voice or img):
    audio_bytes = voice.getvalue() if voice else None
    audio_mime  = (getattr(voice, "type", None) or "audio/wav") if voice else ""
    image_bytes = img.getvalue() if img else None
    image_mime  = (getattr(img, "type", None) or "image/png") if img else ""
    image_name  = img.name if img else ""

    user_msg = {"role": "user", "content": mm_text or ""}
    if audio_bytes:
        user_msg["audio"] = audio_bytes; user_msg["audio_mime"] = audio_mime
    if image_bytes:
        user_msg["image"] = image_bytes
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        if audio_bytes: st.audio(audio_bytes, format=audio_mime)
        if image_bytes: st.image(image_bytes, width=240)
        if mm_text: st.markdown(mm_text)

    with st.chat_message("assistant"):
        t0 = time.time()
        try:
            final = st.write_stream(stream_multimodal(mm_text, audio_bytes, audio_mime,
                                                     image_bytes, image_mime, image_name))
        except Exception as e:
            st.error(f"Request failed: {e}")
            final = f"_Error: {e}_"
        st.caption(f"Response time: {time.time() - t0:.2f}s")

    st.session_state.messages.append({"role": "assistant", "content": final})
    st.session_state.mm_nonce += 1   # clear recorder / uploader for the next turn
    st.rerun()

# --- Chat input ---------------------------------------------------------------
prompt = st.chat_input("Ask a question…")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.time()
        try:
            final = st.write_stream(stream_from_backend(prompt))
        except Exception as e:
            st.error(f"Request failed: {e}")
            final = f"_Error: {e}_"
        st.caption(f"Response time: {time.time() - t0:.2f}s")

        # Basic source parser: collects [source:page] tokens from the answer
        cites = sorted(set(re.findall(r"\[([^\[\]\n]+)\]", final)))
        if cites:
            with st.expander("Sources (parsed from answer)"):
                for c in cites[:30]:
                    st.write("•", c)

    st.session_state.messages.append({"role":"assistant","content":final})
    st.rerun()
