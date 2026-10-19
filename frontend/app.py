"""View layer: Streamlit user interface.
It never runs AI models itself. Every button sends an HTTP request to the FastAPI backend.

Two ways to run:
  * BACKEND_URL is set (e.g. in .env)  -> uses that separate FastAPI server (two terminals)
  * BACKEND_URL is NOT set              -> starts FastAPI inside this app in a background
                                          thread (Streamlit Community Cloud = one server)
Run locally:  streamlit run frontend/app.py
"""
import base64
import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="AI Integrated Web Service", page_icon="🤖", layout="wide")


# ------------------------------------------------------------------ settings
def get_setting(name: str, default: str = "") -> str:
    """Read from Streamlit Cloud secrets first, then from environment / .env."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:          # no secrets.toml file -> ignore
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.getenv(name, default)


# ------------------------------------------------------------------ backend connection
EMBEDDED_PORT = 8000
RUN_EMBEDDED = not get_setting("BACKEND_URL")


@st.cache_resource(show_spinner="Starting the FastAPI backend (first visit only)...")
def start_embedded_backend() -> bool:
    """Start FastAPI (uvicorn) ONCE in a background thread of this server.
    The frontend still talks to it through HTTP, exactly like a separate server."""
    root = Path(__file__).resolve().parent.parent          # project root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL"):         # secrets -> environment
        value = get_setting(key)
        if value:
            os.environ[key] = value

    import uvicorn
    from backend.main import app as fastapi_app            # the same FastAPI app

    server = uvicorn.Server(uvicorn.Config(fastapi_app, host="127.0.0.1",
                                           port=EMBEDDED_PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(60):                                     # wait up to 30 s
        try:
            if requests.get(f"http://127.0.0.1:{EMBEDDED_PORT}/health", timeout=1).ok:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False


if RUN_EMBEDDED:
    start_embedded_backend()
    BACKEND_URL = f"http://127.0.0.1:{EMBEDDED_PORT}"
else:
    BACKEND_URL = get_setting("BACKEND_URL").rstrip("/")


def api(method: str, path: str, timeout: int = 180, **kwargs):
    """Call the backend. Returns (data, error_message)."""
    try:
        r = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
        if r.status_code >= 400:
            try:
                return None, r.json().get("detail", r.text)
            except Exception:
                return None, f"HTTP {r.status_code}: {r.text[:300]}"
        ctype = r.headers.get("content-type", "")
        return (r.json() if "json" in ctype else r.text), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to the backend at {BACKEND_URL}. Is it running?"
    except requests.exceptions.Timeout:
        return None, "The backend took too long. The first request loads the model; try again."


@st.cache_data(ttl=600, show_spinner=False)
def _cached_get(path: str):
    """Only SUCCESSFUL results are cached (10 min). Errors raise, so they are never cached."""
    data, err = api("GET", path, timeout=300)
    if err:
        raise RuntimeError(err)
    return data


def cached_get(path: str):
    """GET requests whose results rarely change (EDA, model comparison)."""
    try:
        return _cached_get(path), None
    except RuntimeError as e:
        return None, str(e)


# ------------------------------------------------------------------ sidebar
health, health_err = api("GET", "/health", timeout=60)
with st.sidebar:
    st.header("System status")
    if health:
        st.success("FastAPI backend connected")
        if health["gemini_key_set"]:
            st.success("Gemini API key set")
        else:
            st.error("Gemini API key missing")
        st.caption(f"Gemini model: {health['gemini_model']}")
    else:
        st.error("Backend not connected")
        st.caption(health_err)
    if RUN_EMBEDDED:
        st.caption("Mode: FastAPI runs inside this server (embedded), "
                    f"called over HTTP at {BACKEND_URL}")
    else:
        st.caption(f"Backend URL: {BACKEND_URL}")
        st.markdown(f"[Open API docs]({BACKEND_URL}/docs)")
    st.divider()
    st.caption("Free servers sleep when idle. The first visit after a pause can take "
               "1-2 minutes, and the first CNN / YOLO request loads the model.")

st.title("AI + Data Analysis Integrated Web Service")
st.caption("FastAPI backend + Streamlit frontend. Five AI modules in one service.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Gemini auto-reply", "2. Data analysis & ML", "3. CNN classification",
    "4. YOLO detection", "5. RAG document Q&A"])


# ================================================================== 2.1
with tab1:
    sub1, sub2, sub3 = st.tabs(["Comment dashboard", "Webhook test", "System settings"])

    with sub1:
        c1, c2 = st.columns([1, 5])
        if c1.button("Refresh", key="refresh_comments"):
            st.rerun()
        comments, err = api("GET", "/comments")
        if err:
            st.error(err)
        elif not comments:
            st.info("No comments yet. Send one from the 'Webhook test' tab.")
        else:
            df = pd.DataFrame(comments)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total", len(df))
            m2.metric("Replied", int((df["status"] == "replied").sum()))
            m3.metric("Pending / error", int((df["status"] != "replied").sum()))
            st.dataframe(df[["comment_id", "platform", "author", "content", "status",
                             "ai_reply", "received_at"]], width="stretch",
                         hide_index=True)

            st.subheader("Comment detail")
            pick = st.selectbox("Choose a comment", df["comment_id"].tolist())
            detail, err = api("GET", f"/comments/{pick}")
            if detail:
                st.json(detail)
                if st.button("Generate the reply again", key="regen"):
                    with st.spinner("Asking Gemini..."):
                        _, err = api("POST", f"/comments/{pick}/reply")
                    if err:
                        st.error(err)
                    else:
                        st.rerun()

    with sub2:
        st.subheader("Webhook request simulator")
        st.caption("This form sends the same JSON that a platform would send to "
                   "POST /webhook/comment.")
        with st.form("webhook_form"):
            a, b = st.columns(2)
            comment_id = a.text_input("Comment ID", value=f"cmt_{pd.Timestamp.now():%H%M%S}")
            platform = b.selectbox("Platform", ["youtube", "instagram", "facebook", "website"])
            post_id = a.text_input("Post ID", value="post_001")
            author = b.text_input("Author", value="Guest")
            content = st.text_area("Content", value="Hello! How long does it take to "
                                                    "receive my order?")
            auto = st.checkbox("Generate the AI reply automatically", value=True)
            sent = st.form_submit_button("Send webhook")
        if sent:
            payload = dict(comment_id=comment_id, platform=platform, post_id=post_id,
                           author=author, content=content)
            with st.spinner("Saving the comment and asking Gemini..."):
                data, err = api("POST", f"/webhook/comment?auto_reply={str(auto).lower()}",
                                json=payload)
            if err:
                st.error(err)
            else:
                (st.success if data["status"] == "replied" else st.warning)(
                    f"Saved. Status: {data['status']}")
                if data.get("ai_reply"):
                    st.markdown(f"**AI reply:** {data['ai_reply']}")
                with st.expander("Request JSON"):
                    st.json(payload)
                with st.expander("Response JSON"):
                    st.json(data)

    with sub3:
        st.subheader("System settings")
        st.write(f"**Backend URL:** `{BACKEND_URL}`")
        if health:
            st.write(f"**Gemini model:** `{health['gemini_model']}`")
            st.write(f"**Gemini API key:** {'set' if health['gemini_key_set'] else 'missing'}")
        st.caption("Keys are stored on the server (.env or hosting secrets), never in the browser.")
        if st.button("Test Gemini connection"):
            with st.spinner("Calling Gemini..."):
                data, err = api("POST", "/gemini/test")
            if err:
                st.error(err)
            elif data["ok"]:
                st.success(data["reply"])
            else:
                st.error(data["reply"])
        if st.button("Delete all comments"):
            data, err = api("DELETE", "/comments")
            if err:
                st.error(err)
            else:
                st.success(f"Deleted {data['deleted']} comments.")


# ================================================================== 2.2
with tab2:
    d1, d2, d3 = st.tabs(["Titanic EDA", "MovieLens EDA", "Model comparison & prediction"])

    with d1:
        with st.spinner("Loading Titanic data..."):
            eda, err = cached_get("/ml/eda/titanic")
        if err:
            st.error(err)
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("train.csv rows", eda["shape_train"][0])
            m2.metric("test.csv rows", eda["shape_test"][0])
            m3.metric("Survival rate (train)", f"{eda['survival_rate']:.1%}")
            m4.metric("test.csv has 'Survived'?", "Yes" if eda["test_has_survived"] else "No")
            if not eda["test_has_survived"]:
                st.info("test.csv has no 'Survived' column, so it cannot be scored. "
                        "We measure performance on a hold-out split of train.csv and use "
                        "test.csv only for predictions.")
            st.dataframe(pd.DataFrame(eda["head"]), width="stretch", hide_index=True)

            miss = pd.DataFrame({"train": eda["missing_train"], "test": eda["missing_test"]})
            miss = miss[(miss["train"] > 0) | (miss["test"] > 0)]
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.bar(miss, barmode="group", title="Missing values by column",
                                   labels={"value": "missing", "index": "column"}),
                            width="stretch")
            ages = pd.DataFrame({"Age": eda["ages"],
                                 "Survived": ["Yes" if s else "No" for s in eda["ages_survived"]]})
            c2.plotly_chart(px.histogram(ages, x="Age", color="Survived", nbins=30,
                                         barmode="overlay", title="Age distribution"),
                            width="stretch")

            c3, c4, c5 = st.columns(3)
            for col, key, title in [(c3, "by_sex", "Survival rate by sex"),
                                    (c4, "by_class", "Survival rate by class"),
                                    (c5, "by_age_group", "Survival rate by age group")]:
                s = pd.Series(eda[key], name="rate")
                col.plotly_chart(px.bar(x=s.index, y=s.values, title=title,
                                        labels={"x": "", "y": "survival rate"}),
                                 width="stretch")

            corr = pd.DataFrame(eda["corr"])
            st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                      zmin=-1, zmax=1, title="Correlation (numeric columns)"),
                            width="stretch")
            st.markdown("**Key findings:** women survived much more than men, 1st class "
                        "survived more than 3rd class, and children had a higher rate. "
                        "Fare correlates with class (rich passengers were in 1st class).")

    with d2:
        st.caption("MovieLens 1M (Python for Data Analysis, Chapter 13.2). The summary of "
                   "1 million ratings was computed once with pandas and saved, so the free "
                   "server does not need the memory to redo it.")
        with st.spinner("Analysing 1 million ratings..."):
            ml, err = cached_get("/ml/eda/movielens")
        if err:
            st.error(err)
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Ratings", f"{ml['n_ratings']:,}")
            m2.metric("Users", f"{ml['n_users']:,}")
            m3.metric("Movies", f"{ml['n_movies']:,}")
            c1, c2 = st.columns(2)
            rd = pd.Series(ml["rating_distribution"])
            c1.plotly_chart(px.bar(x=rd.index, y=rd.values, title="Rating distribution (1-5 stars)",
                                   labels={"x": "stars", "y": "count"}), width="stretch")
            ag = pd.Series(ml["mean_rating_by_age"])
            c2.plotly_chart(px.line(x=ag.index, y=ag.values, markers=True,
                                    title="Average rating by age group",
                                    labels={"x": "age group", "y": "average rating"}),
                            width="stretch")
            g = pd.DataFrame(ml["genres"])
            st.plotly_chart(px.bar(g, x="genre", y="count", color="mean",
                                   title="Number of ratings by genre (color = average rating)"),
                            width="stretch")
            c3, c4 = st.columns(2)
            c3.write("**Top movies rated by women** (250+ ratings)")
            c3.dataframe(pd.DataFrame(ml["top_by_women"]), hide_index=True)
            c4.write("**Most disagreed movies** (highest rating standard deviation)")
            c4.dataframe(pd.DataFrame(ml["most_disagreed"]), hide_index=True)
            c5, c6 = st.columns(2)
            c5.write("**Men liked more than women** (diff = M - F)")
            c5.dataframe(pd.DataFrame(ml["men_prefer"]), hide_index=True)
            c6.write("**Women liked more than men**")
            c6.dataframe(pd.DataFrame(ml["women_prefer"]), hide_index=True)

    with d3:
        with st.spinner("Training and comparing 6 models (first time only)..."):
            res, err = cached_get("/ml/compare")
        if err:
            st.error(err)
        else:
            st.subheader("Model comparison (5-fold cross-validation)")
            table = pd.DataFrame(res["comparison"])
            show = table[["Model", "Recall", "F1", "AUROC", "PR-AUC", "Mean of 4"]]
            st.dataframe(show.style.highlight_max(axis=0, subset=show.columns[1:],
                                                  color="#c7e3d4"),
                         width="stretch", hide_index=True)
            long = show.melt(id_vars="Model", value_vars=["Recall", "F1", "AUROC", "PR-AUC"],
                             var_name="Metric", value_name="Score")
            st.plotly_chart(px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                                   range_y=[0.5, 1]), width="stretch")

            st.success(f"Selected model: **{res['best_model']}**. Rule: {res['selection_rule']}.")
            st.subheader("Final performance (hold-out 20% of train.csv)")
            st.dataframe(pd.DataFrame([res["holdout_metrics"]]), hide_index=True)
            st.caption(f"Hold-out size: {res['holdout_size']} passengers the model never saw.")

            st.subheader("Predictions on test.csv")
            st.write(f"test.csv has {res['test_rows']} passengers and "
                     f"{'a' if res['test_has_label'] else 'NO'} 'Survived' label, so we can only "
                     f"predict. Predicted survival rate: **{res['test_predicted_survival_rate']:.1%}**")
            st.dataframe(pd.DataFrame(res["test_preview"]), hide_index=True)
            csv, err = api("GET", "/ml/test-predictions.csv")
            if csv:
                st.download_button("Download all test predictions (CSV)", csv,
                                   "titanic_test_predictions.csv", "text/csv")

            st.subheader("Predict one passenger")
            with st.form("passenger"):
                a, b, c = st.columns(3)
                pclass = a.selectbox("Class", [1, 2, 3], index=2)
                sex = b.selectbox("Sex", ["male", "female"])
                age = c.number_input("Age", 0.0, 100.0, 30.0)
                sibsp = a.number_input("Siblings / spouses on board", 0, 10, 0)
                parch = b.number_input("Parents / children on board", 0, 10, 0)
                fare = c.number_input("Fare", 0.0, 600.0, 15.0)
                emb = a.selectbox("Embarked (S=Southampton, C=Cherbourg, Q=Queenstown)",
                                  ["S", "C", "Q"])
                go = st.form_submit_button("Predict survival")
            if go:
                p, err = api("POST", "/ml/predict", json=dict(
                    Pclass=pclass, Sex=sex, Age=age, SibSp=sibsp, Parch=parch,
                    Fare=fare, Embarked=emb))
                if err:
                    st.error(err)
                elif p["survived"]:
                    st.success(f"Survived (probability {p['probability']:.1%}, {p['model']})")
                else:
                    st.error(f"Did not survive (survival probability {p['probability']:.1%}, "
                             f"{p['model']})")


# ================================================================== image input helper
def image_input(prefix: str):
    """Returns (files_dict, form_dict, preview) for an uploaded file or an image URL."""
    mode = st.radio("Image source", ["Upload a file", "Image URL"], horizontal=True,
                    key=f"{prefix}_mode")
    if mode == "Upload a file":
        f = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"],
                             key=f"{prefix}_file")
        if f:
            return {"file": (f.name, f.getvalue(), f.type)}, {}, f.getvalue()
    else:
        url = st.text_input("Image URL", key=f"{prefix}_url",
                            placeholder="https://.../photo.jpg")
        if url:
            return None, {"url": url}, url
    return None, None, None


# ================================================================== 2.3
with tab3:
    st.subheader("Image classification with ResNet-18 (ImageNet, 1,000 classes)")
    files, form, preview = image_input("cnn")
    if preview is not None and st.button("Classify image", type="primary"):
        with st.spinner("Classifying (the first run loads the model)..."):
            res, err = api("POST", "/cnn/classify", files=files, data=form)
        if err:
            st.error(err)
        else:
            c1, c2 = st.columns([1, 1])
            c1.image(preview, width="stretch")
            c2.markdown(f"### Prediction: {res['class_name']} ({res['probability']:.1%})")
            if res["is_dog_or_cat"]:
                c2.markdown(f":blue[**It looks like a DOG or a CAT!**] ({res['animal']})")
            else:
                c2.markdown(":red[**Classified as another class.**]")
            top5 = pd.DataFrame(res["top5"])
            c2.plotly_chart(px.bar(top5.iloc[::-1], x="probability", y="class_name",
                                   orientation="h", title="Top-5 classes",
                                   labels={"class_name": ""}), width="stretch")


# ================================================================== 2.4
with tab4:
    st.subheader("Object detection with YOLOv8 Nano (COCO, 80 classes)")
    files, form, preview = image_input("yolo")
    conf = st.slider("Minimum confidence", 0.05, 0.9, 0.25, 0.05)
    if preview is not None and st.button("Detect objects", type="primary"):
        with st.spinner("Detecting (the first run loads the model)..."):
            res, err = api("POST", "/yolo/detect", files=files,
                           data={**(form or {}), "conf": str(conf)})
        if err:
            st.error(err)
        else:
            c1, c2 = st.columns(2)
            c1.image(preview, caption="Original", width="stretch")
            c2.image(base64.b64decode(res["annotated_image_base64"]),
                     caption=f"{res['count']} objects found", width="stretch")
            if res["detections"]:
                det = pd.DataFrame(res["detections"])
                st.write("Detected objects: " + ", ".join(
                    f"{d['class_name']} {d['confidence']:.2f}" for d in res["detections"]))
                st.dataframe(det, width="stretch", hide_index=True)
                st.plotly_chart(px.bar(det["class_name"].value_counts(),
                                       title="Objects per class"), width="stretch")
            else:
                st.info("No objects found. Try a lower confidence.")


# ================================================================== 2.5
with tab5:
    st.subheader("Ask questions about your document (RAG)")
    st.caption("Flow: upload -> split into chunks -> embeddings -> FAISS vector DB -> "
               "find the closest chunks -> Gemini writes the answer.")
    status, _ = api("GET", "/rag/status")
    up = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt", "md"], key="rag_file")
    if up and st.button("Build the vector database"):
        with st.spinner("Reading, splitting and embedding ..."):
            status, err = api("POST", "/rag/upload", files={"file": (up.name, up.getvalue())},
                              timeout=300)
        if err:
            st.error(err)
        else:
            st.success(f"Ready: {status['filename']} -> {status['chunks']} chunks")
    if status and status["ready"]:
        st.info(f"Current document: {status['filename']} ({status['chunks']} chunks)")
        q = st.text_input("Your question", placeholder="What is this document about?")
        k = st.slider("Number of chunks to retrieve (top-k)", 1, 8, 4)
        if q and st.button("Ask", type="primary"):
            with st.spinner("Searching and asking Gemini..."):
                res, err = api("POST", "/rag/ask", json={"question": q, "k": k})
            if err:
                st.error(err)
            else:
                st.markdown("#### Answer")
                st.write(res["answer"])
                st.markdown("#### Source chunks")
                for s in res["sources"]:
                    page = f", page {s['page']}" if s["page"] else ""
                    with st.expander(f"#{s['rank']}{page} (distance {s['score']})"):
                        st.write(s["text"])
    else:
        st.info("Upload a document to start.")
