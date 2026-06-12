# app.py  --  Goodreads Book Recommender (Streamlit)  [enhanced UI preview]
# Run:  streamlit run app.py
# Requires Books.csv and Ratings.csv in the same folder.
# Key resolution order: st.secrets -> env var -> user-entered (BYO) key. Never hard-coded/committed.

import os, random, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from pydantic import BaseModel, Field
from surprise import Dataset, Reader, KNNWithMeans

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Goodreads Recommender", page_icon="\U0001F4DA", layout="wide")

GEMINI_MODEL = "gemini-2.5-flash-lite"   # the Week-4 class model (generous free limits)
MIN_RATINGS  = 20
GEM_MIN, GEM_MAX, GEM_AVG = 5, 9, 4.0
MOODS = ["Funny", "Dark & gripping", "Cozy", "Epic adventure", "Romantic", "Thought-provoking"]


class RankedBook(BaseModel):                       # structured-output schema (Week-4 pattern)
    title: str = Field(description="Exact book title copied from the candidate list.")
    reason: str = Field(description="One sentence on why it fits the reader's preference.")


def _rerun():
    fn = getattr(st, "rerun", getattr(st, "experimental_rerun", None))
    if fn:
        fn()


def _img(row):
    """Render a book cover from image_url, or a book emoji placeholder."""
    url = row.get("image_url", "")
    if isinstance(url, str) and url.startswith("http"):
        st.image(url, use_container_width=True)
    else:
        st.markdown("<div style='font-size:64px;text-align:center'>\U0001F4DA</div>",
                    unsafe_allow_html=True)


# ---------- Data ----------
@st.cache_data(show_spinner=False)
def load_data():
    base = Path(__file__).parent
    books   = pd.read_csv(base / "Books.csv",   on_bad_lines="skip")
    ratings = pd.read_csv(base / "Ratings.csv", on_bad_lines="skip")
    for df in (books, ratings):
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last")
    ratings = ratings[ratings["rating"].between(1, 5)].dropna(subset=["user_id", "book_id", "rating"])
    ratings["book_id"] = ratings["book_id"].astype(int)
    books["book_id"]   = books["book_id"].astype(int)
    ratings = ratings[ratings["book_id"].isin(set(books["book_id"]))]
    uc = ratings["user_id"].value_counts()
    bc = ratings["book_id"].value_counts()
    ratings = ratings[ratings["user_id"].isin(uc[uc >= 5].index) &
                      ratings["book_id"].isin(bc[bc >= 5].index)]
    return books, ratings


# ---------- Model ----------
@st.cache_resource(show_spinner=False)
def train_model(ratings_df):
    # Item-based collaborative filtering, MEAN-CENTRED (KNNWithMeans, Pearson, k=40):
    # predicts each item's mean + the similarity-weighted deviations of the user's rated
    # neighbour items -- the best-ranking model in the bake-off (see notebook 5.4).
    reader = Reader(rating_scale=(1, 5))
    data   = Dataset.load_from_df(ratings_df[["user_id", "book_id", "rating"]], reader)
    model  = KNNWithMeans(k=40, sim_options={"name": "pearson", "user_based": False}, verbose=False)
    model.fit(data.build_full_trainset())
    return model


# ---------- CF Top-N ----------
def get_top_n(model, user_id, books_df, ratings_df, candidate_ids, n=20):
    rated      = set(ratings_df[ratings_df["user_id"] == user_id]["book_id"])
    candidates = [b for b in candidate_ids if b not in rated]
    preds      = [(bid, model.predict(user_id, bid).est) for bid in candidates]
    preds.sort(key=lambda x: x[1], reverse=True)
    top_ids    = [b for b, _ in preds[:n]]
    scores     = {b: s for b, s in preds[:n]}
    result     = books_df[books_df["book_id"].isin(top_ids)].copy()
    result["predicted_rating"] = result["book_id"].map(scores)
    return result.sort_values("predicted_rating", ascending=False)[
        ["book_id", "title", "authors", "original_publication_year",
         "average_rating", "predicted_rating", "image_url"]
    ].reset_index(drop=True)


# ---------- Hidden Gem (separate from the CF list) ----------
@st.cache_data(show_spinner=False)
def rare_find_pool(ratings_df):
    loc = ratings_df.groupby("book_id")["rating"].agg(["count", "mean"])
    mask = loc["count"].between(GEM_MIN, GEM_MAX) & (loc["mean"] >= GEM_AVG)
    return set(loc[mask].index)


def pick_hidden_gem(user_id, pool, ratings_df, books_df, model, salt=0):
    rated = set(ratings_df[ratings_df["user_id"] == user_id]["book_id"])
    cands = sorted(b for b in pool if b not in rated)
    if not cands:
        return None
    bid = random.Random(f"{user_id}-{salt}").choice(cands)
    row = books_df[books_df["book_id"] == bid].iloc[0]
    yr  = int(row["original_publication_year"]) if pd.notna(row["original_publication_year"]) else "N/A"
    return {"title": row["title"], "authors": row["authors"], "year": yr,
            "avg": row["average_rating"], "image_url": row.get("image_url", ""),
            "predicted": model.predict(user_id, bid).est}


def gem_blurb(gem, preference, api_key):
    # Plain-text generate_content (Week-4 demo pattern), new google-genai client.
    client = genai.Client(api_key=api_key)
    prompt = (f"In ONE enthusiastic sentence, pitch this lesser-known book to a reader in the mood "
              f"for '{preference}'. Book: {gem['title']} by {gem['authors']} ({gem['year']}). "
              f"No quotation marks.")
    try:
        return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text.strip()
    except Exception:
        return "A quietly loved pick that few readers here have discovered yet."


# ---------- LLM re-rank (graded CF -> LLM step; Week-4 structured-output pattern) ----------
def llm_rerank(candidates_df, user_preference, api_key):
    # Structured-output re-ranker (Week-4 class pattern): Pydantic schema + response.parsed,
    # so there is no manual JSON parsing. The returned list order IS the ranking, best first.
    # We still enforce the assignment's rule: keep only candidate titles, re-append any the
    # model drops, and fall back to CF order on any error.
    client = genai.Client(api_key=api_key)
    meta  = ["title", "authors", "original_publication_year", "average_rating",
             "predicted_rating", "image_url"]
    valid = list(candidates_df["title"])

    catalog = "\n".join(
        f"- {row['title']} | {row['authors']} | "
        f"{int(row['original_publication_year']) if pd.notna(row['original_publication_year']) else 'N/A'} | "
        f"avg {row['average_rating']:.2f}"
        for _, row in candidates_df.iterrows()
    )
    system_instruction = (
        "You are a book recommendation assistant. RE-RANK the candidate books to best match the "
        "reader's preference, best first. Use ONLY the exact titles from the list and include EVERY "
        "candidate exactly once, each with a one-sentence reason."
    )

    def fallback(note):
        out = candidates_df[meta].copy().reset_index(drop=True)
        out.insert(0, "rank", range(1, len(out) + 1))
        out["explanation"] = note
        return out

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Reader's preference: {user_preference}\n\nCandidate books:\n{catalog}",
            config={
                "system_instruction": system_instruction,
                "response_mime_type": "application/json",
                "response_schema": list[RankedBook],
                "temperature": 0.2,
            },
        )
        picks = resp.parsed or []
    except Exception:
        return fallback("CF order (LLM unavailable).")

    seen, ordered = set(), []
    for p in picks:                         # response.parsed -> typed RankedBook objects
        t = (getattr(p, "title", "") or "").strip()
        if t in valid and t not in seen:
            seen.add(t)
            ordered.append((t, (getattr(p, "reason", "") or "").strip()))
    for t in valid:
        if t not in seen:
            ordered.append((t, "(retained from CF list)"))

    out = pd.DataFrame(ordered, columns=["title", "explanation"])
    out.insert(0, "rank", range(1, len(out) + 1))
    out = out.merge(candidates_df[meta], on="title", how="left")
    return out.sort_values("rank").reset_index(drop=True)


def get_configured_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"] # type: ignore
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


# ============================ APP ============================
books_df, ratings_df = load_data()
top_users = ratings_df["user_id"].value_counts().head(300).index.tolist()

# ---------- Landing gate (closed book) ----------
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.markdown("<div style='text-align:center; font-size:130px; margin-top:30px'>\U0001F4D5</div>",
                unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center; margin-top:-10px'>Goodreads Book Recommender</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray'>"
                "Collaborative Filtering (item-based KNN) + AI Personalisation (Gemini)</p>",
                unsafe_allow_html=True)

    mid = st.columns([1, 2, 1])[1]
    with mid:
        chosen = st.selectbox("Choose a reader to begin", top_users)
        c1, c2 = st.columns(2)
        if c1.button("\U0001F4D6 Open the book", type="primary", use_container_width=True):
            st.session_state.started, st.session_state.user = True, chosen
            _rerun()
        if c2.button("\U0001F3B2 Surprise me", use_container_width=True):
            st.session_state.started, st.session_state.user = True, random.choice(top_users)
            _rerun()
    st.stop()   # nothing below renders until a reader is confirmed

# ---------- Main app (after the book "opens") ----------
selected_user = st.session_state.user
with st.spinner("\U0001F4D6 Opening the book and training the model..."):
    model = train_model(ratings_df)          # cached after first run

api_key_in  = get_configured_key()
bc_full     = ratings_df["book_id"].value_counts()
popular_ids = set(bc_full[bc_full >= MIN_RATINGS].index)
rare_pool   = rare_find_pool(ratings_df)

# Sidebar
st.sidebar.header("Settings")
st.sidebar.markdown(f"**Reader:** `{selected_user}`")
if st.sidebar.button("\U0001F501 Change reader"):
    st.session_state.started = False
    _rerun()
top_n = st.sidebar.slider("# Recommendations", 5, 30, 10)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**CF Model:** Item-Based CF (KNNWithMeans, Pearson)  \n**LLM:** {GEMINI_MODEL} (Google)")

st.markdown(f"## \U0001F4D6 Reading recommendations for `{selected_user}`")

with st.expander("This reader's rating history"):
    hist = (ratings_df[ratings_df["user_id"] == selected_user]
            .merge(books_df[["book_id", "title", "authors"]], on="book_id")
            [["title", "authors", "rating"]]
            .sort_values("rating", ascending=False))
    st.dataframe(hist.head(20), use_container_width=True)

# --- Mood chips (one-click presets that fill the preference box) ---
st.markdown("**Set a mood (optional):**")
chip_cols = st.columns(len(MOODS))
for col, mood in zip(chip_cols, MOODS):
    if col.button(mood, use_container_width=True):
        st.session_state["pref"] = mood.lower()
        _rerun()

# --- CF recommendations as a cover grid ---
st.subheader("\U0001F916 Collaborative Filtering Picks")
cf_recs = get_top_n(model, selected_user, books_df, ratings_df, popular_ids, n=top_n)
PER_ROW = 5
for start in range(0, len(cf_recs), PER_ROW):
    chunk = cf_recs.iloc[start:start + PER_ROW]
    grid  = st.columns(PER_ROW)
    for col, (idx, row) in zip(grid, chunk.iterrows()):
        with col:
            _img(row)
            title = row["title"] if len(str(row["title"])) <= 38 else str(row["title"])[:37] + "\u2026"
            st.markdown(f"**{idx + 1}. {title}**  \n"
                        f"<span style='color:gray'>{str(row['authors'])[:28]}</span>  \n"
                        f"\U0001F52E {row['predicted_rating']:.2f}", unsafe_allow_html=True)

# --- Hidden Gem (serendipity pick; separate from the CF list) ---
st.divider()
st.subheader("\U0001F48E Hidden Gem for You")
st.caption(f"Loved (avg \u2265 {GEM_AVG:.0f}) but rated by only {GEM_MIN}\u2013{GEM_MAX} readers here \u2014 "
           "a serendipity pick, separate from the CF list above.")
salt_key = f"gemsalt_{selected_user}"
gem = pick_hidden_gem(selected_user, rare_pool, ratings_df, books_df, model,
                      salt=st.session_state.get(salt_key, 0))
if gem:
    gcols = st.columns([1, 4])
    with gcols[0]:
        _img(gem)
    with gcols[1]:
        st.markdown(
            f"**{gem['title']}** — {gem['authors']}  \n"
            f"Year: {gem['year']} | Goodreads Avg: {gem['avg']:.2f} | "
            f"\U0001F52E Predicted for you: {gem['predicted']:.2f}"
        )
        b1, b2 = st.columns(2)
        if b1.button("\U0001F3B2 Shuffle gem"):
            st.session_state[salt_key] = st.session_state.get(salt_key, 0) + 1
            _rerun()
        if b2.button("\U00002728 Why you'd love it (AI)"):
            if not api_key_in:
                st.warning("No Gemini key configured.")
            else:
                mood = st.session_state.get("pref", "").strip() or "a good, surprising read"
                with st.spinner("Asking Gemini..."):
                    st.success(gem_blurb(gem, mood, api_key_in))
else:
    st.info("This reader has already rated every hidden gem in the pool!")

# --- AI re-ranking of the CF list (the graded step) ---
st.divider()
st.subheader("\U00002728 AI Re-ranking (Gemini)")
if api_key_in:
    st.caption("Using the app's configured Gemini key.")
else:
    st.caption("No key configured. Paste your own Gemini key to enable AI features "
               "(used only for this session, never stored).")
    api_key_in = st.text_input("Gemini API Key", type="password")

pref_in = st.text_area("Your mood / preference", key="pref",
                       placeholder="e.g. funny and lighthearted", height=70)

if st.button("Re-rank with AI", type="primary"):
    if not api_key_in:
        st.error("Please enter your Gemini API key.")
    elif not pref_in.strip():
        st.warning("Pick a mood chip above or describe your preference.")
    else:
        with st.spinner("Consulting Gemini..."):
            try:
                reranked = llm_rerank(cf_recs, pref_in, api_key_in)
                st.success(f"Re-ranked based on: {pref_in}")
                for _, row in reranked.iterrows():
                    rc = st.columns([1, 5])
                    with rc[0]:
                        _img(row)
                    with rc[1]:
                        yr = int(row["original_publication_year"]) if pd.notna(row["original_publication_year"]) else "N/A"
                        st.markdown(
                            f"**#{int(row['rank'])}: {row['title']}** — {row['authors']}  \n"
                            f"*{row['explanation']}*  \n"
                            f"Avg Rating: {row['average_rating']:.2f} | Year: {yr}"
                        )
            except Exception as e:
                st.error(f"LLM error: {e}")
