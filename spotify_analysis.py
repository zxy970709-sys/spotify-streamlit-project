# =============================================
# spotify_app.py – Spotify Project Web App
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Spotify Data Explorer", layout="wide")

# ---------- GLOBAL CSS (center DataFrames) ----------
st.markdown("""
    <style>
    .stDataFrame table {
        text-align: center !important;
    }
    .stDataFrame th, .stDataFrame td {
        text-align: center !important;
        vertical-align: middle !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
st.title("🎵 Spotify Dataset Interactive Web App")

csv_path = "spotify.csv"
try:
    df = pd.read_csv(csv_path)
    st.success(f"Loaded '{csv_path}' successfully. Total rows: {df.shape[0]}")
except FileNotFoundError:
    st.error(f"Could not find '{csv_path}'. Make sure it is in the same folder as this app.")
    st.stop()

# ---------- FEATURE ENGINEERING ----------
# 1) Duration as mm:ss
if "duration_ms" in df.columns:
    df["duration_m:s"] = df["duration_ms"].apply(
        lambda x: f"{int(x // 60000)}:{int((x % 60000) // 1000):02d}"
    )

# 2) Musical scale from key + mode (e.g., C# Minor)
key_dict = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
    6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
}
mode_dict = {0: 'Minor', 1: 'Major'}

def key_mode_to_scale(row):
    if "key" not in row or "mode" not in row:
        return None
    key = row["key"]
    mode = row["mode"]
    if pd.isna(key) or pd.isna(mode):
        return None
    try:
        return f"{key_dict[int(key)]} {mode_dict[int(mode)]}"
    except Exception:
        return None

if "key" in df.columns and "mode" in df.columns:
    df["scale"] = df.apply(key_mode_to_scale, axis=1)

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("🔍 Filter Options")

if "genre" in df.columns:
    genres = sorted(df["genre"].dropna().unique().tolist())
else:
    genres = []

selected_genres = st.sidebar.multiselect(
    "Filter by genre:",
    options=genres,
    default=genres[:3] if genres else []
)

popularity_threshold = st.sidebar.slider(
    "Minimum popularity:",
    min_value=0,
    max_value=100,
    value=60
)

filtered_df = df.copy()
if selected_genres:
    filtered_df = filtered_df[filtered_df["genre"].isin(selected_genres)]
if "popularity" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["popularity"] >= popularity_threshold]

st.write(f"### Tracks after filtering: **{filtered_df.shape[0]}**")

# =====================================================
# === EXAMPLE TRACKS + ARTIST / SIMILARITY + SEARCH ===
# =====================================================
st.subheader("🎧 Example Tracks for Current Filters")

if filtered_df is None or filtered_df.empty:
    st.warning("No tracks match the current filters. Try selecting more genres or lowering the popularity threshold.")
else:
    # ---------- Apply popularity threshold first ----------
    if "popularity" in filtered_df.columns:
        examples_raw = filtered_df[filtered_df["popularity"] >= popularity_threshold].copy()
    else:
        examples_raw = filtered_df.copy()

    if examples_raw.empty:
        st.warning("No tracks match the current filters with this popularity threshold.")
    else:
        # ---------- Keep original df index for later lookups ----------
        examples_raw = examples_raw.reset_index().rename(columns={"index": "df_index"})

        # Detect track & artist columns
        track_col = None
        if "track_name" in examples_raw.columns:
            track_col = "track_name"
        elif "track" in examples_raw.columns:
            track_col = "track"

        artist_col = None
        if "artist" in examples_raw.columns:
            artist_col = "artist"
        elif "artist_name" in examples_raw.columns:
            artist_col = "artist_name"

        # ---------- Group by track + artist to avoid duplicates ----------
        group_cols = []
        if track_col:
            group_cols.append(track_col)
        if artist_col:
            group_cols.append(artist_col)
        if not group_cols:
            group_cols = ["df_index"]

        import pandas as pd

        def join_genres(series):
            vals = [str(x) for x in series.dropna().unique()]
            return ", ".join(sorted(vals)) if vals else ""

        agg_dict = {}
        for col in examples_raw.columns:
            if col in group_cols:
                continue
            if col == "genre":
                agg_dict[col] = join_genres
            elif col == "df_index":
                agg_dict[col] = "first"
            elif col == "popularity":
                agg_dict[col] = "max"
            else:
                if pd.api.types.is_numeric_dtype(examples_raw[col]):
                    agg_dict[col] = "mean"
                else:
                    agg_dict[col] = "first"

        examples_grouped = examples_raw.groupby(group_cols, as_index=False).agg(agg_dict)

        total_matches = examples_grouped.shape[0]
        st.write(
            f"Unique tracks matching current filters (genre + popularity ≥ {popularity_threshold}): "
            f"**{total_matches}**"
        )

        if total_matches == 0:
            st.warning("No unique tracks remain after grouping.")
        else:
            # Let user choose how many rows to show
            default_n = 20 if total_matches >= 20 else total_matches
            n_rows_to_show = st.slider(
                "Number of example tracks to display",
                min_value=1,
                max_value=min(200, total_matches),
                value=default_n,
            )

            # Sort by popularity if present
            if "popularity" in examples_grouped.columns:
                examples_grouped = examples_grouped.sort_values("popularity", ascending=False)

            # ---------- Correlation for top-related features ----------
            top_related_features = []
            if "popularity" in examples_grouped.columns:
                numeric_cols = examples_grouped.select_dtypes(
                    include=["int64", "float64", "int32", "float32"]
                ).columns.tolist()
                feature_candidates = [c for c in numeric_cols if c != "popularity"]
                if feature_candidates:
                    corrs = examples_grouped[feature_candidates].corrwith(
                        examples_grouped["popularity"]
                    ).abs().dropna()
                    if not corrs.empty:
                        top_related_features = list(
                            corrs.sort_values(ascending=False).head(3).index
                        )

            # ---------- Build display table ----------
            display_base_df = examples_grouped.head(n_rows_to_show).copy()

            preferred_order = [
                track_col,
                artist_col,
                "genre",
                "popularity",
                "duration_m:s",
                "danceability",
                "energy",
                "tempo",
                "scale",
                "valence",
                "acousticness",
                "speechiness",
                "instrumentalness",
                "liveness",
                "df_index",  # internal
            ]
            display_cols = [c for c in preferred_order if c and c in display_base_df.columns]
            if display_cols:
                display_base_df = display_base_df[display_cols]

            # UI DataFrame: hide df_index from display
            ui_df = display_base_df.drop(columns=["df_index"], errors="ignore")

            # Center-align all table text via CSS
            st.markdown(
                """
<style>
.dataframe td, .dataframe th {
    text-align: center !important;
}
</style>
""",
                unsafe_allow_html=True,
            )

            # Highlight top 3 most popularity-related features
            def highlight_top_cols(col):
                if col.name in top_related_features:
                    return ['background-color: rgba(255, 0, 0, 0.25)'] * len(col)
                else:
                    return [''] * len(col)

            styled_examples = ui_df.style.apply(highlight_top_cols, axis=0)

            if top_related_features:
                st.caption(
                    "Columns highlighted in light red are the **three features most correlated "
                    f"with popularity** for the current filtered subset: "
                    f"`{', '.join(top_related_features)}`."
                )

            st.dataframe(styled_examples, use_container_width=True)

            # =====================================================
            # === ARTIST DROPDOWN: TOP SONGS FROM EXAMPLES ========
            # =====================================================
            st.subheader("🎤 Explore Artists from These Examples")

            if artist_col is not None and artist_col in display_base_df.columns:
                example_artists = display_base_df[artist_col].dropna().unique().tolist()

                if example_artists:
                    selected_artist = st.selectbox(
                        "Choose an artist (from the example tracks above):",
                        sorted(example_artists),
                        index=0,
                    )

                    artist_tracks = df[df[artist_col] == selected_artist].copy()

                    if artist_tracks.empty:
                        st.info(f"No tracks for **{selected_artist}** found in the dataset.")
                    else:
                        # Deduplicate artist tracks by (track, artist) and merge genres
                        artist_tracks = artist_tracks.reset_index().rename(columns={"index": "df_index"})

                        agg_dict_at = {}
                        for col in artist_tracks.columns:
                            if col == "genre":
                                agg_dict_at[col] = join_genres
                            elif col == "df_index":
                                agg_dict_at[col] = "first"
                            elif col == "popularity":
                                agg_dict_at[col] = "max"
                            else:
                                if pd.api.types.is_numeric_dtype(artist_tracks[col]):
                                    agg_dict_at[col] = "mean"
                                else:
                                    agg_dict_at[col] = "first"

                        group_cols_at = []
                        if track_col and track_col in artist_tracks.columns:
                            group_cols_at.append(track_col)
                        if artist_col and artist_col in artist_tracks.columns:
                            group_cols_at.append(artist_col)
                        if not group_cols_at:
                            group_cols_at = ["df_index"]

                        artist_grouped = artist_tracks.groupby(group_cols_at, as_index=False).agg(agg_dict_at)

                        if "popularity" in artist_grouped.columns:
                            artist_grouped = artist_grouped.sort_values("popularity", ascending=False)

                        top_n_artist = min(5, artist_grouped.shape[0])

                        st.write(
                            f"Top **{top_n_artist}** tracks by **{selected_artist}** in this dataset "
                            "(deduplicated by track, with all genres merged):"
                        )

                        artist_display_cols = []
                        for col in [
                            track_col,
                            artist_col,
                            "genre",
                            "popularity",
                            "duration_m:s",
                            "danceability",
                            "energy",
                            "tempo",
                            "scale",
                            "valence",
                        ]:
                            if col in artist_grouped.columns and col not in artist_display_cols:
                                artist_display_cols.append(col)

                        st.dataframe(
                            artist_grouped[artist_display_cols].head(top_n_artist)
                            if artist_display_cols
                            else artist_grouped.head(top_n_artist),
                            use_container_width=True,
                        )
                else:
                    st.info("No artists found in the current example table.")
            else:
                st.warning("No artist column found; cannot build artist explorer.")

            # =====================================================
            # === SIMILAR TRACK FINDER (BASED ON EXAMPLES) ========
            # =====================================================
            st.subheader("🎯 Find Similar Tracks to One Example")

            if track_col is not None and track_col in display_base_df.columns:
                temp_df = display_base_df.copy()
                if artist_col in temp_df.columns:
                    temp_df["track_label"] = temp_df[track_col].astype(str) + " — " + temp_df[artist_col].astype(str)
                else:
                    temp_df["track_label"] = temp_df[track_col].astype(str)

                track_options = temp_df["track_label"].tolist()

                selected_track_label = st.selectbox(
                    "Choose a track from the examples above:",
                    track_options,
                )

                if selected_track_label:
                    selected_row_base = temp_df[temp_df["track_label"] == selected_track_label].iloc[0]

                    df_index_col = "df_index" if "df_index" in selected_row_base.index else None

                    similarity_features = [
                        "danceability", "energy", "valence", "tempo", "loudness",
                        "acousticness", "speechiness", "instrumentalness", "liveness"
                    ]
                    similarity_features = [c for c in similarity_features if c in df.columns]

                    if df_index_col is None or not similarity_features:
                        st.info("Not enough information to compute similarity for this track.")
                    else:
                        import numpy as np

                        try:
                            selected_row_full = df.loc[selected_row_base[df_index_col]]
                        except KeyError:
                            selected_row_full = None

                        if selected_row_full is None:
                            st.info("Could not locate full feature row for the selected track.")
                        else:
                            candidates = df.drop(selected_row_full.name, errors="ignore").copy()
                            candidates = candidates.dropna(subset=similarity_features)

                            ref_vec = selected_row_full[similarity_features].astype(float).values
                            cand_mat = candidates[similarity_features].astype(float).values

                            dists = np.linalg.norm(cand_mat - ref_vec, axis=1)
                            candidates["similarity_distance"] = dists

                            # Deduplicate similar tracks and merge genres
                            candidates = candidates.reset_index().rename(columns={"index": "df_index"})

                            agg_dict_sim = {}
                            for col in candidates.columns:
                                if col == "genre":
                                    agg_dict_sim[col] = join_genres
                                elif col == "df_index":
                                    agg_dict_sim[col] = "first"
                                elif col == "similarity_distance":
                                    agg_dict_sim[col] = "min"
                                elif col == "popularity":
                                    agg_dict_sim[col] = "max"
                                else:
                                    if pd.api.types.is_numeric_dtype(candidates[col]):
                                        agg_dict_sim[col] = "mean"
                                    else:
                                        agg_dict_sim[col] = "first"

                            group_cols_sim = []
                            if track_col and track_col in candidates.columns:
                                group_cols_sim.append(track_col)
                            if artist_col and artist_col in candidates.columns:
                                group_cols_sim.append(artist_col)
                            if not group_cols_sim:
                                group_cols_sim = ["df_index"]

                            similar_tracks = candidates.groupby(group_cols_sim, as_index=False).agg(agg_dict_sim)
                            similar_tracks = similar_tracks.sort_values("similarity_distance", ascending=True).head(5)

                            sim_display_cols = []
                            for col in [
                                track_col, artist_col, "genre", "popularity",
                                "duration_m:s", "danceability", "energy", "tempo",
                                "scale", "valence", "similarity_distance"
                            ]:
                                if col in similar_tracks.columns and col not in sim_display_cols:
                                    sim_display_cols.append(col)

                            st.write("Top **5** most similar tracks in the dataset (deduplicated by track, with merged genres):")
                            st.dataframe(
                                similar_tracks[sim_display_cols]
                                if sim_display_cols
                                else similar_tracks,
                                use_container_width=True,
                            )
            else:
                st.warning("No track-name column found; cannot build similar-track finder.")

# =====================================================
# === GLOBAL ARTIST SEARCH (ACROSS WHOLE DATASET) =====
# =====================================================
st.subheader("🔎 Search Artist and View Top Songs (Full Dataset)")

artist_col_global = None
if "artist" in df.columns:
    artist_col_global = "artist"
elif "artist_name" in df.columns:
    artist_col_global = "artist_name"

if artist_col_global is None:
    st.warning("No artist column found in the dataset, so search is unavailable.")
else:
    search_query = st.text_input(
        "Type an artist name (full or partial):",
        "",
        help="Searches across all tracks in this dataset (case-insensitive).",
    )

    if search_query.strip():
        matches = df[df[artist_col_global].str.contains(
            search_query, case=False, na=False
        )].copy()

        if matches.empty:
            st.info("No artists found matching that search. Try a different spelling or a shorter fragment.")
        else:
            matches = matches.reset_index().rename(columns={"index": "df_index"})

            def join_genres_search(series):
                vals = [str(x) for x in series.dropna().unique()]
                return ", ".join(sorted(vals)) if vals else ""

            agg_dict_search = {}
            for col in matches.columns:
                if col == "genre":
                    agg_dict_search[col] = join_genres_search
                elif col == "df_index":
                    agg_dict_search[col] = "first"
                elif col == "popularity":
                    agg_dict_search[col] = "max"
                else:
                    if pd.api.types.is_numeric_dtype(matches[col]):
                        agg_dict_search[col] = "mean"
                    else:
                        agg_dict_search[col] = "first"

            track_col_global = None
            if "track_name" in matches.columns:
                track_col_global = "track_name"
            elif "track" in matches.columns:
                track_col_global = "track"

            group_cols_search = []
            if track_col_global and track_col_global in matches.columns:
                group_cols_search.append(track_col_global)
            if artist_col_global and artist_col_global in matches.columns:
                group_cols_search.append(artist_col_global)
            if not group_cols_search:
                group_cols_search = ["df_index"]

            artist_grouped_search = matches.groupby(group_cols_search, as_index=False).agg(agg_dict_search)

            if "popularity" in artist_grouped_search.columns:
                artist_grouped_search = artist_grouped_search.sort_values("popularity", ascending=False)

            total_tracks = artist_grouped_search.shape[0]

            if total_tracks >= 10:
                max_for_slider = min(50, total_tracks)
                num_tracks = st.slider(
                    "Number of top tracks to display for this artist search",
                    min_value=10,
                    max_value=max_for_slider,
                    value=10,
                )
            else:
                num_tracks = total_tracks
                st.caption(
                    f"Only {total_tracks} track(s) found for this search in the dataset."
                )

            top_tracks = artist_grouped_search.head(num_tracks)

            display_cols_search = []
            for col in [
                track_col_global,
                artist_col_global,
                "genre",
                "popularity",
                "duration_m:s",
                "danceability",
                "energy",
                "tempo",
                "scale",
                "valence",
            ]:
                if col in top_tracks.columns and col not in display_cols_search:
                    display_cols_search.append(col)

            st.write(
                f"Showing **{top_tracks.shape[0]}** track(s) matching artist search "
                f"**\"{search_query}\"** (deduplicated by track, with merged genres, sorted by popularity)."
            )

            st.dataframe(
                top_tracks[display_cols_search]
                if display_cols_search
                else top_tracks,
                use_container_width=True,
            )
    else:
        st.caption("Start typing an artist's name above to see their top songs in this dataset.")
        
# =====================================================
# === EXPLORATORY DATA ANALYSIS SECTION
# =====================================================
st.header("🧠 Exploratory Data Analysis")

# ----- 1) Correlation heatmap -----
st.subheader("📌 Correlation Heatmap of Numerical Features")

numeric_df = df.select_dtypes(include=["int64", "float64"])
if not numeric_df.empty:
    corr = numeric_df.corr()

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        labels=dict(color="Correlation"),
        title="Correlation Between Numerical Features",
        width=900,
        height=800
    )
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("No numeric columns available for correlation heatmap.")

# ----- 2) Feature vs popularity (bar chart, by genre) -----
if "popularity" in df.columns and "genre" in df.columns:
    st.subheader("📊 How Audio Features Relate to Popularity (Grouped by Genre)")

    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    exclude_cols = ["popularity", "duration_ms", "key", "mode"]
    numeric_features = [c for c in numeric_features if c not in exclude_cols]

    if numeric_features:
        chosen_feature = st.selectbox(
            "Select a feature for the X-axis (grouped):",
            options=numeric_features,
            key="bar_feature_select"
        )

        num_bins = st.slider(
            "Number of bins to group this feature:",
            min_value=3,
            max_value=12,
            value=6,
            key="bar_bins_slider"
        )

        df_copy = df.copy()
        df_copy = df_copy.dropna(subset=[chosen_feature, "popularity", "genre"])
        df_copy["feature_bin"] = pd.cut(df_copy[chosen_feature], bins=num_bins)

        grouped = (
            df_copy
            .groupby(["feature_bin", "genre"], observed=True)["popularity"]
            .mean()
            .reset_index()
        )

        # Convert Interval -> string for Plotly
        grouped["feature_bin"] = grouped["feature_bin"].astype(str)

        st.dataframe(grouped, use_container_width=True)

        fig_bar = px.bar(
            grouped,
            x="feature_bin",
            y="popularity",
            color="genre",
            barmode="group",
            title=f"Average Popularity grouped by {chosen_feature}, split by Genre",
            labels={
                "feature_bin": f"{chosen_feature} (grouped)",
                "popularity": "Average Popularity"
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("""
                    
**Notes:**

- Different ranges of the selected audio feature correspond to different popularity levels.
- Each color is a genre, so we can see genre-specific patterns.
- For example, some genres might favor higher danceability or energy more than others.
- This resembles how music platforms use audio features to understand what makes a track successful.
""")
    else:
        st.info("Not enough numeric audio features for feature-vs-popularity analysis.")
else:
    st.info("Need both 'popularity' and 'genre' columns for this analysis.")

# ----- 3) Scatter plot: feature vs popularity (old scatter you wanted) -----
if "popularity" in df.columns:
    st.subheader("📈 Scatter Plot: Feature vs Popularity (Colored by Genre)")

    numeric_features_scatter = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_features_scatter = [c for c in numeric_features_scatter if c != "popularity"]

    if numeric_features_scatter:
        scatter_feature = st.selectbox(
            "Select a feature for the X-axis (scatter):",
            options=numeric_features_scatter,
            key="scatter_feature_select"
        )

        df_scatter = df.copy()
        cols_needed = [scatter_feature, "popularity"]
        if "genre" in df.columns:
            cols_needed.append("genre")

        df_scatter = df_scatter.dropna(subset=cols_needed)

        if "genre" in df_scatter.columns:
            fig_scatter = px.scatter(
                df_scatter,
                x=scatter_feature,
                y="popularity",
                color="genre",
                opacity=0.6,
                title=f"{scatter_feature} vs Popularity (by Genre)"
            )
        else:
            fig_scatter = px.scatter(
                df_scatter,
                x=scatter_feature,
                y="popularity",
                opacity=0.6,
                title=f"{scatter_feature} vs Popularity"
            )

        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No numeric features available for scatter plot.")
else:
    st.info("No 'popularity' column for scatter plot.")

# =====================================================
# === SIMPLE MODEL COMPARISON SECTION (IMPROVED) ===
# =====================================================
st.header("🤖 Model Comparison: Predicting High vs Low Popularity")

if "popularity" in df.columns:
    # Work on a copy and drop rows with missing popularity
    df_model = df.copy().dropna(subset=["popularity"])

    # Binary label based on current popularity threshold (from sidebar)
    df_model["popularity_label"] = (df_model["popularity"] >= popularity_threshold).astype(int)

    model_features = [
        "danceability", "energy", "valence", "tempo", "loudness",
        "acousticness", "speechiness", "instrumentalness", "liveness"
    ]
    model_features = [c for c in model_features if c in df_model.columns]

    if model_features:
        # Use only rows that have all model features
        X = df_model[model_features].dropna()
        y = df_model.loc[X.index, "popularity_label"]

        # Optional: subsample for speed if dataset is very large
        max_rows = 15000  # adjust if needed
        if X.shape[0] > max_rows:
            sample_idx = X.sample(n=max_rows, random_state=42).index
            X = X.loc[sample_idx]
            y = y.loc[sample_idx]

        if X.shape[0] > 50:
            st.write(f"Using **{X.shape[0]}** tracks for modeling.")

            test_size = st.slider(
                "Test size (fraction of data used for testing):",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
            )

            with st.spinner("Training models..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                rf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
                knn = KNeighborsClassifier(n_neighbors=15)

                rf.fit(X_train, y_train)
                knn.fit(X_train, y_train)

                y_pred_rf = rf.predict(X_test)
                y_pred_knn = knn.predict(X_test)

                rf_acc = accuracy_score(y_test, y_pred_rf)
                knn_acc = accuracy_score(y_test, y_pred_knn)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Random Forest Accuracy", f"{rf_acc:.4f}")
            with col2:
                st.metric("KNN Accuracy", f"{knn_acc:.4f}")

            st.markdown(f"""
- Test size: **{test_size:.2f}**
- Threshold for "high popularity": **{popularity_threshold}**
- Features used: `{", ".join(model_features)}`
            """)

            st.markdown("""
You’ll notice the accuracy does **not** change dramatically as the test size changes.
That’s because the dataset is fairly large and the binary label is quite separable
with these audio features, so small changes in the split do not strongly affect
overall performance — especially when rounded.
""")

        else:
            st.info("Not enough tracks for a reliable train/test split.")
    else:
        st.warning("No suitable numeric features found for modeling.")
else:
    st.warning("No 'popularity' column found in the dataset.")