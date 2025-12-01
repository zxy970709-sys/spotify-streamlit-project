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
# === EXAMPLE TRACKS SECTION (Dataset 1 + 2 + 3)
# =====================================================
st.header("🎧 Example Tracks from Filtered Data")

if filtered_df.empty:
    st.warning("No tracks match the current filters. Try changing genre or popularity threshold.")
else:
    # How many examples to show
    default_n = 20 if filtered_df.shape[0] >= 20 else filtered_df.shape[0]
    n_rows_to_show = st.slider(
        "Number of example tracks to display:",
        min_value=1,
        max_value=min(200, filtered_df.shape[0]),
        value=default_n
    )

    example_cols = [
        "track_name", "artist_name", "genre", "popularity",
        "duration_m:s", "danceability", "energy", "tempo", "scale", "valence"
    ]
    example_cols = [c for c in example_cols if c in filtered_df.columns]

    # Preserve original index so we can map back to df for similarity
    examples_df = (
        filtered_df
        .sort_values("popularity", ascending=False)
        .loc[:, example_cols]
        .head(n_rows_to_show)
    )

    st.subheader("Dataset 1: Example Tracks")
    st.dataframe(examples_df, use_container_width=True)

    st.markdown("---")

    # ---------- Dataset 2: Artist explorer ----------
    st.subheader("Dataset 2: Explore Songs by an Artist")

    if "artist_name" in examples_df.columns:
        artists_in_examples = examples_df["artist_name"].dropna().unique().tolist()
        if artists_in_examples:
            selected_artist = st.selectbox(
                "Select an artist from the example table:",
                options=artists_in_examples
            )
            artist_tracks = df[df["artist_name"] == selected_artist].copy()
            if "popularity" in artist_tracks.columns:
                artist_tracks = artist_tracks.sort_values("popularity", ascending=False)

            artist_cols = [
                "track_name", "genre", "popularity",
                "duration_m:s", "danceability", "energy", "tempo", "scale", "valence"
            ]
            artist_cols = [c for c in artist_cols if c in artist_tracks.columns]

            st.write(f"Top {min(5, artist_tracks.shape[0])} tracks by **{selected_artist}** in this dataset:")
            if not artist_tracks.empty:
                st.dataframe(
                    artist_tracks[artist_cols].head(5),
                    use_container_width=True
                )
            else:
                st.info("No tracks found for this artist in the dataset.")
        else:
            st.info("No artists available in the current examples.")
    else:
        st.info("Artist information is not available in this dataset.")

    st.markdown("---")

    # ---------- Dataset 3: Similar track finder ----------
    st.subheader("Dataset 3: Find Similar Tracks to One Example")

    if "track_name" in examples_df.columns:
        # Labels like "Song — Artist" for dropdown
        if "artist_name" in examples_df.columns:
            track_labels = examples_df.apply(
                lambda r: f"{r['track_name']} — {r['artist_name']}", axis=1
            )
        else:
            track_labels = examples_df["track_name"].astype(str)

        # Map label -> original index (from df/filtered_df)
        label_to_index = {
            label: idx for label, idx in zip(track_labels, examples_df.index)
        }

        selected_label = st.selectbox(
            "Select a track from the example table:",
            options=list(label_to_index.keys())
        )

        if st.button("Show similar tracks"):
            row_idx = label_to_index[selected_label]

            # Prefer using the full df row for more features
            if row_idx in df.index:
                selected_row_full = df.loc[row_idx]
            else:
                selected_row_full = examples_df.loc[row_idx]

            feature_candidates = [
                "danceability", "energy", "valence", "tempo",
                "loudness", "acousticness", "speechiness",
                "instrumentalness", "liveness"
            ]
            feature_cols = [
                c for c in feature_candidates
                if c in df.columns and not pd.isna(selected_row_full.get(c, np.nan))
            ]

            if feature_cols:
                # Start from full dataset
                candidates = df.dropna(subset=feature_cols).copy()

                # Same genre if possible
                if "genre" in df.columns and not pd.isna(selected_row_full.get("genre", np.nan)):
                    candidates = candidates[candidates["genre"] == selected_row_full["genre"]]

                # Drop the selected track itself if it's in candidates (by index)
                if row_idx in candidates.index:
                    candidates = candidates.drop(index=row_idx)

                if not candidates.empty:
                    # Reference vector from selected track
                    ref_vec = selected_row_full[feature_cols].astype(float).values

                    def compute_distance(r):
                        return np.linalg.norm(r[feature_cols].astype(float).values - ref_vec)

                    candidates["similarity_distance"] = candidates.apply(
                        compute_distance, axis=1
                    )

                    similar_tracks = candidates.sort_values("similarity_distance").head(5)

                    similar_cols = [
                        "track_name", "artist_name", "genre", "popularity",
                        "duration_m:s", "danceability", "energy", "tempo",
                        "scale", "valence", "similarity_distance"
                    ]
                    similar_cols = [
                        c for c in similar_cols if c in similar_tracks.columns
                    ]

                    st.dataframe(
                        similar_tracks[similar_cols],
                        use_container_width=True
                    )
                else:
                    st.warning("No candidate tracks available for similarity search.")
            else:
                st.warning("Not enough numeric audio features available to compute similarity.")
    else:
        st.info("Track names are not available for similarity search.")

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