
import os
import sys
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from real_outfit_recommender import recommend_outfits

IMAGE_ROOT = os.path.join(
    PROJECT_ROOT, "data", "real_images", "img"
)

METADATA_FILE = os.path.join(
    PROJECT_ROOT, "data", "fashion_subset.csv"
)

TRYON_URL = "https://yisol-idm-vton.hf.space"

st.set_page_config(
    page_title="Women's AI Fashion Studio",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Women's AI Fashion Studio")

st.write(
    "Discover personalized outfit combinations and explore "
    "AI-powered virtual try-on."
)

st.info(
    "Outfit scores are ranking scores, not accuracy percentages. "
    "Recommendations depend on the available dataset and profile "
    "interactions. The virtual try-on demo is externally hosted."
)


@st.cache_data
def load_metadata():
    if not os.path.isfile(METADATA_FILE):
        return {}

    df = pd.read_csv(METADATA_FILE)

    if not {"product_name", "image_path"}.issubset(df.columns):
        return {}

    df = df.drop_duplicates(
        subset=["product_name"],
        keep="first"
    )

    return df.set_index("product_name")["image_path"].to_dict()


PRODUCT_IMAGES = load_metadata()


def find_image(product_name):
    relative_path = PRODUCT_IMAGES.get(str(product_name))

    if isinstance(relative_path, str) and relative_path.strip():
        path = relative_path.replace("\\", os.sep).replace("/", os.sep)

        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)

        path = os.path.normpath(path)

        if os.path.isfile(path):
            return path

    folder = os.path.join(IMAGE_ROOT, str(product_name))

    if os.path.isdir(folder):
        extensions = (".jpg", ".jpeg", ".png", ".webp")

        files = sorted(
            name for name in os.listdir(folder)
            if name.lower().endswith(extensions)
        )

        if files:
            return os.path.join(folder, files[0])

    return None


def show_product(product_name, unique_id="product"):
    path = find_image(product_name)

    if not path:
        st.warning(f"Image not found for {product_name}")
        st.caption(str(product_name).replace("_", " "))
        return

    st.image(
        path,
        caption=str(product_name).replace("_", " "),
        use_container_width=True
    )

    extension = os.path.splitext(path)[1].lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }

    try:
        with open(path, "rb") as file:
            image_data = file.read()

        st.download_button(
            "Download garment image",
            data=image_data,
            file_name=os.path.basename(path),
            mime=mime_types.get(
                extension,
                "application/octet-stream"
            ),
            key=f"download_{unique_id}_{product_name}"
        )
    except OSError:
        st.warning("Unable to read this image for downloading.")


recommend_tab, tryon_tab = st.tabs(
    ["👗 Outfit Recommendations", "✨ Virtual Try-On"]
)


with recommend_tab:
    st.subheader("Personalized Outfit Recommendations")

    st.write(
        "Choose a profile to generate combinations of tops and "
        "bottoms ranked using visual similarity and interaction data."
    )

    user_id = st.selectbox(
        "Choose profile",
        ["U001", "U002", "U003", "U004", "U005"]
    )

    top_n = st.slider(
        "Number of outfits",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("Generate Outfits", type="primary"):
        try:
            with st.spinner("Finding outfit combinations..."):
                results = recommend_outfits(
                    user_id,
                    top_n=top_n,
                    style="Women's-Style Clothing"
                )

            st.session_state["women_outfits"] = results
            st.session_state["women_outfits_user"] = user_id

        except Exception as error:
            st.error(f"Could not generate recommendations: {error}")

    results = st.session_state.get("women_outfits")
    results_user = st.session_state.get("women_outfits_user")

    if results is not None and results_user == user_id:
        if results.empty:
            st.warning(
                "No matching outfits found for this profile. "
                "Try another profile."
            )
        else:
            results = results.drop_duplicates(
                subset=["top", "bottom"]
            ).reset_index(drop=True)

            st.success(
                f"Found {len(results)} outfit combinations."
            )

            for index, (_, row) in enumerate(
                results.iterrows(),
                start=1
            ):
                with st.container(border=True):
                    st.markdown(f"### Outfit {index}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Top")
                        show_product(
                            row["top"],
                            unique_id=f"outfit_{index}_top"
                        )

                    with col2:
                        st.subheader("Bottom")
                        show_product(
                            row["bottom"],
                            unique_id=f"outfit_{index}_bottom"
                        )

                    compatibility = row.get(
                        "compatibility_score", 0.0
                    )
                    preference = row.get(
                        "preference_score", 0.0
                    )
                    ranking = row.get("final_score", 0.0)

                    st.caption(
                        f"Compatibility: {compatibility:.3f} | "
                        f"Preference: {preference:.3f} | "
                        f"Ranking: {ranking:.3f}"
                    )


with tryon_tab:
    st.subheader("AI Virtual Try-On")

    st.write(
        "Open the hosted model to upload a person photo and a "
        "garment image, then generate a virtual try-on result."
    )

    st.warning(
        "This model is hosted externally and may sometimes be "
        "unavailable. Results for sarees and other traditional "
        "clothing are not guaranteed."
    )

    components.iframe(
        TRYON_URL,
        height=800,
        scrolling=True
    )

    st.markdown(
        f"[Open virtual try-on in a new tab]({TRYON_URL})"
    )


st.divider()

st.caption(
    "Women's AI Fashion Studio | DeepFashion image features | "
    "Personalized outfit recommendations | Open-source virtual try-on"
)
