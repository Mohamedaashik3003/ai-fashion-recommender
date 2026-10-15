
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

FEATURE_FILE = "data/real_cnn_features.csv"
INTERACTION_FILE = "data/real_interactions.csv"

INTERACTION_WEIGHTS = {
    "like": 5.0,
    "wishlist": 4.0,
    "cart": 3.0,
    "view": 1.0,
    "dislike": -5.0
}


def load_data():
    features = pd.read_csv(FEATURE_FILE)
    interactions = pd.read_csv(INTERACTION_FILE)

    required_feature_columns = {"product_name", "category"}
    required_interaction_columns = {
        "user_id", "product_name", "interaction"
    }

    if not required_feature_columns.issubset(features.columns):
        raise ValueError(
            "real_cnn_features.csv must contain product_name and category."
        )

    if not required_interaction_columns.issubset(interactions.columns):
        raise ValueError(
            "real_interactions.csv must contain user_id, product_name, and interaction."
        )

    features = features.dropna(subset=["product_name", "category"]).copy()
    interactions = interactions.dropna(
        subset=["user_id", "product_name", "interaction"]
    ).copy()

    features["product_name"] = features["product_name"].astype(str)
    features["category"] = features["category"].astype(str)
    interactions["product_name"] = interactions["product_name"].astype(str)
    interactions["interaction"] = (
        interactions["interaction"].astype(str).str.lower().str.strip()
    )

    return features, interactions


def build_product_data(features):
    feature_columns = [
        column for column in features.columns
        if column.startswith("feature_")
    ]

    if not feature_columns:
        raise ValueError(
            "No feature_ columns found in real_cnn_features.csv."
        )

    product_features = features.groupby("product_name")[
        feature_columns
    ].mean()

    product_categories = features.groupby("product_name")[
        "category"
    ].first()

    return product_features, product_categories


def classify_style(product_name, category=""):
    name = str(product_name).lower().replace("_", " ").replace("-", " ")
    category = str(category).lower()

    women_keywords = [
        "women", "woman", "womens", "ladies", "female",
        "bralette", "bikini", "maternity", "bodycon", "blouse",
        "peplum", "pencil skirt", "mini skirt", "maxi dress",
        "midi dress", "sundress", "cocktail dress", "wedding dress"
    ]

    men_keywords = [
        "mens", "men's", "menswear", "male", "swim trunks"
    ]

    if any(word in name for word in women_keywords):
        return "Women"

    if any(word in name for word in men_keywords):
        return "Men"

    if category == "dress":
        return "Women"

    return "Unisex"


def calculate_user_preference(product_features, user_interactions):
    preference_scores = pd.Series(
        0.0, index=product_features.index
    )

    valid = user_interactions[
        user_interactions["product_name"].isin(product_features.index)
    ]

    if valid.empty:
        return preference_scores

    interaction_products = valid["product_name"].unique()
    interaction_vectors = product_features.loc[interaction_products].values

    similarity_matrix = cosine_similarity(
        product_features.values, interaction_vectors
    )

    weights = []
    for product_name in interaction_products:
        rows = valid[valid["product_name"] == product_name]
        weight = sum(
            INTERACTION_WEIGHTS.get(value, 0.0)
            for value in rows["interaction"]
        )
        weights.append(weight)

    scores = similarity_matrix @ weights

    return pd.Series(scores, index=product_features.index)


def normalize_scores(scores):
    if scores.empty:
        return scores

    minimum = scores.min()
    maximum = scores.max()

    if maximum > minimum:
        return (scores - minimum) / (maximum - minimum)

    return pd.Series(0.0, index=scores.index)


def recommend_outfits(user_id, top_n=10, style="All Styles"):
    features, interactions = load_data()

    if top_n < 1:
        return pd.DataFrame()

    user_interactions = interactions[
        interactions["user_id"].astype(str) == str(user_id)
    ].copy()

    product_features, product_categories = build_product_data(features)

    product_styles = pd.Series({
        product: classify_style(
            product, product_categories.get(product, "")
        )
        for product in product_features.index
    })

    if style == "Men's-Style Clothing":
        allowed_styles = ["Men", "Unisex"]
    elif style == "Women's-Style Clothing":
        allowed_styles = ["Women", "Unisex"]
    else:
        allowed_styles = ["Men", "Women", "Unisex"]

    allowed_products = product_styles[
        product_styles.isin(allowed_styles)
    ].index

    product_features = product_features.loc[
        product_features.index.intersection(allowed_products)
    ]
    product_categories = product_categories.loc[product_features.index]

    if product_features.empty:
        return pd.DataFrame()

    valid_interactions = user_interactions[
        user_interactions["product_name"].isin(product_features.index)
    ]

    disliked_products = set(
        valid_interactions.loc[
            valid_interactions["interaction"] == "dislike",
            "product_name"
        ]
    )

    preference_scores = normalize_scores(
        calculate_user_preference(product_features, valid_interactions)
    )

    top_products = product_categories[
        product_categories.str.lower().str.strip() == "top"
    ].index

    bottom_products = product_categories[
        product_categories.str.lower().str.strip() == "bottom"
    ].index

    tops = [
        product for product in top_products
        if product not in disliked_products
    ]
    bottoms = [
        product for product in bottom_products
        if product not in disliked_products
    ]

    if not tops or not bottoms:
        return pd.DataFrame(
            columns=[
                "top", "bottom", "compatibility_score",
                "preference_score", "final_score"
            ]
        )

    compatibility = (
        cosine_similarity(
            product_features.loc[tops].values,
            product_features.loc[bottoms].values
        ) + 1
    ) / 2

    top_preferences = preference_scores.loc[tops].values
    bottom_preferences = preference_scores.loc[bottoms].values

    preference_matrix = (
        top_preferences[:, None] + bottom_preferences[None, :]
    ) / 2

    final_matrix = compatibility * 0.60 + preference_matrix * 0.40

    candidates = []

    for i, top in enumerate(tops):
        for j, bottom in enumerate(bottoms):
            candidates.append({
                "top": top,
                "bottom": bottom,
                "compatibility_score": float(compatibility[i, j]),
                "preference_score": float(preference_matrix[i, j]),
                "final_score": float(final_matrix[i, j])
            })

    candidates.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    # Diversify results so clothing items are not repeatedly suggested.
    selected = []
    used_tops = set()
    used_bottoms = set()

    for candidate in candidates:
        if (
            candidate["top"] in used_tops
            or candidate["bottom"] in used_bottoms
        ):
            continue

        selected.append(candidate)
        used_tops.add(candidate["top"])
        used_bottoms.add(candidate["bottom"])

        if len(selected) >= top_n:
            break

    # If strict diversification leaves too few outfits, fill remaining slots
    # with the best unique pairs, while still excluding disliked products.
    if len(selected) < top_n:
        selected_pairs = {
            (item["top"], item["bottom"]) for item in selected
        }

        for candidate in candidates:
            pair = (candidate["top"], candidate["bottom"])

            if pair in selected_pairs:
                continue

            selected.append(candidate)
            selected_pairs.add(pair)

            if len(selected) >= top_n:
                break

    return pd.DataFrame(selected).reset_index(drop=True)


if __name__ == "__main__":
    for selected_style in [
        "All Styles",
        "Men's-Style Clothing",
        "Women's-Style Clothing"
    ]:
        print(f"\nStyle: {selected_style}")

        results = recommend_outfits(
            "U001", top_n=5, style=selected_style
        )

        if results.empty:
            print("No matching outfits found.")
        else:
            print(
                results[
                    [
                        "top", "bottom", "compatibility_score",
                        "preference_score", "final_score"
                    ]
                ].to_string(index=False)
            )

