import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_products():
    return pd.read_csv("data/products.csv")


def load_interactions():
    return pd.read_csv("data/interactions.csv")


def load_users():
    return pd.read_csv("data/users.csv")


def load_image_features():
    return pd.read_csv("data/cnn_features.csv")


def calculate_visual_similarity(
    product_id,
    liked_products,
    image_features
):
    feature_columns = [
        column
        for column in image_features.columns
        if column.startswith("feature_")
    ]

    target = image_features[
        image_features["product_id"] == product_id
    ]

    if target.empty or not liked_products:
        return 0.0

    target_features = target[
        feature_columns
    ].values

    liked = image_features[
        image_features["product_id"].isin(liked_products)
    ]

    if liked.empty:
        return 0.0

    liked_features = liked[
        feature_columns
    ].values

    similarities = cosine_similarity(
        target_features,
        liked_features
    )[0]

    return float(similarities.mean())


def recommend_for_user(user_id):

    products = load_products()
    interactions = load_interactions()
    users = load_users()
    image_features = load_image_features()

    user = users[
        users["user_id"] == user_id
    ].iloc[0]

    user_interactions = interactions[
        interactions["user_id"] == user_id
    ]

    gender = user["gender"]
    style = user["preferred_style"]
    occasion = user["preferred_occasion"]
    budget = user["budget"]

    recommendations = products[
        (products["gender"] == gender) &
        (products["style"] == style) &
        (products["occasion"] == occasion) &
        (products["price"] <= budget)
    ].copy()

    liked_products = user_interactions[
        user_interactions["interaction"] == "like"
    ]["product_id"].tolist()

    disliked_products = user_interactions[
        user_interactions["interaction"] == "dislike"
    ]["product_id"].tolist()

    recommendations["rating_score"] = (
        recommendations["rating"]
    )

    recommendations["interaction_score"] = 0.0

    recommendations.loc[
        recommendations["product_id"].isin(liked_products),
        "interaction_score"
    ] += 1.0

    recommendations.loc[
        recommendations["product_id"].isin(disliked_products),
        "interaction_score"
    ] -= 1.0

    recommendations["visual_similarity"] = (
        recommendations["product_id"].apply(
            lambda product_id:
            calculate_visual_similarity(
                product_id,
                liked_products,
                image_features
            )
        )
    )

    recommendations["final_score"] = (
        recommendations["rating_score"] * 0.40
        +
        recommendations["interaction_score"] * 0.30
        +
        recommendations["visual_similarity"] * 0.30
    )

    recommendations = recommendations.sort_values(
        by="final_score",
        ascending=False
    )

    return recommendations


if __name__ == "__main__":

    results = recommend_for_user("U001")

    print(
        "Hybrid AI-Based Personalized Recommendations:"
    )

    print(
        results[
            [
                "product_id",
                "subcategory",
                "color",
                "price",
                "rating",
                "rating_score",
                "interaction_score",
                "visual_similarity",
                "final_score"
            ]
        ].to_string(index=False)
    )