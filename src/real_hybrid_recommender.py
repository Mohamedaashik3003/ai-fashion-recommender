import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


FEATURE_FILE = "data/real_cnn_features.csv"
USER_FILE = "data/users.csv"
INTERACTION_FILE = "data/interactions.csv"


def load_data():

    features = pd.read_csv(FEATURE_FILE)
    users = pd.read_csv(USER_FILE)
    interactions = pd.read_csv(INTERACTION_FILE)

    return features, users, interactions


def recommend_for_user(user_id, top_n=10):

    products, users, interactions = load_data()

    user = users[
        users["user_id"] == user_id
    ].iloc[0]

    user_interactions = interactions[
        interactions["user_id"] == user_id
    ]

    liked_products = user_interactions[
        user_interactions["interaction"] == "like"
    ]["product_id"].tolist()

    disliked_products = user_interactions[
        user_interactions["interaction"] == "dislike"
    ]["product_id"].tolist()

    feature_columns = [
        column
        for column in products.columns
        if column.startswith("feature_")
    ]

    recommendations = products.copy()

    recommendations["preference_score"] = 0.0
    recommendations["interaction_score"] = 0.0
    recommendations["visual_score"] = 0.0
    recommendations["budget_score"] = 0.0

    if "preferred_color" in user:

        color_matches = (
            recommendations["product_name"]
            .str.lower()
            .str.contains(
                str(user["preferred_color"]).lower(),
                na=False
            )
        )

        recommendations.loc[
            color_matches,
            "preference_score"
        ] += 1.0

    if liked_products:

        liked_data = products[
            products["product_name"].isin(
                liked_products
            )
        ]

        if not liked_data.empty:

            liked_features = liked_data[
                feature_columns
            ].values

            all_features = recommendations[
                feature_columns
            ].values

            similarity = cosine_similarity(
                all_features,
                liked_features
            ).mean(axis=1)

            recommendations[
                "visual_score"
            ] = similarity

    recommendations.loc[
        recommendations["product_name"].isin(
            liked_products
        ),
        "interaction_score"
    ] += 1.0

    recommendations.loc[
        recommendations["product_name"].isin(
            disliked_products
        ),
        "interaction_score"
    ] -= 1.0

    recommendations["budget_score"] = (
        recommendations["category"].map(
            lambda x: 1.0
        )
    )

    recommendations["final_score"] = (
        recommendations["visual_score"] * 0.40
        +
        recommendations["preference_score"] * 0.20
        +
        recommendations["interaction_score"] * 0.25
        +
        recommendations["budget_score"] * 0.15
    )

    recommendations = recommendations.sort_values(
        by="final_score",
        ascending=False
    )

    return recommendations.head(top_n)


if __name__ == "__main__":

    results = recommend_for_user(
        "U001",
        top_n=10
    )

    print(
        "Personalized Hybrid Fashion Recommendations:"
    )

    print(
        results[
            [
                "product_name",
                "category",
                "visual_score",
                "preference_score",
                "interaction_score",
                "final_score"
            ]
        ].to_string(index=False)
    )
