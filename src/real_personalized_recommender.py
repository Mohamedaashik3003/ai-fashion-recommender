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

    features = pd.read_csv(
        FEATURE_FILE
    )

    interactions = pd.read_csv(
        INTERACTION_FILE
    )

    return features, interactions


def build_product_features(features):

    feature_columns = [
        column
        for column in features.columns
        if column.startswith("feature_")
    ]

    product_features = features.groupby(
        "product_name"
    )[feature_columns].mean()

    product_categories = features.groupby(
        "product_name"
    )["category"].first()

    return (
        product_features,
        product_categories
    )


def calculate_personalized_scores(
    product_features,
    user_interactions
):

    all_products = product_features.index

    interaction_scores = pd.Series(
        0.0,
        index=all_products
    )

    positive_scores = pd.Series(
        0.0,
        index=all_products
    )

    negative_scores = pd.Series(
        0.0,
        index=all_products
    )

    valid_interactions = 0

    for _, row in user_interactions.iterrows():

        product_name = row[
            "product_name"
        ]

        interaction_type = row[
            "interaction"
        ]

        if product_name not in product_features.index:
            continue

        weight = INTERACTION_WEIGHTS.get(
            interaction_type,
            0.0
        )

        if weight == 0:
            continue

        valid_interactions += 1

        target_vector = product_features.loc[
            product_name
        ].values.reshape(
            1,
            -1
        )

        similarities = cosine_similarity(
            target_vector,
            product_features.values
        )[0]

        similarity_series = pd.Series(
            similarities,
            index=all_products
        )

        weighted_similarity = (
            similarity_series * weight
        )

        interaction_scores = (
            interaction_scores
            +
            weighted_similarity
        )

        if weight > 0:

            positive_scores = (
                positive_scores
                +
                weighted_similarity
            )

        else:

            negative_scores = (
                negative_scores
                +
                weighted_similarity
            )

    return (
        interaction_scores,
        positive_scores,
        negative_scores,
        valid_interactions
    )


def recommend_for_user(
    user_id,
    top_n=10
):

    features, interactions = load_data()

    user_interactions = interactions[
        interactions["user_id"] == user_id
    ].copy()

    if user_interactions.empty:

        print(
            f"No interactions found for {user_id}."
        )

        return pd.DataFrame()

    (
        product_features,
        product_categories
    ) = build_product_features(
        features
    )

    (
        interaction_scores,
        positive_scores,
        negative_scores,
        valid_interactions
    ) = calculate_personalized_scores(
        product_features,
        user_interactions
    )

    recommendations = pd.DataFrame(
        index=product_features.index
    )

    recommendations["category"] = (
        product_categories
    )

    recommendations["interaction_score"] = (
        interaction_scores
    )

    recommendations["positive_score"] = (
        positive_scores
    )

    recommendations["negative_score"] = (
        negative_scores
    )

    recommendations["visual_score"] = (
        positive_scores
    )

    interacted_products = set(
        user_interactions[
            "product_name"
        ].tolist()
    )

    recommendations = recommendations[
        ~recommendations.index.isin(
            interacted_products
        )
    ]

    if recommendations.empty:

        return recommendations.reset_index()

    visual_min = recommendations[
        "visual_score"
    ].min()

    visual_max = recommendations[
        "visual_score"
    ].max()

    if visual_max > visual_min:

        recommendations[
            "visual_score"
        ] = (
            recommendations[
                "visual_score"
            ] - visual_min
        ) / (
            visual_max - visual_min
        )

    else:

        recommendations[
            "visual_score"
        ] = 0.0

    interaction_min = recommendations[
        "interaction_score"
    ].min()

    interaction_max = recommendations[
        "interaction_score"
    ].max()

    if interaction_max > interaction_min:

        recommendations[
            "interaction_score"
        ] = (
            recommendations[
                "interaction_score"
            ] - interaction_min
        ) / (
            interaction_max - interaction_min
        )

    else:

        recommendations[
            "interaction_score"
        ] = 0.0

    recommendations["final_score"] = (
        recommendations[
            "visual_score"
        ] * 0.60
        +
        recommendations[
            "interaction_score"
        ] * 0.40
    )

    recommendations = recommendations.sort_values(
        by="final_score",
        ascending=False
    )

    recommendations = recommendations.head(
        top_n
    )

    recommendations = recommendations.reset_index()

    print(
        f"\nValid interactions used: "
        f"{valid_interactions}"
    )

    return recommendations


if __name__ == "__main__":

    user_id = "U001"

    results = recommend_for_user(
        user_id,
        top_n=10
    )

    print(
        "\nPersonalized AI Fashion Recommendations"
    )

    print(
        f"\nUser: {user_id}\n"
    )

    if results.empty:

        print(
            "No recommendations available."
        )

    else:

        print(
            results[
                [
                    "product_name",
                    "category",
                    "visual_score",
                    "interaction_score",
                    "final_score"
                ]
            ].to_string(
                index=False
            )
        )