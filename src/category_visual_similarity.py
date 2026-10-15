import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


INPUT_FILE = "data/real_cnn_features.csv"


def load_features():
    return pd.read_csv(INPUT_FILE)


def find_similar_images(
    image_index,
    top_n=5
):

    df = load_features()

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    target = df.iloc[image_index]

    target_category = target["category"]

    target_features = target[
        feature_columns
    ].values.reshape(1, -1)

    category_df = df[
        df["category"] == target_category
    ].copy()

    category_features = category_df[
        feature_columns
    ].values

    similarities = cosine_similarity(
        target_features,
        category_features
    )[0]

    category_df["similarity"] = similarities

    category_df = category_df[
        category_df.index != image_index
    ]

    category_df = category_df.sort_values(
        by="similarity",
        ascending=False
    )

    return target, category_df.head(top_n)


if __name__ == "__main__":

    image_index = 0

    target, results = find_similar_images(
        image_index,
        top_n=5
    )

    print("Target image:")
    print(
        target[
            [
                "product_name",
                "category",
                "image_path"
            ]
        ].to_string()
    )

    print(
        "\nCategory-filtered visual recommendations:"
    )

    print(
        results[
            [
                "product_name",
                "category",
                "similarity",
                "image_path"
            ]
        ].to_string(index=False)
    )