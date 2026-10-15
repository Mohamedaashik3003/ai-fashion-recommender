import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_features():
    return pd.read_csv("data/cnn_features.csv")


def find_similar_products(product_id, top_n=5):

    df = load_features()

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    product = df[
        df["product_id"] == product_id
    ]

    if product.empty:
        raise ValueError(
            f"Product {product_id} not found."
        )

    product_features = product[
        feature_columns
    ].values

    all_features = df[
        feature_columns
    ].values

    similarities = cosine_similarity(
        product_features,
        all_features
    )[0]

    df["similarity"] = similarities

    df = df[
        df["product_id"] != product_id
    ]

    df = df.sort_values(
        by="similarity",
        ascending=False
    )

    return df.head(top_n)


if __name__ == "__main__":

    product_id = "P001"

    results = find_similar_products(
        product_id,
        top_n=5
    )

    print(
        f"Products visually similar to {product_id}:"
    )

    print(
        results[
            [
                "product_id",
                "similarity"
            ]
        ].to_string(index=False)
    )