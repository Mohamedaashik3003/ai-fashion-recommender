import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


FEATURE_FILE = "data/real_cnn_features.csv"


def load_features():

    return pd.read_csv(
        FEATURE_FILE
    )


def find_similar_products(
    product_name,
    top_n=10
):

    df = load_features()

    feature_columns = [
        column
        for column in df.columns
        if column.startswith("feature_")
    ]

    target = df[
        df["product_name"] == product_name
    ]

    if target.empty:

        raise ValueError(
            f"Product '{product_name}' not found."
        )

    target_features = target[
        feature_columns
    ].mean(
        axis=0
    ).values.reshape(
        1,
        -1
    )

    product_features = df.groupby(
        "product_name"
    )[feature_columns].mean()

    similarities = cosine_similarity(
        target_features,
        product_features.values
    )[0]

    results = pd.DataFrame({
        "product_name":
            product_features.index,
        "similarity":
            similarities
    })

    results = results[
        results["product_name"] != product_name
    ]

    category = target[
        "category"
    ].iloc[0]

    product_categories = df.groupby(
        "product_name"
    )["category"].first()

    results["category"] = results[
        "product_name"
    ].map(
        product_categories
    )

    results = results[
        results["category"] == category
    ]

    results = results.sort_values(
        by="similarity",
        ascending=False
    )

    return results.head(
        top_n
    )


if __name__ == "__main__":

    product_name = (
        "Floral_Print_Lace-Paneled_Dress"
    )

    results = find_similar_products(
        product_name,
        top_n=10
    )

    print(
        f"\nProducts visually similar to:"
    )

    print(
        product_name
    )

    print()

    print(
        results.to_string(
            index=False
        )
    )