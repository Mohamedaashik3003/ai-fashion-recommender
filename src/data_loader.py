import pandas as pd


def load_products():
    return pd.read_csv("data/products.csv")


def load_interactions():
    return pd.read_csv("data/interactions.csv")


if __name__ == "__main__":
    products = load_products()
    interactions = load_interactions()

    print("Products:", len(products))
    print("Interactions:", len(interactions))

    print("\nUser Interactions:")
    print(interactions)