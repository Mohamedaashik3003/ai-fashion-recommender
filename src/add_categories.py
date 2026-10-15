
import os
import re
import pandas as pd

INPUT_FILE = "data/fashion_metadata.csv"
OUTPUT_FILE = "data/fashion_metadata_categorized.csv"


def normalize_name(product_name):
    name = str(product_name).lower()
    name = name.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", name).strip()


def contains_word(name, words):
    return any(
        re.search(r"\b" + re.escape(word) + r"\b", name)
        for word in words
    )


def detect_category(product_name):
    name = normalize_name(product_name)

    categories = {
        "Saree": ["saree", "sari"],
        "Lehenga": ["lehenga", "lehnga"],
        "Kurti": ["kurti"],
        "Salwar Suit": ["salwar", "kameez", "churidar", "anarkali"],
        "Ethnic Set": ["ethnic set", "suit set"],
        "Dupatta": ["dupatta"],
        "Bottom": [
            "jeans", "trousers", "pants", "shorts", "skirt",
            "leggings", "joggers", "jeggings", "culottes",
            "cutoffs", "chinos", "sweatpants", "capri", "palazzo"
        ],
        "Dress": ["dress", "romper", "jumpsuit", "gown"],
        "Top": [
            "tee", "t shirt", "shirt", "blouse", "top", "tank",
            "hoodie", "sweater", "sweatshirt", "cardigan", "jacket",
            "coat", "kimono", "poncho", "blazer", "tunic", "vest",
            "bodysuit", "kaftan"
        ],
        "Footwear": [
            "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
            "heel", "heels", "sandal", "sandals", "loafer", "flats"
        ],
        "Accessory": [
            "bag", "purse", "backpack", "scarf", "belt", "wallet",
            "necklace", "bracelet", "earrings", "earring", "watch",
            "sunglasses"
        ]
    }

    for category in [
        "Saree", "Lehenga", "Kurti", "Salwar Suit",
        "Ethnic Set", "Dupatta", "Bottom", "Dress",
        "Top", "Footwear", "Accessory"
    ]:
        if contains_word(name, categories[category]):
            return category

    return "Other"


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Cannot find {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "product_name" not in df.columns:
        raise ValueError("Missing product_name column in metadata.")

    df["category"] = df["product_name"].apply(detect_category)
    df.to_csv(OUTPUT_FILE, index=False)

    unique_products = df.drop_duplicates("product_name")

    print(f"Saved categorized metadata to: {OUTPUT_FILE}")
    print("\nImage records by category:")
    print(df["category"].value_counts().to_string())

    print("\nUnique products by category:")
    print(
        unique_products["category"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()
