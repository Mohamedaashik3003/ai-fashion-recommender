import pandas as pd


def load_products():
    return pd.read_csv("data/products.csv")


def calculate_compatibility(top, bottom):
    score = 0

    if top["gender"] == bottom["gender"]:
        score += 30

    if top["occasion"] == bottom["occasion"]:
        score += 30

    if top["style"] == bottom["style"]:
        score += 25

    if top["category"] != bottom["category"]:
        score += 15

    return score


def generate_outfits():
    products = load_products()

    tops = products[
        products["category"] == "Top"
    ]

    bottoms = products[
        products["category"] == "Bottom"
    ]

    outfits = []

    for _, top in tops.iterrows():
        for _, bottom in bottoms.iterrows():

            score = calculate_compatibility(
                top,
                bottom
            )

            outfits.append({
                "top_id": top["product_id"],
                "top_name": top["subcategory"],
                "top_color": top["color"],
                "bottom_id": bottom["product_id"],
                "bottom_name": bottom["subcategory"],
                "bottom_color": bottom["color"],
                "total_price": top["price"] + bottom["price"],
                "compatibility_score": score
            })

    outfits = pd.DataFrame(outfits)

    outfits = outfits.sort_values(
        by="compatibility_score",
        ascending=False
    )

    return outfits


if __name__ == "__main__":
    outfits = generate_outfits()

    print("AI-Based Outfit Recommendations:")
    print(
        outfits[
            [
                "top_id",
                "top_name",
                "top_color",
                "bottom_id",
                "bottom_name",
                "bottom_color",
                "total_price",
                "compatibility_score"
            ]
        ].head(10)
    )