import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def load_products():
    return pd.read_csv("data/products.csv")


def load_users():
    return pd.read_csv("data/users.csv")


def load_interactions():
    return pd.read_csv("data/interactions.csv")


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


def calculate_interaction_score(
    top_id,
    bottom_id,
    interactions
):
    score = 0

    product_ids = [top_id, bottom_id]

    user_interactions = interactions[
        interactions["product_id"].isin(product_ids)
    ]

    for _, interaction in user_interactions.iterrows():

        if interaction["interaction"] == "like":
            score += 15

        elif interaction["interaction"] == "cart":
            score += 10

        elif interaction["interaction"] == "wishlist":
            score += 8

        elif interaction["interaction"] == "dislike":
            score -= 15

    return score


def recommend_outfits(user_id):

    products = load_products()
    users = load_users()
    interactions = load_interactions()

    user = users[
        users["user_id"] == user_id
    ].iloc[0]

    user_interactions = interactions[
        interactions["user_id"] == user_id
    ]

    tops = products[
        (products["category"] == "Top") &
        (products["gender"] == user["gender"]) &
        (products["style"] == user["preferred_style"]) &
        (products["occasion"] == user["preferred_occasion"])
    ]

    bottoms = products[
        (products["category"] == "Bottom") &
        (products["gender"] == user["gender"])
    ]

    outfits = []

    for _, top in tops.iterrows():

        for _, bottom in bottoms.iterrows():

            total_price = top["price"] + bottom["price"]

            if total_price <= user["budget"] * 2:

                compatibility = calculate_compatibility(
                    top,
                    bottom
                )

                preference_score = 0

                if top["color"] == user["preferred_color"]:
                    preference_score += 10

                if bottom["color"] == user["preferred_color"]:
                    preference_score += 10

                interaction_score = calculate_interaction_score(
                    top["product_id"],
                    bottom["product_id"],
                    user_interactions
                )

                final_score = (
                    compatibility +
                    preference_score +
                    interaction_score
                )

                outfits.append({
                    "top_id": top["product_id"],
                    "top": top["subcategory"],
                    "top_color": top["color"],
                    "bottom_id": bottom["product_id"],
                    "bottom": bottom["subcategory"],
                    "bottom_color": bottom["color"],
                    "top_image": top["image_path"],
                    "bottom_image": bottom["image_path"],
                    "total_price": total_price,
                    "final_score": final_score
                })

    outfits = pd.DataFrame(outfits)

    if outfits.empty:
        return outfits

    return outfits.sort_values(
        by="final_score",
        ascending=False
    )


def display_outfits(outfits):

    top_outfits = outfits.head(4)

    for index, outfit in top_outfits.iterrows():

        top_image = Image.open(
            outfit["top_image"]
        )

        bottom_image = Image.open(
            outfit["bottom_image"]
        )

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(top_image)
        plt.title(
            f'{outfit["top"]}\n'
            f'{outfit["top_color"]}'
        )
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(bottom_image)
        plt.title(
            f'{outfit["bottom"]}\n'
            f'{outfit["bottom_color"]}'
        )
        plt.axis("off")

        plt.suptitle(
            f'Outfit Score: {outfit["final_score"]} | '
            f'Total Price: ₹{outfit["total_price"]}'
        )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    results = recommend_outfits("U001")

    print("Top Personalized Outfits:")
    print(
        results[
            [
                "top_id",
                "bottom_id",
                "total_price",
                "final_score"
            ]
        ].head(4)
    )

    display_outfits(results)