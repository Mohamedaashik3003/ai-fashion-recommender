import pandas as pd


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
                    "total_price": total_price,
                    "compatibility": compatibility,
                    "preference_score": preference_score,
                    "interaction_score": interaction_score,
                    "final_score": final_score
                })

    outfits = pd.DataFrame(outfits)

    if outfits.empty:
        return outfits

    outfits = outfits.sort_values(
        by="final_score",
        ascending=False
    )

    return outfits


if __name__ == "__main__":

    results = recommend_outfits("U001")

    print("Personalized Outfit Recommendations:")

    print(
        results[
            [
                "top_id",
                "top",
                "top_color",
                "bottom_id",
                "bottom",
                "bottom_color",
                "total_price",
                "compatibility",
                "preference_score",
                "interaction_score",
                "final_score"
            ]
        ]
    )