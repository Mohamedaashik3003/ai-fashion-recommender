import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


FEATURE_FILE = "data/real_cnn_features.csv"
INTERACTION_FILE = "data/real_interactions.csv"
IMAGE_ROOT = "data/real_images/img"


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


def find_image(product_name):

    folder = os.path.join(
        IMAGE_ROOT,
        product_name
    )

    if not os.path.isdir(folder):
        return None

    images = [
        file
        for file in os.listdir(folder)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if not images:
        return None

    return os.path.join(
        folder,
        images[0]
    )


def display_outfits(
    results
):

    for index, row in results.iterrows():

        top_name = row["top"]
        bottom_name = row["bottom"]

        top_image = find_image(
            top_name
        )

        bottom_image = find_image(
            bottom_name
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 6)
        )

        if top_image:

            image = Image.open(
                top_image
            )

            axes[0].imshow(
                image
            )

        axes[0].set_title(
            top_name.replace(
                "_",
                " "
            )
        )

        axes[0].axis("off")

        if bottom_image:

            image = Image.open(
                bottom_image
            )

            axes[1].imshow(
                image
            )

        axes[1].set_title(
            bottom_name.replace(
                "_",
                " "
            )
        )

        axes[1].axis("off")

        fig.suptitle(
            f"Personalized Outfit #{index + 1}\n"
            f"Compatibility: "
            f"{row['compatibility_score']:.3f}   "
            f"Preference: "
            f"{row['preference_score']:.3f}   "
            f"Final Score: "
            f"{row['final_score']:.3f}",
            fontsize=12
        )

        plt.tight_layout()

        plt.show()


if __name__ == "__main__":

    from real_outfit_recommender import (
        recommend_outfits
    )

    user_id = "U001"

    results = recommend_outfits(
        user_id,
        top_n=5
    )

    if results.empty:

        print(
            "No outfit recommendations available."
        )

    else:

        print(
            "\nDisplaying top 5 personalized outfits..."
        )

        display_outfits(
            results
        )