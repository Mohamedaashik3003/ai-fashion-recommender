import pandas as pd
import random


INPUT_FILE = "data/fashion_subset.csv"
OUTPUT_FILE = "data/real_interactions.csv"


df = pd.read_csv(INPUT_FILE)

random.seed(42)

users = [
    "U001",
    "U002",
    "U003",
    "U004",
    "U005"
]

interactions = [
    "view",
    "like",
    "dislike",
    "wishlist",
    "cart"
]

records = []

for user_id in users:

    sample = df.sample(
        n=50,
        random_state=random.randint(
            1,
            10000
        )
    )

    for _, row in sample.iterrows():

        records.append({
            "user_id": user_id,
            "product_name": row["product_name"],
            "image_path": row["image_path"],
            "category": row["category"],
            "interaction": random.choice(
                interactions
            )
        })


interaction_df = pd.DataFrame(
    records
)

interaction_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "Real interaction dataset created successfully."
)

print(
    "Total interactions:",
    len(interaction_df)
)

print(
    "\nInteraction distribution:"
)

print(
    interaction_df["interaction"].value_counts()
)

print(
    "\nFirst 10 records:"
)

print(
    interaction_df.head(10).to_string(
        index=False
    )
)