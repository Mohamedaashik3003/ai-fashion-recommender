import pandas as pd


INPUT_FILE = "data/fashion_metadata.csv"
OUTPUT_FILE = "data/fashion_subset.csv"


df = pd.read_csv(
    INPUT_FILE
)


sample_sizes = {
    "Top": 1500,
    "Dress": 1500,
    "Bottom": 1500,
    "Footwear": 300,
    "Other": 200
}


subset_parts = []


for category, sample_size in sample_sizes.items():

    category_data = df[
        df["category"] == category
    ]

    available = len(
        category_data
    )

    if available == 0:

        print(
            f"Skipping {category}: "
            f"no images available."
        )

        continue

    actual_size = min(
        sample_size,
        available
    )

    sample = category_data.sample(
        n=actual_size,
        random_state=42
    )

    subset_parts.append(
        sample
    )


subset = pd.concat(
    subset_parts,
    ignore_index=True
)


subset = subset.sample(
    frac=1,
    random_state=42
).reset_index(
    drop=True
)


subset.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "Fashion subset created successfully."
)

print(
    "Total images:",
    len(subset)
)

print(
    "\nCategory distribution:"
)

print(
    subset["category"].value_counts()
)

print(
    "\nUnique products:",
    subset["product_name"].nunique()
)

print(
    "\nFirst 10 records:"
)

print(
    subset.head(10).to_string(
        index=False
    )
)