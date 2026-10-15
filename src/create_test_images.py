from PIL import Image, ImageDraw
import os

output_dir = "data/images"
os.makedirs(output_dir, exist_ok=True)

products = {
    "P001": ("Black T-Shirt", "black"),
    "P002": ("White T-Shirt", "white"),
    "P003": ("Blue Shirt", "blue"),
    "P004": ("Grey Hoodie", "grey"),
    "P005": ("Blue Jeans", "blue"),
    "P006": ("Black Trousers", "black"),
    "P007": ("Pink T-Shirt", "pink"),
    "P008": ("Red Dress", "red"),
    "P009": ("Blue Jeans", "blue"),
    "P010": ("Black Hoodie", "black")
}

for product_id, (name, color) in products.items():
    image = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((100, 80, 300, 400), fill=color)
    draw.text((120, 430), name, fill="black")

    image.save(f"{output_dir}/{product_id}.jpg")

print("10 test fashion images created successfully.")