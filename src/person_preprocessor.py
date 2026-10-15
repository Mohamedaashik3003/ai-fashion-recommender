from PIL import Image
import os


OUTPUT_DIR = "data/tryon"


def preprocess_person_image(
    image,
    output_name="person.jpg"
):

    image = image.convert("RGB")

    image.thumbnail(
        (768, 1024),
        Image.Resampling.LANCZOS
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    image.save(
        output_path,
        quality=95
    )

    return output_path


if __name__ == "__main__":

    print(
        "Person image preprocessing module ready."
    )