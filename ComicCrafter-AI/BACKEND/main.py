import os
import generate_panels
import generate_image
import process_comic
from PIL import Image

PANEL_FOLDER = "PANEL_IMAGES"
OUTPUT_FOLDER = "OUTPUT"

os.makedirs(PANEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def main():

    # Step 1: Enter user Story Prompt
    user_prompt = input("Enter your story prompt: ").strip()

    # Step 2: Art Style Selection
    print("\n Choose an Art Style: Manga, Anime, American, Belgian")
    art_style = input("Enter your choice: ").strip().capitalize()
    
    valid_styles = ["Manga", "Anime", "American", "Belgian"]
    if art_style not in valid_styles:
        print(" Invalid style! Defaulting to 'Anime'")
        art_style = "Anime"

    # Step 3: Generate panel descriptions & dialogues
    print("\n Generating panel descriptions & dialogues...")
    panel_data = generate_panels.generate_panels(user_prompt, art_style)  

    # Step 4: Generate images for panels
    print("\n Generating images for comic panels...")
    image_paths = generate_image.generate_images(panel_data, art_style)
    if len(image_paths) != 6:
        print(" Failed to generate all panel images.")
        return

    # Step 5: Extract panel dialogues
    panel_texts = [panel["Text"] for panel in panel_data]

    # Step 6: Create final comic strip
    print("\n Creating the final comic strip...")
    
    output_path = os.path.join(OUTPUT_FOLDER, "comic_strip_with_text.png")
    process_comic.create_comic_strip_with_text(image_paths, panel_texts, output_path)
    print(f"\nComic generation complete! Check '{output_path}'")


if __name__ == "__main__":
    main()
