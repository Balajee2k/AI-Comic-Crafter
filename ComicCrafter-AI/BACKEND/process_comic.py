import os
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT_SIZE = 42
TEXT_HEIGHT = 100
PANEL_SPACING = 15
BORDER_THICKNESS = 6
TEXT_BOX_BORDER = 4
OUTLINE_THICKNESS = 2


def load_default_font(size):
    """Loads Arial font or falls back to PIL default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        print("Arial font not found! Using Pillow's default font.")
        return ImageFont.load_default()


def add_border(image, border_thickness, color="black"):
    """Adds a bold border around the image."""
    bordered_image = Image.new(
        "RGB",
        (image.width + 2 * border_thickness, image.height + 2 * border_thickness),
        color
    )
    bordered_image.paste(image, (border_thickness, border_thickness))
    return bordered_image


def wrap_text(draw, text, font, max_width):
    """Wraps text into multiple lines based on panel width."""
    lines = []
    words = text.split()
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]

        if line_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)
    
    return lines


def draw_text_with_outline(draw, position, text, font, fill_color="black", outline_color="white", outline_thickness=OUTLINE_THICKNESS):
    """Draws text with outline for better readability."""
    x, y = position
    for dx in range(-outline_thickness, outline_thickness + 1):
        for dy in range(-outline_thickness, outline_thickness + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=fill_color)


def add_text_below(image, text, font, text_height=TEXT_HEIGHT):
    """Adds multiline text below each panel with padding and centering."""
    width, height = image.size
    new_height = height + text_height
    new_image = Image.new("RGB", (width, new_height), "white")
    new_image.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_image)
    
    text_box = [(0, height), (width, new_height)]
    draw.rectangle(text_box, outline="black", width=TEXT_BOX_BORDER)

    max_text_width = width - 20  
    lines = wrap_text(draw, text, font, max_text_width)

    line_height = font.getbbox("A")[3] - font.getbbox("A")[1]   

    total_text_height = len(lines) * line_height
    text_y = height + (text_height - total_text_height) // 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (width - text_width) // 2
        draw_text_with_outline(draw, (text_x, text_y), line, font)
        text_y += line_height

    return new_image


def process_image(image_path, output_path, text="Sample Text"):
    """Adds border, multiline text, and saves the modified image."""
    img = Image.open(image_path)
    font = load_default_font(DEFAULT_FONT_SIZE)

    img_with_border = add_border(img, BORDER_THICKNESS)
    final_image = add_text_below(img_with_border, text, font)

    final_image.save(output_path)
    print(f"Image saved at: {output_path}")


def create_comic_strip_with_text(panel_images, panel_texts, output_image_path):
    """Combines six images into a 3x2 comic strip with multiline text on each panel."""

    if len(panel_images) != 6 or len(panel_texts) != 6:
        raise ValueError("There must be exactly 6 panel images and 6 panel texts.")

    missing = [path for path in panel_images if not os.path.exists(path)]
    if missing:
        print("Missing image files:", missing)
        raise FileNotFoundError("Some panel images are missing!")

    # Processing each panel
    processed_panels = []
    for i in range(6):
        img = Image.open(panel_images[i])
        font = load_default_font(DEFAULT_FONT_SIZE)
        img_with_text = add_text_below(img, panel_texts[i], font)
        processed_panels.append(img_with_text)

    width, height = processed_panels[0].size
    comic_width = width * 2      
    comic_height = height * 3    
    comic_strip = Image.new("RGB", (comic_width, comic_height), "white")

    for i, panel in enumerate(processed_panels):
        x = (i % 2) * width     
        y = (i // 2) * height    
        comic_strip.paste(panel, (x, y))

    comic_strip.save(output_image_path)
    print(f"Comic strip saved at {output_image_path}")


if __name__ == "__main__":
    input_folder = "PANEL_IMAGES"
    output_folder = "OUTPUT"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    input_paths = [os.path.join(input_folder, f"panel_{i+1}.png") for i in range(6)]
    texts = [f"Dialogue {i+1} with multiline wrapping support for long text" for i in range(6)]
    
    output_image_path = os.path.join(output_folder, "final_comic_strip.png")
    create_comic_strip_with_text(input_paths, texts, output_image_path)
