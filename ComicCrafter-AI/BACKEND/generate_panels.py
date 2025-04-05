import os
import re
from dotenv import load_dotenv
import google.generativeai as genai  

def load_api_keys():
    """Loads API keys from .env file."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
    return api_key

def configure_gemini(api_key):
    """Configures the Gemini AI model."""
    genai.configure(api_key=api_key)

def generate_panels(scenario, art_style):
    """
    Generates six structured comic panels based on the given scenario and art style.
    Returns a list of dictionaries containing descriptions and dialogues.
    """
    template = """
    You are a professional comic book creator.
    You will be given a short scenario, and you must split it into exactly 6 comic panels.
    **Art Style:** {art_style}
    For each comic panel, provide:
    1. **Description**: A detailed background and character description (comma-separated, not full sentences).
    2. **Text**: Exact dialogue in quotation marks, or if no dialogue, leave it empty or use `...`.
    Ensure all text is clear, meaningful, and in proper English.
    Format:
    # Panel 1
    Description: [Background and character details]
    Text: "[Character]: [Dialogue]" OR "..." if no dialogue.
    # Panel 2
    Description: [Background and character details]
    Text: "[Character]: [Dialogue]" OR "..." if no dialogue.
    # end
    Short Scenario:
    {scenario}
    """
    formatted_prompt = template.format(scenario=scenario, art_style=art_style)
    
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(formatted_prompt)
    
    if not response or not response.text:
        raise Exception("Error: Failed to generate panel descriptions.")
    
    return extract_panel_info(response.text.strip())

def extract_panel_info(text):
    """Extracts structured panel descriptions and dialogues from the generated text."""
    panel_info_list = []
    panel_blocks = re.split(r"# Panel \d+", text)
    
    for block in panel_blocks:
        if block.strip():
            panel_info = {}
            desc_match = re.search(r"Description:\s*(.+)", block, re.IGNORECASE)
            panel_info['Description'] = desc_match.group(1).strip() if desc_match else "Unknown scene."
            text_match = re.findall(r'Text:\s*"([^"]+)"', block, re.IGNORECASE | re.DOTALL)
            panel_info['Text'] = " ".join(text_match) if text_match else "..."
            panel_info_list.append(panel_info)
    
    if len(panel_info_list) != 6:
        raise ValueError(f"Expected 6 panels, but got {len(panel_info_list)}. Check Gemini's output.")
    
    return panel_info_list

# --- New Functions for Story Generation ---
def generate_story(scenario, art_style):
    """
    Generates a structured story with a title, introduction, storyline, climax, and moral.
    """
    template = """
    You are a professional storyteller.
    Given a short scenario, create a structured story with a title and exactly 4 sections:
    **Title**: A catchy title for the story.
    1. **Introduction**: Introduce the main character(s) and setting in 1-2 sentences.
    2. **Storyline**: Describe the sequence of events leading to the climax in 2-3 sentences.
    3. **Climax**: Highlight the peak action or turning point in 1-2 sentences.
    4. **Moral**: Provide a concluding lesson or takeaway in 1 sentence.
    **Art Style Context:** {art_style}
    Format:
    # Title
    [Title text]
    # Introduction
    [Text]
    # Storyline
    [Text]
    # Climax
    [Text]
    # Moral
    [Text]
    Short Scenario:
    {scenario}
    """
    formatted_prompt = template.format(scenario=scenario, art_style=art_style)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(formatted_prompt)
    if not response or not response.text:
        raise Exception("Error: Failed to generate story.")
    return extract_story_info(response.text.strip())

def extract_story_info(text):
    """
    Extracts the story sections from the generated text.
    """
    story_info = {}
    sections = re.split(r"#\s*(\w+)", text)
    for i in range(1, len(sections), 2):
        section_name = sections[i].lower()
        section_content = sections[i+1].strip()
        if section_name in ["title", "introduction", "storyline", "climax", "moral"]:
            story_info[section_name] = section_content
    return story_info

if __name__ == '__main__':
    try:
        gemini_api_key = load_api_keys()
        configure_gemini(gemini_api_key)

        scenario = input("Enter your short comic scenario: ")
        print("\nChoose an art style: Manga, Anime, American, Belgian")
        art_style = input("Enter art style: ").strip().capitalize()

        valid_styles = ["Manga", "Anime", "American", "Belgian"]
        if art_style not in valid_styles:
            print("Invalid art style! Defaulting to 'Anime'.")
            art_style = "Anime"

        panels = generate_panels(scenario, art_style)
        
        for i, panel in enumerate(panels, 1):
            print(f"\nPanel {i}:")
            print(f"Description: {panel['Description']}")
            print(f"Text: {panel['Text']}")
    except Exception as e:
        print(f"Something went wrong: {e}")