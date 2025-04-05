
import os
import io
import requests
from dotenv import load_dotenv
from PIL import Image
import time # Import time for potential retries or delays

# Load environment variables
load_dotenv()
CLIPDROP_API_KEY = os.getenv("CLIPDROP_API_KEY")

# --- Crucial Check ---
if not CLIPDROP_API_KEY:
    # This error will stop the script if the key isn't found.
    raise ValueError("❌ CLIPDROP_API_KEY environment variable not set or found. Please ensure it's in your .env file or system environment.")
else:
    # Optional: Print confirmation that key was loaded (for debugging)
    print("✅ CLIPDROP_API_KEY loaded successfully.")

# Output directory for images
OUTPUT_DIR = "PANEL_IMAGES"
# Ensure the directory exists before trying to save into it
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Output directory '{OUTPUT_DIR}' checked/created.")
except OSError as e:
    # Handle potential errors during directory creation (e.g., permission issues)
    raise OSError(f"❌ Failed to create output directory '{OUTPUT_DIR}': {e}")


# Art style mappings
STYLE_MAPPINGS = {
    "Manga": "High-contrast black and white sketch with sharp, clean lines, exaggerated facial expressions, and dramatic shading. No bright colors, only grayscale tones",
    "Anime": "Vibrant colors with smooth cel shading, large expressive eyes, and detailed hair. Dynamic action poses with fluid motion lines",
    "American": "Bold outlines with heavy inking, bright and saturated colors, and exaggerated muscular features. Classic comic book style",
    "Belgian": "Clean, clear lines with soft, flat shading. Rich and detailed backgrounds in a semi-realistic style, inspired by Tintin comics",
}

# System instructions
SYSTEM_INSTRUCTIONS = """
STRICT INSTRUCTIONS:
Generate a high-quality, visually appealing image, consisting of the following elements:
- No speech bubbles, no text, no symbols, no gibberish language.
- Only clear, clean, and high-quality visual details.
- Do NOT add any text or letters in the image.
- No distorted, strange, unrealistic, or ugly facial features or elements.
- Ensure realistic proportions, natural expressions, and artistic coherence.
"""

# --- API Endpoint ---
CLIPDROP_API_URL = "https://clipdrop-api.co/text-to-image/v1"

def generate_images(panel_data, art_style):
    """Generates images (one per panel) based on panel descriptions using Clipdrop API."""
    print(f"--- Starting Image Generation for Style: {art_style} ---") # Debug Start

    if art_style not in STYLE_MAPPINGS:
        print(f"❌ Invalid art style selected: {art_style}") # Log error
        raise ValueError(f"Invalid art style! Choose from: {', '.join(STYLE_MAPPINGS.keys())}.")

    if not isinstance(panel_data, list) or not panel_data:
         print(f"❌ Invalid or empty panel_data received: {panel_data}") # Log error
         # Return empty list or raise error, depending on desired behavior
         return [] # Returning empty list if input is bad

    image_paths = []
    max_retries = 2 # Number of retries on failure
    retry_delay = 3 # Seconds to wait between retries

    # Loop through all panels in the provided data
    for i, panel in enumerate(panel_data):
        # Basic check for panel structure
        if not isinstance(panel, dict) or "Description" not in panel:
             print(f"⚠️ Skipping panel {i+1}: Invalid format or missing 'Description'. Panel data: {panel}")
             continue # Skip this panel

        panel_description = panel.get("Description", "Missing description") # Use .get for safety
        print(f"\n🔄 Generating image for Panel {i+1}...")
        print(f"   Description: {panel_description[:100]}...") # Print first 100 chars

        # --- Construct the Prompt ---
        full_prompt = (
            f"{panel_description}.\n"
            f"Art Style: {STYLE_MAPPINGS[art_style]}.\n"
            f"{SYSTEM_INSTRUCTIONS}"
        )
        # print(f"   Full Prompt (first 100 chars): {full_prompt[:100]}...") # Uncomment for deep debug

        # --- Prepare Request ---
        headers = {"x-api-key": CLIPDROP_API_KEY}
        payload = {"prompt": full_prompt}

        # --- Make API Call with Retries ---
        response = None
        for attempt in range(max_retries + 1):
            try:
                print(f"   Attempt {attempt + 1}/{max_retries + 1}: Calling Clipdrop API...")
                response = requests.post(
                    CLIPDROP_API_URL,
                    headers=headers,
                    json=payload, # Use json=payload
                    timeout=90 # Increased timeout slightly
                )
                print(f"   API Response Status Code: {response.status_code}") # Debug Status

                # Check response status
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '').lower()
                    print(f"   Response Content-Type: {content_type}")
                    if 'image' in content_type:
                        # Check if content is empty even on success
                        if not response.content:
                             print(f"   ⚠️ Warning: Received status 200 and image Content-Type, but response body is empty.")
                             response = None # Treat as failure
                             if attempt < max_retries: time.sleep(retry_delay)
                             continue # Retry if possible
                        break # SUCCESS - Exit retry loop
                    else:
                        print(f"   ⚠️ Warning: Received status 200, but Content-Type is not image ({content_type}). Response text: {response.text[:500]}...")
                        response = None # Treat as failure
                        if attempt < max_retries: time.sleep(retry_delay)
                        continue # Retry if possible

                elif response.status_code == 401 or response.status_code == 403:
                     print(f"   ❌ Error from API ({response.status_code}): Invalid API Key or insufficient permissions. Check your key.")
                     # No point retrying key errors, break inner loop
                     response = None
                     break
                elif response.status_code == 429:
                     print("   ❌ API Rate Limit Hit (429).")
                     response = None
                     wait_time = retry_delay * (attempt + 2) * 2 # Exponential backoff might be better
                     print(f"   Waiting {wait_time}s before retrying...")
                     time.sleep(wait_time)
                     continue
                elif response.status_code == 400:
                     print(f"   ❌ Error from API ({response.status_code}): Bad Request. Check prompt or parameters. Response: {response.text[:500]}...")
                     # May not be worth retrying bad requests
                     response = None
                     break
                else:
                    # Other errors (e.g., 5xx Server Errors)
                    print(f"   ❌ Error from API (Status {response.status_code}): {response.text[:500]}...")
                    response = None
                    if attempt < max_retries:
                        print(f"   Waiting {retry_delay}s before retrying...")
                        time.sleep(retry_delay)
                    continue # Retry if possible

            except requests.exceptions.Timeout:
                print(f"   ❌ Timeout Error on attempt {attempt + 1}. The API took too long to respond.")
                response = None
                if attempt < max_retries: time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Network or Request Error on attempt {attempt + 1}: {e}")
                response = None
                if attempt < max_retries: time.sleep(retry_delay)
            # Outer loop continues to next attempt if needed

        # --- Process Successful Response (only if response is valid after retries) ---
        if response and response.status_code == 200 and response.content:
            try:
                print("   Processing received image data...")
                image_bytes = response.content
                image = Image.open(io.BytesIO(image_bytes))

                # Ensure image is in RGB format
                if image.mode != "RGB":
                    print(f"   Converting image from {image.mode} to RGB.")
                    image = image.convert("RGB")

                # --- Save Image ---
                image_filename = f"panel_{i+1}.png"
                image_path = os.path.join(OUTPUT_DIR, image_filename)

                print(f"   Attempting to save image to: {image_path}")
                image.save(image_path, format="PNG") # Explicitly save as PNG
                image_paths.append(image_path) # Append path ONLY if save successful
                print(f"   ✅ Image for Panel {i+1} saved successfully.")

            except FileNotFoundError:
                # This specifically catches if OUTPUT_DIR doesn't exist at save time (unlikely with check above, but safe)
                 print(f"   ❌ Error Saving Image: Output directory '{OUTPUT_DIR}' not found.")
            except IOError as e:
                 # Catches issues like permission errors during save, disk full, etc.
                 print(f"   ❌ Error Saving Image (IOError) for panel {i+1}: {e}")
            except Exception as e:
                 # Catch other potential PIL errors during open/convert/save
                 print(f"   ❌ Error processing/saving image data for panel {i+1}: {e}")
                 print(f"   Failed Image Size (if available): {len(image_bytes)} bytes")


        else:
            # This block executes if all retries failed or response was invalid/empty
            print(f"   ❌ Failed to generate image for panel {i+1} after all attempts or due to critical API error.")
            # Do NOT append a path here

    print(f"\n--- Image Generation Finished ---")
    print(f"   Successfully generated and saved {len(image_paths)} images.") # Debug Count
    if len(image_paths) != len(panel_data):
         print(f"   ⚠️ Warning: Expected {len(panel_data)} images based on input, but only generated {len(image_paths)}. Check logs above for errors on specific panels.")

    # Return the list of paths. It will be empty or shorter than 6 if errors occurred.
    return image_paths

# Example usage (for testing this script directly)
if __name__ == "__main__":
    print("\n--- Running Test Generation ---")
    if not CLIPDROP_API_KEY:
        print("Cannot run test: CLIPDROP_API_KEY not found.")
    else:
        test_panels = [
            {"Description": "A cyberpunk street market at night, rain falling, neon signs reflecting on wet pavement."},
            {"Description": "Close up on a mysterious figure in a trench coat looking at a glowing data-pad."},
            {"Description": "A security drone flies overhead, scanning the area with a red light."},
            {"Description": "The figure quickly hides the data-pad and blends into the crowd."},
            {"Description": "Wide shot of the crowded market, vendors selling exotic tech and food."},
            {"Description": "The figure slips into a dark alleyway, disappearing from view."},
        ]

        try:
            # Test with "American" style
            generated_paths = generate_images(test_panels, "American")
            print("\n--- Test Result ---")
            print(f"Generated image paths: {generated_paths}")
            print(f"Number of images generated: {len(generated_paths)}")
            if generated_paths:
                 print(f"Check the '{OUTPUT_DIR}' folder for the generated images.")
            if len(generated_paths) != len(test_panels):
                 print(f"*** Note: Not all images were generated successfully. Review logs above. ***")

        except ValueError as ve:
            print(f"Test failed with ValueError: {ve}")
        except OSError as oe:
             print(f"Test failed with OSError (Directory issue?): {oe}")
        except Exception as ex:
            print(f"Test failed with unexpected error: {ex}")