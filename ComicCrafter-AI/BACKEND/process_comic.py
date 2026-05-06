"""
Enhanced Comic Processing Module
=================================
Advanced comic assembly with:
- Dynamic emotion-based panel layouts
- Adaptive panel sizing based on story pacing
- Professional PDF generation with comic formatting
- Multiple export formats (PNG, PDF, Individual panels)
- Evaluation metrics integration

Research Contributions:
- Novel dynamic layout algorithms for AI comics
- Emotion-aware panel composition
- Quality-aware comic assembly
"""

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Dict, Tuple, Optional, Any
import io
import math

# PDF generation imports
try:
    from reportlab.lib.pagesizes import A4, letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Spacer, PageBreak
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch, cm
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ ReportLab not available. PDF generation will be limited.")


# Configuration constants
DEFAULT_FONT_SIZE = 36
TITLE_FONT_SIZE = 48
TEXT_HEIGHT = 80
PANEL_SPACING = 12
BORDER_THICKNESS = 4
TEXT_BOX_BORDER = 3
OUTLINE_THICKNESS = 2
MARGIN = 20

# Emotion-based border styles
EMOTION_BORDER_STYLES = {
    "joy": {"color": "#FFD700", "style": "solid", "thickness": 3},
    "sadness": {"color": "#4A6FA5", "style": "solid", "thickness": 2},
    "fear": {"color": "#2F4F4F", "style": "broken", "thickness": 4},
    "anger": {"color": "#8B0000", "style": "double", "thickness": 5},
    "surprise": {"color": "#FF6B35", "style": "none", "thickness": 0},
    "tension": {"color": "#1A1A2E", "style": "solid", "thickness": 4},
    "excitement": {"color": "#FF4500", "style": "solid", "thickness": 3},
    "love": {"color": "#FF69B4", "style": "rounded", "thickness": 2},
    "mystery": {"color": "#4B0082", "style": "solid", "thickness": 3},
    "action": {"color": "#DC143C", "style": "broken", "thickness": 3},
    "comedy": {"color": "#32CD32", "style": "wavy", "thickness": 2},
    "dramatic": {"color": "#000000", "style": "double", "thickness": 4},
    "neutral": {"color": "#000000", "style": "solid", "thickness": 3}
}

# Panel size multipliers based on relative_size
SIZE_MULTIPLIERS = {
    "small": 0.8,
    "medium": 1.0,
    "large": 1.3,
    "splash": 2.0
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a font with fallback options."""
    font_options = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "Helvetica.ttf",
        "FreeSans.ttf"
    ]
    
    if bold:
        font_options = ["arialbd.ttf", "Arial Bold.ttf"] + font_options
    
    for font_name in font_options:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    
    print("⚠️ No TrueType fonts found. Using default font.")
    return ImageFont.load_default()


def load_default_font(size: int) -> ImageFont.FreeTypeFont:
    """Loads Arial font or falls back to PIL default."""
    return load_font(size)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def add_border(image: Image.Image, border_thickness: int, 
               color: str = "black", style: str = "solid") -> Image.Image:
    """
    Adds a styled border around the image.
    
    Args:
        image: PIL Image
        border_thickness: Border width in pixels
        color: Border color (name or hex)
        style: Border style (solid, broken, double, none)
    
    Returns:
        Image with border
    """
    if border_thickness <= 0 or style == "none":
        return image
    
    # Parse color
    if color.startswith('#'):
        border_color = hex_to_rgb(color)
    else:
        border_color = color
    
    new_width = image.width + 2 * border_thickness
    new_height = image.height + 2 * border_thickness
    
    if style == "solid":
        bordered_image = Image.new("RGB", (new_width, new_height), border_color)
        bordered_image.paste(image, (border_thickness, border_thickness))
    
    elif style == "double":
        # Double border effect
        outer_thickness = border_thickness
        inner_thickness = max(1, border_thickness // 2)
        gap = max(2, border_thickness // 3)
        
        total_border = outer_thickness + gap + inner_thickness
        bordered_image = Image.new("RGB", 
                                   (image.width + 2 * total_border, 
                                    image.height + 2 * total_border), 
                                   "white")
        
        # Draw outer border
        draw = ImageDraw.Draw(bordered_image)
        draw.rectangle([0, 0, bordered_image.width - 1, bordered_image.height - 1], 
                      outline=border_color, width=outer_thickness)
        # Draw inner border
        inner_offset = outer_thickness + gap
        draw.rectangle([inner_offset, inner_offset, 
                       bordered_image.width - 1 - inner_offset, 
                       bordered_image.height - 1 - inner_offset], 
                      outline=border_color, width=inner_thickness)
        
        bordered_image.paste(image, (total_border, total_border))
    
    elif style == "broken":
        # Broken/jagged border effect
        bordered_image = Image.new("RGB", (new_width, new_height), "white")
        bordered_image.paste(image, (border_thickness, border_thickness))
        
        draw = ImageDraw.Draw(bordered_image)
        # Draw broken line segments
        segment_length = 15
        for x in range(0, new_width, segment_length * 2):
            draw.line([(x, 0), (min(x + segment_length, new_width), 0)], 
                     fill=border_color, width=border_thickness)
            draw.line([(x, new_height - 1), (min(x + segment_length, new_width), new_height - 1)], 
                     fill=border_color, width=border_thickness)
        for y in range(0, new_height, segment_length * 2):
            draw.line([(0, y), (0, min(y + segment_length, new_height))], 
                     fill=border_color, width=border_thickness)
            draw.line([(new_width - 1, y), (new_width - 1, min(y + segment_length, new_height))], 
                     fill=border_color, width=border_thickness)
    
    else:
        # Default solid
        bordered_image = Image.new("RGB", (new_width, new_height), border_color)
        bordered_image.paste(image, (border_thickness, border_thickness))
    
    return bordered_image


def wrap_text(draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, 
              max_width: int) -> List[str]:
    """Wraps text into multiple lines based on max width."""
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
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)
    
    return lines


def draw_text_with_outline(draw: ImageDraw.Draw, position: Tuple[int, int], 
                           text: str, font: ImageFont.FreeTypeFont,
                           fill_color: str = "black", 
                           outline_color: str = "white",
                           outline_thickness: int = OUTLINE_THICKNESS):
    """Draws text with outline for better readability."""
    x, y = position
    for dx in range(-outline_thickness, outline_thickness + 1):
        for dy in range(-outline_thickness, outline_thickness + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=fill_color)


def add_text_below(image: Image.Image, text: str, font: ImageFont.FreeTypeFont,
                   text_height: int = TEXT_HEIGHT,
                   bg_color: str = "white",
                   text_color: str = "black") -> Image.Image:
    """Adds multiline text below each panel with padding and centering."""
    width, height = image.size
    new_height = height + text_height
    new_image = Image.new("RGB", (width, new_height), bg_color)
    new_image.paste(image, (0, 0))

    draw = ImageDraw.Draw(new_image)
    
    # Draw text box border
    text_box = [(0, height), (width - 1, new_height - 1)]
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
        draw_text_with_outline(draw, (text_x, text_y), line, font,
                              fill_color=text_color)
        text_y += line_height

    return new_image


def process_image(image_path: str, output_path: str, text: str = "Sample Text"):
    """Adds border, multiline text, and saves the modified image."""
    img = Image.open(image_path)
    font = load_default_font(DEFAULT_FONT_SIZE)

    img_with_border = add_border(img, BORDER_THICKNESS)
    final_image = add_text_below(img_with_border, text, font)

    final_image.save(output_path)
    print(f"Image saved at: {output_path}")


class DynamicComicLayout:
    """
    Dynamic comic layout engine with emotion-aware sizing.
    """
    
    def __init__(self, canvas_width: int = 1200, canvas_height: int = 1800):
        """
        Initialize the layout engine.
        
        Args:
            canvas_width: Total canvas width
            canvas_height: Total canvas height
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.margin = MARGIN
        self.gutter = PANEL_SPACING
    
    def calculate_dynamic_layout(self, panel_data: List[Dict], 
                                  num_columns: int = 2) -> List[Dict]:
        """
        Calculate dynamic panel positions based on emotions and sizes.
        
        Args:
            panel_data: List of panel dictionaries with emotion info
            num_columns: Base number of columns
        
        Returns:
            List of layout dictionaries with positions
        """
        num_panels = len(panel_data)
        layouts = []
        
        # Calculate available space
        available_width = self.canvas_width - (2 * self.margin)
        available_height = self.canvas_height - (2 * self.margin)
        
        # Calculate base cell size
        num_rows = math.ceil(num_panels / num_columns)
        base_cell_width = (available_width - (num_columns - 1) * self.gutter) // num_columns
        base_cell_height = (available_height - (num_rows - 1) * self.gutter) // num_rows
        
        # Get size multipliers for each panel
        sizes = []
        for panel in panel_data:
            relative_size = panel.get("relative_size", "medium")
            multiplier = SIZE_MULTIPLIERS.get(relative_size, 1.0)
            sizes.append(multiplier)
        
        # Calculate positions using row-based layout
        current_x = self.margin
        current_y = self.margin
        row_height = 0
        row_panels = []
        
        for i, panel in enumerate(panel_data):
            multiplier = sizes[i]
            
            # Calculate this panel's size
            is_splash = panel.get("relative_size") == "splash"
            
            if is_splash:
                # Splash panels take full width
                panel_width = available_width
                panel_height = int(base_cell_height * 1.5)
            else:
                panel_width = int(base_cell_width * min(multiplier, 1.3))
                panel_height = int(base_cell_height * min(multiplier, 1.2))
            
            # Check if we need to wrap to next row
            if current_x + panel_width > self.canvas_width - self.margin or is_splash:
                if row_panels:
                    # Finalize current row
                    current_x = self.margin
                    current_y += row_height + self.gutter
                    row_height = 0
                    row_panels = []
            
            # Get border style based on emotion
            emotion = panel.get("emotion", "neutral")
            border_info = EMOTION_BORDER_STYLES.get(emotion, EMOTION_BORDER_STYLES["neutral"])
            
            layout = {
                "panel_index": i,
                "x": current_x,
                "y": current_y,
                "width": panel_width,
                "height": panel_height,
                "emotion": emotion,
                "border_color": border_info["color"],
                "border_style": border_info["style"],
                "border_thickness": border_info["thickness"]
            }
            
            layouts.append(layout)
            row_panels.append(layout)
            
            current_x += panel_width + self.gutter
            row_height = max(row_height, panel_height)
        
        return layouts
    
    def create_comic_strip(self, panel_images: List[str], 
                           panel_texts: List[str],
                           panel_data: List[Dict] = None,
                           title: str = None) -> Image.Image:
        """
        Create a comic strip with dynamic layout.
        
        Args:
            panel_images: List of image paths
            panel_texts: List of dialogue texts
            panel_data: Optional panel data with emotion info
            title: Optional title for the comic
        
        Returns:
            Assembled comic strip image
        """
        # Load images
        loaded_images = []
        for path in panel_images:
            if os.path.exists(path):
                loaded_images.append(Image.open(path))
            else:
                print(f"⚠️ Image not found: {path}")
        
        if not loaded_images:
            raise ValueError("No valid images found")
        
        # Calculate layout
        if panel_data:
            layouts = self.calculate_dynamic_layout(panel_data)
        else:
            # Simple grid layout
            layouts = self._simple_grid_layout(len(loaded_images))
        
        # Calculate canvas size based on layouts
        max_y = max(l["y"] + l["height"] for l in layouts)
        title_height = 80 if title else 0
        total_height = max_y + self.margin + title_height
        
        # Create canvas
        comic = Image.new("RGB", (self.canvas_width, int(total_height)), "white")
        draw = ImageDraw.Draw(comic)
        
        # Add title if provided
        if title:
            title_font = load_font(TITLE_FONT_SIZE, bold=True)
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (self.canvas_width - title_width) // 2
            draw.text((title_x, 10), title, font=title_font, fill="black")
            
            # Adjust layouts for title
            for layout in layouts:
                layout["y"] += title_height
        
        # Place panels
        font = load_font(DEFAULT_FONT_SIZE)
        
        for i, (layout, img) in enumerate(zip(layouts, loaded_images)):
            # Resize image to fit layout
            target_width = layout["width"]
            target_height = layout["height"] - TEXT_HEIGHT
            
            # Maintain aspect ratio
            img_ratio = img.width / img.height
            target_ratio = target_width / target_height
            
            if img_ratio > target_ratio:
                new_width = target_width
                new_height = int(target_width / img_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * img_ratio)
            
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Add border based on emotion
            bordered_img = add_border(
                resized_img,
                layout.get("border_thickness", BORDER_THICKNESS),
                layout.get("border_color", "black"),
                layout.get("border_style", "solid")
            )
            
            # Add text below
            text = panel_texts[i] if i < len(panel_texts) else "..."
            final_panel = add_text_below(bordered_img, text, font)
            
            # Paste onto canvas
            paste_x = layout["x"]
            paste_y = layout["y"]
            
            # Center if panel is smaller than allocated space
            if final_panel.width < layout["width"]:
                paste_x += (layout["width"] - final_panel.width) // 2
            
            comic.paste(final_panel, (int(paste_x), int(paste_y)))
        
        return comic
    
    def _simple_grid_layout(self, num_panels: int) -> List[Dict]:
        """Create a simple grid layout."""
        layouts = []
        num_columns = 2
        num_rows = math.ceil(num_panels / num_columns)
        
        available_width = self.canvas_width - (2 * self.margin)
        available_height = self.canvas_height - (2 * self.margin)
        
        cell_width = (available_width - (num_columns - 1) * self.gutter) // num_columns
        cell_height = (available_height - (num_rows - 1) * self.gutter) // num_rows
        
        for i in range(num_panels):
            row = i // num_columns
            col = i % num_columns
            
            layout = {
                "panel_index": i,
                "x": self.margin + col * (cell_width + self.gutter),
                "y": self.margin + row * (cell_height + self.gutter),
                "width": cell_width,
                "height": cell_height,
                "border_color": "black",
                "border_style": "solid",
                "border_thickness": BORDER_THICKNESS
            }
            layouts.append(layout)
        
        return layouts


def create_comic_strip_with_text(panel_images: List[str], 
                                  panel_texts: List[str],
                                  output_image_path: str,
                                  panel_data: List[Dict] = None,
                                  title: str = None,
                                  use_dynamic_layout: bool = True):
    """
    Creates a comic strip with dynamic or standard layout.
    
    Args:
        panel_images: List of panel image paths
        panel_texts: List of dialogue texts
        output_image_path: Path to save the final comic
        panel_data: Optional panel data with emotion info
        title: Optional title for the comic
        use_dynamic_layout: Whether to use emotion-based layout
    """
    num_panels = len(panel_images)
    
    if num_panels != len(panel_texts):
        print(f"⚠️ Panel count mismatch: {num_panels} images, {len(panel_texts)} texts")
        # Pad texts if needed
        while len(panel_texts) < num_panels:
            panel_texts.append("...")
    
    # Check for missing images
    missing = [path for path in panel_images if not os.path.exists(path)]
    if missing:
        print(f"⚠️ Missing image files: {missing}")
        raise FileNotFoundError("Some panel images are missing!")
    
    if use_dynamic_layout and panel_data:
        # Use dynamic layout engine
        layout_engine = DynamicComicLayout()
        comic = layout_engine.create_comic_strip(
            panel_images, panel_texts, panel_data, title
        )
    else:
        # Use standard 2-column layout
        comic = _create_standard_layout(panel_images, panel_texts, title)
    
    comic.save(output_image_path)
    print(f"✅ Comic strip saved at {output_image_path}")


def _create_standard_layout(panel_images: List[str], 
                            panel_texts: List[str],
                            title: str = None) -> Image.Image:
    """Create comic with standard 2-column layout."""
    # Load and process panels
    processed_panels = []
    font = load_default_font(DEFAULT_FONT_SIZE)
    
    for i, img_path in enumerate(panel_images):
        img = Image.open(img_path)
        img_with_border = add_border(img, BORDER_THICKNESS)
        text = panel_texts[i] if i < len(panel_texts) else "..."
        img_with_text = add_text_below(img_with_border, text, font)
        processed_panels.append(img_with_text)
    
    # Calculate dimensions
    panel_width = max(p.width for p in processed_panels)
    panel_height = max(p.height for p in processed_panels)
    
    num_columns = 2
    num_rows = math.ceil(len(processed_panels) / num_columns)
    
    comic_width = panel_width * num_columns + PANEL_SPACING * (num_columns + 1)
    comic_height = panel_height * num_rows + PANEL_SPACING * (num_rows + 1)
    
    # Add title height if needed
    title_height = 80 if title else 0
    comic_height += title_height
    
    # Create comic
    comic = Image.new("RGB", (comic_width, comic_height), "white")
    draw = ImageDraw.Draw(comic)
    
    # Add title
    if title:
        title_font = load_font(TITLE_FONT_SIZE, bold=True)
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_text_width = title_bbox[2] - title_bbox[0]
        title_x = (comic_width - title_text_width) // 2
        draw.text((title_x, 15), title, font=title_font, fill="black")
    
    # Place panels
    for i, panel in enumerate(processed_panels):
        col = i % num_columns
        row = i // num_columns
        
        x = PANEL_SPACING + col * (panel_width + PANEL_SPACING)
        y = title_height + PANEL_SPACING + row * (panel_height + PANEL_SPACING)
        
        # Center panel if smaller
        x_offset = (panel_width - panel.width) // 2
        y_offset = (panel_height - panel.height) // 2
        
        comic.paste(panel, (x + x_offset, y + y_offset))
    
    return comic


def create_comic_pdf(comic_image_path: str, 
                     pdf_output_path: str,
                     story_data: Dict = None,
                     panel_data: List[Dict] = None,
                     evaluation_metrics: Dict = None) -> bool:
    """
    Create a professional PDF from the comic.
    
    Args:
        comic_image_path: Path to the comic image
        pdf_output_path: Path for the output PDF
        story_data: Optional story information to include
        panel_data: Optional panel data for additional pages
        evaluation_metrics: Optional quality metrics to include
    
    Returns:
        True if successful, False otherwise
    """
    if not REPORTLAB_AVAILABLE:
        print("❌ ReportLab not available. Cannot create PDF.")
        return False
    
    if not os.path.exists(comic_image_path):
        print(f"❌ Comic image not found: {comic_image_path}")
        return False

    def build_emotional_arc_summary(panels: List[Dict]) -> Tuple[str, float]:
        """Return emotion flow text and average intensity."""
        if not panels:
            return "", 0.0

        emotions = [str(p.get("emotion", "neutral")).upper() for p in panels]
        intensities = [p.get("emotion_intensity", p.get("intensity", 0.5)) for p in panels]
        try:
            intensity_avg = sum(float(x) for x in intensities) / len(intensities)
        except Exception:
            intensity_avg = 0.0

        flow = " -> ".join(emotions)
        return flow, intensity_avg
    
    try:
        # Create document
        doc = SimpleDocTemplate(
            pdf_output_path,
            pagesize=A4,
            leftMargin=30,
            rightMargin=30,
            topMargin=30,
            bottomMargin=30
        )
        
        styles = getSampleStyleSheet()
        story_elements = []
        
        # Add title if available
        if story_data and 'title' in story_data:
            title_style = ParagraphStyle(
                'ComicTitle',
                parent=styles['Heading1'],
                fontSize=24,
                alignment=1,  # Center
                spaceAfter=20
            )
            story_elements.append(Paragraph(story_data['title'], title_style))
        
        # Add comic image
        with Image.open(comic_image_path) as img:
            img_width, img_height = img.size
        
        aspect = img_height / float(img_width)
        page_width, page_height = A4
        
        display_width = page_width - 60  # Account for margins
        display_height = display_width * aspect
        
        if display_height > page_height - 100:
            display_height = page_height - 100
            display_width = display_height / aspect
        
        comic_img = RLImage(comic_image_path, width=display_width, height=display_height)
        comic_img.hAlign = 'CENTER'
        story_elements.append(comic_img)

        # Add emotional arc below the comic image if panel data exists
        if panel_data:
            arc_flow, intensity_avg = build_emotional_arc_summary(panel_data)
            story_elements.append(Spacer(1, 12))
            story_elements.append(Paragraph("Emotional Arc", styles['Heading2']))
            story_elements.append(Spacer(1, 6))
            if arc_flow:
                story_elements.append(Paragraph(f"<b>Emotion Flow:</b> {arc_flow}", styles['Normal']))
                story_elements.append(Spacer(1, 6))

            arc_table = Table(
                [["Average Intensity", f"{intensity_avg:.0%}"]],
                colWidths=[3*inch, 1.5*inch]
            )
            arc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black)
            ]))
            story_elements.append(arc_table)
        
        # Add story sections on new page if available
        if story_data:
            story_elements.append(PageBreak())
            story_elements.append(Paragraph("Story Details", styles['Heading2']))
            story_elements.append(Spacer(1, 12))
            
            section_style = styles['Normal']
            
            for section in ['introduction', 'storyline', 'climax', 'resolution', 'moral']:
                if section in story_data:
                    story_elements.append(Paragraph(f"<b>{section.capitalize()}:</b>", styles['Heading3']))
                    story_elements.append(Paragraph(story_data[section], section_style))
                    story_elements.append(Spacer(1, 8))
        
        # Add evaluation metrics if available
        if evaluation_metrics:
            story_elements.append(PageBreak())
            story_elements.append(Paragraph("Quality Evaluation", styles['Heading2']))
            story_elements.append(Spacer(1, 12))
            
            # Create metrics table
            metrics_data = [
                ["Metric", "Score"],
                ["Overall Quality", f"{evaluation_metrics.get('overall_score', 'N/A')}"],
                ["Visual Consistency", f"{evaluation_metrics.get('detailed_metrics', {}).get('visual_consistency_score', 'N/A')}"],
                ["Narrative Flow", f"{evaluation_metrics.get('detailed_metrics', {}).get('narrative_flow_score', 'N/A')}"],
                ["Emotion Conveyance", f"{evaluation_metrics.get('detailed_metrics', {}).get('emotion_conveyance_score', 'N/A')}"],
            ]
            
            table = Table(metrics_data, colWidths=[3*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story_elements.append(table)
        
        # Build PDF
        doc.build(story_elements)
        print(f"✅ PDF saved at {pdf_output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create PDF: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_individual_panels(panel_images: List[str], 
                             output_folder: str,
                             panel_texts: List[str] = None,
                             with_text: bool = True) -> List[str]:
    """
    Export individual processed panels.
    
    Args:
        panel_images: List of panel image paths
        output_folder: Folder to save processed panels
        panel_texts: Optional list of dialogue texts
        with_text: Whether to add text below panels
    
    Returns:
        List of exported panel paths
    """
    os.makedirs(output_folder, exist_ok=True)
    exported = []
    
    font = load_default_font(DEFAULT_FONT_SIZE)
    
    for i, img_path in enumerate(panel_images):
        if not os.path.exists(img_path):
            print(f"⚠️ Skipping missing image: {img_path}")
            continue
        
        img = Image.open(img_path)
        img_with_border = add_border(img, BORDER_THICKNESS)
        
        if with_text and panel_texts and i < len(panel_texts):
            final_img = add_text_below(img_with_border, panel_texts[i], font)
        else:
            final_img = img_with_border
        
        output_path = os.path.join(output_folder, f"processed_panel_{i+1}.png")
        final_img.save(output_path)
        exported.append(output_path)
        print(f"✅ Exported panel {i+1} to {output_path}")
    
    return exported


if __name__ == "__main__":
    input_folder = "PANEL_IMAGES"
    output_folder = "OUTPUT"

    os.makedirs(output_folder, exist_ok=True)

    input_paths = [os.path.join(input_folder, f"panel_{i+1}.png") for i in range(6)]
    texts = [f"Dialogue {i+1} - A sample text for testing" for i in range(6)]
    
    # Test with sample panel data
    sample_panel_data = [
        {"emotion": "neutral", "relative_size": "medium"},
        {"emotion": "tension", "relative_size": "medium"},
        {"emotion": "fear", "relative_size": "small"},
        {"emotion": "action", "relative_size": "large"},
        {"emotion": "dramatic", "relative_size": "splash"},
        {"emotion": "joy", "relative_size": "medium"}
    ]
    
    output_image_path = os.path.join(output_folder, "final_comic_strip.png")
    
    try:
        create_comic_strip_with_text(
            input_paths, texts, output_image_path,
            panel_data=sample_panel_data,
            title="Test Comic",
            use_dynamic_layout=True
        )
        print("✅ Test comic created successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    create_comic_strip_with_text(input_paths, texts, output_image_path)
