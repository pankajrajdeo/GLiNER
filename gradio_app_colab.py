import gradio as gr
import numpy as np
from gliner import GLiNER
import re
import html
import hashlib
import sys

# Function to generate consistent colors from entity type names
def get_entity_color(entity_type):
    """Generate a consistent color based on the entity type string."""
    # Use hash of the entity type to generate consistent colors
    hash_obj = hashlib.md5(entity_type.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Convert first 6 characters of hash to RGB values
    r = int(hash_hex[:2], 16)
    g = int(hash_hex[2:4], 16)
    b = int(hash_hex[4:6], 16)
    
    # Ensure colors are vibrant enough (boost saturation/value)
    # Calculate HSV from RGB
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    
    # Adjust saturation and brightness for readability
    if max_val > 0:
        # Boost saturation
        saturation_boost = 0.7
        if max_val != min_val:
            r = int(r + (max_val - r) * saturation_boost)
            g = int(g + (max_val - g) * saturation_boost)
            b = int(b + (max_val - b) * saturation_boost)
        
        # Ensure brightness is appropriate for colored backgrounds with white text
        brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        if brightness > 0.65:  # If too bright, darken it
            r = int(r * 0.8)
            g = int(g * 0.8)
            b = int(b * 0.8)
        elif brightness < 0.2:  # If too dark, lighten it
            r = min(255, int(r * 1.5))
            g = min(255, int(g * 1.5))
            b = min(255, int(b * 1.5))
    
    # Convert back to hex
    return f"#{r:02x}{g:02x}{b:02x}"

# Dictionary to cache colors for entity types
entity_color_cache = {}

def get_color_for_entity(entity_type):
    if entity_type not in entity_color_cache:
        entity_color_cache[entity_type] = get_entity_color(entity_type)
    return entity_color_cache[entity_type]

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    model = GLiNER.from_pretrained(model_path, load_tokenizer=True)
    model.eval()
    return model

def filter_overlapping_entities(entities):
    # Sort entities by score in descending order
    sorted_entities = sorted(entities, key=lambda x: x.get("score", 0), reverse=True)
    
    # Keep track of token positions that are already covered
    covered_positions = set()
    filtered_entities = []
    
    for entity in sorted_entities:
        start = entity.get("start") or entity.get("token_start", 0)
        end = entity.get("end") or entity.get("token_end", 0)
        
        # Check if this entity overlaps with any already covered position
        current_positions = set(range(start, end + 1))
        if not current_positions.intersection(covered_positions):
            filtered_entities.append(entity)
            covered_positions.update(current_positions)
    
    return filtered_entities

# The fixed highlight_entities function to match GLiNER's output format
def highlight_entities(text, entities):
    safe_text = html.escape(text)
    words = safe_text.split()
    
    word_entities = {}
    for entity in entities:
        # Check the entity format and adapt accordingly
        start = entity.get("start") or entity.get("token_start") 
        end = entity.get("end") or entity.get("token_end")
        # GLiNER might use 'label' instead of 'type'
        entity_type = entity.get("type") or entity.get("label")
        
        if start is None or end is None or entity_type is None:
            print(f"Warning: Skipping entity with invalid format: {entity}")
            continue
            
        if start >= len(words) or end >= len(words):
            continue
            
        for i in range(start, end + 1):
            if i < len(words):
                if i not in word_entities:
                    word_entities[i] = []
                word_entities[i].append((entity_type, start, end))
    
    # Print the first entity to debug
    if entities and len(entities) > 0:
        print("First entity format:", entities[0])
        
    # Rest of function remains the same
    result = []
    for i, word in enumerate(words):
        if i in word_entities:
            for entity_type, start, end in word_entities[i]:
                color = get_color_for_entity(entity_type)
                result.append(f'<span style="background-color: {color};">{word}</span>')
                
                if i == end:
                    result.append(f' <span style="background-color: {color}; color: white; padding: 0px 4px; border-radius: 3px; font-size: 0.8em; font-weight: bold;">{entity_type}</span> ')
                else:
                    result.append(' ')
                break
        else:
            result.append(word + ' ')
    
    return ''.join(result)

def predict_entities(text, labels_input, threshold, allow_nested):
    labels = [label.strip() for label in labels_input.split(",")]
    
    # First, print what the entities look like to debug
    entities = model.predict_entities(text, labels, threshold=threshold)
    
    # Print sample of the entity format
    print(f"Entity format example: {entities[0] if entities else 'No entities found'}")
    
    if not allow_nested:
        entities = filter_overlapping_entities(entities)
    
    highlighted_html = highlight_entities(text, entities)
    
    entity_list = "<div style='margin-top: 15px;'>"
    if entities:
        entity_list += "<p><strong>Detected entities:</strong></p><ul>"
        for e in entities:
            # Adapt to possible different key names
            entity_text = e.get("text", "")
            entity_type = e.get("type") or e.get("label", "")
            entity_score = e.get("score", 0.0)
            
            entity_list += f"<li>{entity_text} - {entity_type} (score: {entity_score:.2f})</li>"
        entity_list += "</ul>"
    else:
        entity_list += "<p>No entities detected.</p>"
    entity_list += "</div>"
    
    return highlighted_html + entity_list

example_biomedical_text = """Analysis of peptides and proteins by temperature-responsive chromatographic system using N-isopropylacrylamide polymer-modified columns. A new method of HPLC using packing materials modified with a temperature responsive polymer, poly(N-isopropylacrylamide) (PIPAAm), was developed. Homogeneous PIPAAm polymer and its copolymer with butyl methacrylate (BMA) were synthesized and grafted to aminopropyl silica by activated ester-amine coupling and they were used as packing materials."""

example_biomedical_labels = "CONCEPT,CHEMICAL_ENTITY,METHOD_OR_TECHNIQUE,PHYSICAL_PROPERTY,GENE_OR_GENE_PRODUCT"

def create_interface(model_path):
    global model
    model = load_model(model_path)
    
    with gr.Blocks(title="NuNER Zero", css="span { display: inline; }") as demo:
        gr.Markdown("# NuNER Zero")
        
        with gr.Accordion("How to run this model locally", open=False):
            gr.Markdown("Instructions for running the model locally would go here.")
        
        with gr.Column():
            text_input = gr.Textbox(
                label="Text input",
                placeholder="Enter text here...",
                value=example_biomedical_text,
                lines=5
            )
            
            labels_input = gr.Textbox(
                label="Labels",
                placeholder="Enter comma-separated labels",
                value=example_biomedical_labels,
                lines=2
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    threshold = gr.Slider(
                        label="Threshold",
                        minimum=0.1,
                        maximum=0.9,
                        value=0.3,
                        step=0.05
                    )
                    gr.Markdown("Lower the threshold to increase how many entities get predicted.")
                    
                with gr.Column(scale=1):
                    gr.Markdown("Allow for nested NER?")
                    nested_ner = gr.Checkbox(label="Nested NER", value=False)
            
            gr.Markdown("### Predicted Entities")
            output = gr.HTML()
            
            submit_btn = gr.Button("Submit", variant="primary")
            submit_btn.click(
                fn=predict_entities,
                inputs=[text_input, labels_input, threshold, nested_ner],
                outputs=output
            )
    
    return demo

# For Colab, use a simplified argument handling approach
model_path = "/content/GLiNER/models/bioformer_ner/checkpoint-562"  # Default path

# Create and launch the interface
demo = create_interface(model_path)
demo.launch(share=True)
