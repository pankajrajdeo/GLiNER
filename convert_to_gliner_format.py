import json
import re
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Also download punkt_tab which is needed for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    # Additional attempt to ensure models are loaded
    nltk.download('punkt_tab', quiet=True)

def tokenize_text(text):
    """Tokenizes the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

def find_entity_spans(tokenized_text, entity_text):
    """Find the token spans for a given entity in the text.
    
    Args:
        tokenized_text: List of tokens
        entity_text: Entity string to search for
    
    Returns:
        List of tuples (start_idx, end_idx) for all occurrences
    """
    entity_tokens = tokenize_text(entity_text.lower())
    if not entity_tokens:
        return []
        
    matches = []
    
    # Search for the entity in the tokenized text
    for i in range(len(tokenized_text) - len(entity_tokens) + 1):
        # Compare normalized versions (lowercase) to handle case differences
        token_span = tokenized_text[i:i + len(entity_tokens)]
        if " ".join([t.lower() for t in token_span]) == " ".join(entity_tokens).lower():
            matches.append((i, i + len(entity_tokens) - 1))
    
    return matches

def is_overlapping(span1, span2):
    """Check if two spans overlap.
    
    Args:
        span1: Tuple of (start_idx, end_idx)
        span2: Tuple of (start_idx, end_idx)
        
    Returns:
        True if the spans overlap, False otherwise
    """
    return max(span1[0], span2[0]) <= min(span1[1], span2[1])

def remove_overlapping_entities(entities):
    """Remove overlapping entities, keeping the longest one or the one that comes first in case of a tie.
    
    Args:
        entities: List of [start_idx, end_idx, entity_type]
        
    Returns:
        List of non-overlapping entities
    """
    if not entities:
        return []
    
    # Sort by span length (descending) and then by start index (ascending)
    sorted_entities = sorted(entities, key=lambda x: (-(x[1] - x[0]), x[0]))
    
    non_overlapping = []
    used_spans = set()
    
    for entity in sorted_entities:
        start, end, entity_type = entity
        span = (start, end)
        
        # Check if this span overlaps with any already used span
        if not any(is_overlapping(span, used_span) for used_span in used_spans):
            non_overlapping.append(entity)
            used_spans.add(span)
    
    # Sort by token span for consistent output
    return sorted(non_overlapping, key=lambda x: (x[0], x[1]))

def split_text_to_sentences(text):
    """Split text into sentences using NLTK.
    
    Args:
        text: Text string to split
        
    Returns:
        List of sentence strings
    """
    try:
        return sent_tokenize(text)
    except Exception as e:
        print(f"NLTK sentence tokenization failed: {e}")
        print("Using fallback regex sentence splitter")
        # Simple fallback: split on typical sentence endings
        import re
        # Split on periods, question marks, exclamation points followed by space and uppercase letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Further split any very long sentences that might still be too long
        result = []
        for sentence in sentences:
            # If sentence is very long, split on commas or semicolons
            if len(sentence) > 300:
                subsents = re.split(r'(?<=[,;])\s+', sentence)
                result.extend(subsents)
            else:
                result.append(sentence)
        
        return result

def get_token_to_char_map(text, tokens):
    """Create a mapping from token indices to character positions.
    
    Args:
        text: Original text string
        tokens: List of tokens
        
    Returns:
        List of (start_char, end_char) positions for each token
    """
    token_char_map = []
    char_idx = 0
    
    for token in tokens:
        # Skip whitespace
        while char_idx < len(text) and text[char_idx].isspace():
            char_idx += 1
            
        # Find token in text
        token_start = char_idx
        
        # Handle special case where token might not be exact match in text
        # (e.g., tokenization quirks with punctuation)
        if char_idx + len(token) <= len(text) and text[char_idx:char_idx + len(token)].lower() == token.lower():
            char_idx += len(token)
        else:
            # Search for token
            found = False
            for i in range(char_idx, len(text)):
                if i + len(token) <= len(text) and text[i:i + len(token)].lower() == token.lower():
                    token_start = i
                    char_idx = i + len(token)
                    found = True
                    break
            if not found:
                # If we can't find the token, just use the current position and advance
                char_idx += len(token)
        
        token_char_map.append((token_start, char_idx))
    
    return token_char_map

def split_long_documents_by_sentences_with_model_tokenizer(data, model_tokenizer, max_length=512, overlap_sentences=1):
    """Split long documents into chunks by sentence boundaries using the model's tokenizer.
    
    Args:
        data: List of examples with "tokenized_text" and "ner"
        model_tokenizer: The actual model tokenizer to use for token counting
        max_length: Maximum sequence length (default: 512)
        overlap_sentences: Number of sentences to overlap between chunks (default: 1)
        
    Returns:
        List of processed examples with sequences <= max_length
    """
    processed_data = []
    long_docs_count = 0
    chunks_created = 0
    entities_preserved = 0
    total_entities = 0
    oversized_chunks = 0
    token_chunked_sentences = 0
    
    for example in tqdm(data, desc="Splitting long documents with model tokenizer"):
        tokens = example["tokenized_text"]
        entities = example["ner"]
        total_entities += len(entities)
        
        # First check if the document is actually too long using the model tokenizer
        full_text = " ".join(tokens)
        model_tokens = model_tokenizer.encode(full_text)
        
        if len(model_tokens) <= max_length:
            # Document is short enough for the model, keep as is
            processed_data.append(example)
            entities_preserved += len(entities)
            continue
            
        long_docs_count += 1
        
        # Reconstruct the original text and split into sentences
        sentences = split_text_to_sentences(full_text)
        
        # Get token counts for each sentence using the model tokenizer
        # Subtract 2 for special tokens ([CLS] and [SEP])
        sentence_token_counts = [len(model_tokenizer.encode(sentence)) - 2 for sentence in sentences]
        
        # Check if any single sentence is too long
        long_sentences = [i for i, count in enumerate(sentence_token_counts) if count > max_length - 2]
        if long_sentences:
            print(f"Warning: {len(long_sentences)} sentences are too long individually. "
                 f"These will be split using token-based chunking.")
            token_chunked_sentences += len(long_sentences)
        
        # Group sentences into chunks that fit within max_length
        chunks = []
        current_chunk_indices = []
        current_token_count = 0
        
        i = 0
        while i < len(sentences):
            token_count = sentence_token_counts[i]
            
            # Check if this single sentence is too long for max_length
            if token_count > max_length - 2:
                # If we have accumulated sentences, finalize the current chunk
                if current_chunk_indices:
                    chunks.append(current_chunk_indices.copy())
                    current_chunk_indices = []
                    current_token_count = 0
                
                # Apply token-based chunking to this long sentence
                long_sentence = sentences[i]
                long_sentence_token_chunks = token_chunk_sentence(long_sentence, model_tokenizer, max_length)
                
                # Get the character offsets for this sentence in the original text
                sentence_start = full_text.find(long_sentence)
                
                for chunk_text in long_sentence_token_chunks:
                    # Process each token chunk as a separate "sentence chunk"
                    chunk_tokens = tokenize_text(chunk_text)
                    
                    # Map entities to this chunk if they fall within its character range
                    chunk_start_char = sentence_start + long_sentence.find(chunk_text)
                    chunk_end_char = chunk_start_char + len(chunk_text)
                    
                    # Get token-to-char map for original text
                    original_token_char_map = get_token_to_char_map(full_text, tokens)
                    
                    # Find entities that belong in this chunk
                    chunk_entities = []
                    for entity in entities:
                        ent_start, ent_end, ent_type = entity
                        
                        # Get character positions of this entity in original text
                        if ent_start < len(original_token_char_map) and ent_end < len(original_token_char_map):
                            entity_start_char = original_token_char_map[ent_start][0]
                            entity_end_char = original_token_char_map[ent_end][1]
                            
                            # Check if entity is fully within this chunk
                            if entity_start_char >= chunk_start_char and entity_end_char <= chunk_end_char:
                                # Find entity's position in the chunk tokens
                                entity_text = full_text[entity_start_char:entity_end_char]
                                chunk_entity_spans = find_entity_spans(chunk_tokens, entity_text)
                                
                                for span_start, span_end in chunk_entity_spans:
                                    chunk_entities.append([span_start, span_end, ent_type])
                                    entities_preserved += 1
                    
                    # Add this chunk if it contains entities
                    if chunk_entities:
                        processed_data.append({
                            "tokenized_text": chunk_tokens,
                            "ner": chunk_entities
                        })
                        chunks_created += 1
                
                i += 1  # Move to next sentence
                continue
            
            # If adding this sentence would exceed max_length, finalize current chunk
            if current_token_count + token_count > max_length - 2 and current_chunk_indices:
                # Add current chunk to list of chunks
                chunks.append(current_chunk_indices.copy())
                
                # Start new chunk with overlap - using configurable sentence overlap
                overlap_start = max(0, len(current_chunk_indices) - overlap_sentences)
                current_chunk_indices = current_chunk_indices[overlap_start:]
                current_token_count = sum(sentence_token_counts[j] for j in current_chunk_indices)
            
            # Add current sentence to chunk
            current_chunk_indices.append(i)
            current_token_count += token_count
            i += 1
        
        # Add the last chunk if not empty
        if current_chunk_indices:
            chunks.append(current_chunk_indices)
        
        # Process each chunk
        for chunk_indices in chunks:
            # Get text for this chunk
            chunk_text = " ".join(sentences[idx] for idx in chunk_indices)
            
            # Verify chunk length with model tokenizer
            chunk_tokens_count = len(model_tokenizer.encode(chunk_text))
            if chunk_tokens_count > max_length:
                # This chunk is still too long - this could happen if our token count estimation was off
                oversized_chunks += 1
                print(f"Warning: Generated chunk has {chunk_tokens_count} tokens, exceeding max length of {max_length}")
                
                # Try to reduce chunk size by removing sentences until it fits
                while len(chunk_indices) > 1 and chunk_tokens_count > max_length:
                    # Remove the last sentence
                    chunk_indices.pop()
                    chunk_text = " ".join(sentences[idx] for idx in chunk_indices)
                    chunk_tokens_count = len(model_tokenizer.encode(chunk_text))
                
                if chunk_tokens_count > max_length:
                    # Still too long with just one sentence, apply token-based chunking
                    print(f"Warning: Even after reduction, chunk has {chunk_tokens_count} tokens. Applying token-based chunking.")
                    continue  # Skip this chunk - it will be handled by token-based chunking
            
            # Tokenize using our standard tokenizer for consistency with entity spans
            chunk_tokens = tokenize_text(chunk_text)
            
            # Map entities to the chunk
            # First, construct a token map from original text to chunk text
            original_to_chunk_map = {}
            
            # Tokenize original text and chunk text
            original_text = full_text
            
            # Get character positions for each sentence
            char_positions = []
            pos = 0
            for sentence in sentences:
                start = original_text.find(sentence, pos)
                end = start + len(sentence)
                char_positions.append((start, end))
                pos = end
            
            # Get character range for this chunk
            chunk_start_char = char_positions[chunk_indices[0]][0]
            chunk_end_char = char_positions[chunk_indices[-1]][1]
            
            # Get token-to-char map for original text
            original_token_char_map = get_token_to_char_map(original_text, tokens)
            
            # Map original tokens to chunk tokens
            chunk_entities = []
            for entity in entities:
                ent_start, ent_end, ent_type = entity
                
                # Get character positions of this entity in original text
                if ent_start < len(original_token_char_map) and ent_end < len(original_token_char_map):
                    entity_start_char = original_token_char_map[ent_start][0]
                    entity_end_char = original_token_char_map[ent_end][1]
                    
                    # Check if entity is within this chunk
                    if entity_start_char >= chunk_start_char and entity_end_char <= chunk_end_char:
                        # Entity is in this chunk, adjust positions
                        entity_text = original_text[entity_start_char:entity_end_char]
                        
                        # Find this entity in the chunk tokens
                        chunk_entity_spans = find_entity_spans(chunk_tokens, entity_text)
                        
                        for span_start, span_end in chunk_entity_spans:
                            chunk_entities.append([span_start, span_end, ent_type])
                            entities_preserved += 1
            
            # Create entry for this chunk
            if chunk_entities:  # Only add chunks with entities to save space
                processed_data.append({
                    "tokenized_text": chunk_tokens,
                    "ner": chunk_entities
                })
                chunks_created += 1
    
    print(f"Split {long_docs_count} long documents into {chunks_created} chunks using sentence boundaries")
    print(f"Preserved {entities_preserved} out of {total_entities} entities ({entities_preserved/total_entities*100:.2f}%)")
    if oversized_chunks > 0:
        print(f"Warning: {oversized_chunks} chunks were initially over the maximum length and had to be adjusted")
    if token_chunked_sentences > 0:
        print(f"Applied token-based chunking to {token_chunked_sentences} sentences that exceeded maximum length")
    
    return processed_data

def token_chunk_sentence(sentence, model_tokenizer, max_length):
    """Split an individual sentence into chunks based on token count when sentence is too long.
    
    Args:
        sentence: The sentence to split
        model_tokenizer: The model tokenizer to use for token counting
        max_length: Maximum sequence length
        
    Returns:
        List of sentence chunks that fit within max_length
    """
    # Get the tokenized representation of the sentence
    encoded = model_tokenizer.encode(sentence)
    
    # Subtract 2 for [CLS] and [SEP] tokens
    if len(encoded) <= max_length:
        return [sentence]
    
    # Split into chunks of max_length-2 tokens
    effective_max_length = max_length - 2
    token_chunks = []
    
    for i in range(0, len(encoded), effective_max_length):
        chunk_tokens = encoded[i:i+effective_max_length]
        
        # To convert back to text, we need to:
        # 1. Decode the chunk tokens
        # 2. Find where this text is in the original sentence
        
        # Skip [CLS] and [SEP] if they exist in the tokenization
        if i == 0 and encoded[0] == model_tokenizer.cls_token_id:
            chunk_tokens = chunk_tokens[1:]
        if i + effective_max_length >= len(encoded) and encoded[-1] == model_tokenizer.sep_token_id:
            chunk_tokens = chunk_tokens[:-1]
        
        chunk_text = model_tokenizer.decode(chunk_tokens)
        
        # Find this chunk in the original sentence
        # For some tokenizers, the decoded text might not exactly match the original due to
        # special characters, spacing, etc. So we find the best approximation
        
        if chunk_text not in sentence:
            # Try to find a clean break point - preferably at word boundaries
            words = chunk_text.split()
            if len(words) > 1:
                # Try with one fewer word
                shorter_text = " ".join(words[:-1])
                if shorter_text in sentence:
                    chunk_text = shorter_text
        
        token_chunks.append(chunk_text)
    
    # Check resulting chunks - merge very small ones if needed
    merged_chunks = []
    current_chunk = ""
    
    for chunk in token_chunks:
        if not current_chunk:
            current_chunk = chunk
        elif len(model_tokenizer.encode(current_chunk + " " + chunk)) - 2 <= effective_max_length:
            current_chunk += " " + chunk
        else:
            merged_chunks.append(current_chunk)
            current_chunk = chunk
    
    if current_chunk:
        merged_chunks.append(current_chunk)
    
    return merged_chunks

def convert_to_gliner_format(input_file, output_file, sample_size=None, max_seq_length=512, overlap_sentences=1):
    """Convert the merged JSON to GLiNER format.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the output JSON file
        sample_size: Optional number of entries to process for testing
        max_seq_length: Maximum sequence length for document splitting
        overlap_sentences: Number of sentences to overlap between chunks (default: 1)
    """
    # Load the bioformer tokenizer for accurate token counting
    print("Loading bioformer tokenizer...")
    try:
        model_tokenizer = AutoTokenizer.from_pretrained("bioformers/bioformer-16L")
        print("Bioformer tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading bioformer tokenizer: {e}")
        print("Falling back to BERT tokenizer")
        model_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load the data
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Optionally use a smaller sample for testing
    if sample_size and sample_size < len(data):
        data = data[:sample_size]
        print(f"Using sample of {sample_size} entries")
    
    print(f"Processing {len(data)} entries...")
    gliner_data = []
    entity_count = 0
    matched_entity_count = 0
    total_entity_mentions = 0  # Total entity mentions found
    overlap_count = 0
    entries_with_entities = 0
    
    # Convert each entry
    for entry in tqdm(data, desc="Converting data format"):
        text = entry['text']
        tokenized_text = tokenize_text(text)
        entities = entry['entities']
        entity_count += len(entities)
        
        # Find token spans for each entity
        all_spans = []
        for entity_info in entities:
            entity_text = entity_info['entity']
            entity_type = entity_info['type']
            
            # Find all occurrences of this entity
            spans = find_entity_spans(tokenized_text, entity_text)
            
            # Track how many unique entities were found at least once
            if spans:
                matched_entity_count += 1
            
            # Add all spans to NER list
            for start_idx, end_idx in spans:
                all_spans.append([start_idx, end_idx, entity_type])
                total_entity_mentions += 1
        
        # Remove overlapping entities
        original_count = len(all_spans)
        non_overlapping_spans = remove_overlapping_entities(all_spans)
        overlap_count += original_count - len(non_overlapping_spans)
        
        # Create GLiNER format entry
        gliner_entry = {
            "tokenized_text": tokenized_text,
            "ner": non_overlapping_spans
        }
        
        # Only add entries that have at least one entity
        if len(non_overlapping_spans) > 0:
            gliner_data.append(gliner_entry)
            entries_with_entities += 1
    
    print(f"Converted {entries_with_entities} entries with entities")
    print(f"Found {matched_entity_count} unique entities out of {entity_count} ({matched_entity_count/entity_count*100:.2f}%)")
    print(f"Total entity mentions found: {total_entity_mentions}")
    print(f"Removed {overlap_count} overlapping entities")
    print(f"Final entity mentions after removing overlaps: {total_entity_mentions - overlap_count}")
    
    # Split long documents to ensure they fit within model's context window
    print("\nChecking for documents exceeding max sequence length...")
    
    # Check if documents are too long using model tokenizer
    # This is more accurate than using len(tokens)
    long_docs = 0
    for entry in tqdm(gliner_data, desc="Checking document lengths"):
        text = " ".join(entry["tokenized_text"])
        tokens = model_tokenizer.encode(text)
        if len(tokens) > max_seq_length:
            long_docs += 1
    
    if long_docs > 0:
        print(f"Found {long_docs} documents exceeding model's max length of {max_seq_length}")
        print(f"Splitting documents with sentence-based chunking using bioformer tokenizer...")
        gliner_data = split_long_documents_by_sentences_with_model_tokenizer(
            gliner_data, model_tokenizer, max_length=max_seq_length, overlap_sentences=overlap_sentences
        )
    else:
        print("No documents exceed max sequence length. No splitting needed.")
    
    # Save the results
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gliner_data, f)
    
    # Save a small sample for inspection
    sample_output_file = output_file.replace('.json', '_sample.json')
    with open(sample_output_file, 'w', encoding='utf-8') as f:
        json.dump(gliner_data[:10], f, indent=2)
    print(f"Saved sample of 10 entries to {sample_output_file}")
    
    print("Done!")

if __name__ == "__main__":
    input_file = "/Users/rajlq7/Desktop/GLiNER/final_merged.json"
    output_file = "/Users/rajlq7/Desktop/GLiNER/gliner_training_data.json"
    
    # Run on the full dataset with sentence-based document splitting
    # Using 1 sentence overlap between chunks and bioformer tokenizer
    convert_to_gliner_format(input_file, output_file, max_seq_length=512, overlap_sentences=1) 