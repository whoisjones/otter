import json
from datasets import load_dataset

def tokens_to_char_spans(tokens):
    """
    Convert tokens to character offsets assuming space-separated tokens.
    Returns a list of (start_char, end_char) for each token.
    """
    char_spans = []
    current_pos = 0
    
    for token in tokens:
        start = current_pos
        end = current_pos + len(token)
        char_spans.append((start, end))
        current_pos = end + 1  # +1 for space
    
    return char_spans

def transform_bio_to_spans(tokens, ner_tags, id2label):
    """
    Convert BIO format annotations to span-based format with exclusive end indices.
    
    Returns:
        token_spans: List of dicts with keys: start, end (exclusive), label
        char_spans: List of dicts with keys: start, end (exclusive), label
    """
    token_spans = []
    char_spans = []
    
    # Get character positions for each token
    token_char_positions = tokens_to_char_spans(tokens)
    
    start_idx = None
    ent_type = None
    
    for idx, tag_id in enumerate(ner_tags):
        tag = id2label[tag_id]
        if tag == 'O':
            if start_idx is not None:
                # Close previous entity (exclusive end)
                end_idx = idx
                token_spans.append({
                    "start": start_idx,
                    "end": end_idx,
                    "label": ent_type
                })
                
                # Character span (exclusive end)
                char_start = token_char_positions[start_idx][0]
                char_end = token_char_positions[idx - 1][1]
                char_spans.append({
                    "start": char_start,
                    "end": char_end,
                    "label": ent_type
                })
                
                start_idx = None
                ent_type = None
        elif tag.startswith('B-'):
            if start_idx is not None:
                # Close previous entity before starting a new one (exclusive end)
                end_idx = idx
                token_spans.append({
                    "start": start_idx,
                    "end": end_idx,
                    "label": ent_type
                })
                
                char_start = token_char_positions[start_idx][0]
                char_end = token_char_positions[idx - 1][1]
                char_spans.append({
                    "start": char_start,
                    "end": char_end,
                    "label": ent_type
                })
            
            start_idx = idx
            ent_type = tag[2:]  # Remove 'B-'
        elif tag.startswith('I-'):
            if start_idx is None:
                # I- without B- before, treat as B-
                start_idx = idx
                ent_type = tag[2:]
            # else: continue the current entity
    
    # Handle last span if exists (exclusive end)
    if start_idx is not None:
        end_idx = len(ner_tags)
        token_spans.append({
            "start": start_idx,
            "end": end_idx,
            "label": ent_type
        })
        
        char_start = token_char_positions[start_idx][0]
        char_end = token_char_positions[-1][1]
        char_spans.append({
            "start": char_start,
            "end": char_end,
            "label": ent_type
        })
    
    return token_spans, char_spans

def transform_dataset(input_file, output_file, id2label):
    """
    Transform test.jsonl to include tokens, text, char spans, and token spans.
    """
    transformed_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            sample = json.loads(line)
            
            # Extract tokens and ner_tags
            tokens = sample['tokens']
            ner_tags = sample['ner_tags']
            
            # Create text string (space-separated tokens)
            text = ' '.join(tokens)
            
            # Convert BIO tags to spans
            token_spans, char_spans = transform_bio_to_spans(tokens, ner_tags, id2label)
            
            # Create transformed sample
            transformed_sample = {
                'id': sample.get('id', ''),
                'tokens': tokens,
                'text': text,
                'token_spans': token_spans,  # [{"start": ..., "end": ..., "label": ...}, ...]
                'char_spans': char_spans      # [{"start": ..., "end": ..., "label": ...}, ...]
            }
            
            transformed_data.append(transformed_sample)
    
    # Write to output file
    with open(output_file, 'w') as f:
        for sample in transformed_data:
            f.write(json.dumps(sample) + '\n')
    
if __name__ == '__main__':
    dataset = load_dataset('conll2003', revision='convert/parquet')
    dataset['train'].to_json('/vol/tmp/goldejon/ner/data/conll2003/train.jsonl')
    dataset['validation'].to_json('/vol/tmp/goldejon/ner/data/conll2003/validation.jsonl')
    dataset['test'].to_json('/vol/tmp/goldejon/ner/data/conll2003/test.jsonl')
    id2label = {i: label for i, label in enumerate(dataset['train'].features["ner_tags"].feature.names)}
    transform_dataset('/vol/tmp/goldejon/ner/data/conll2003/test.jsonl', '/vol/tmp/goldejon/ner/data/conll2003/test.jsonl', id2label)
    transform_dataset('/vol/tmp/goldejon/ner/data/conll2003/validation.jsonl', '/vol/tmp/goldejon/ner/data/conll2003/validation.jsonl', id2label)
    transform_dataset('/vol/tmp/goldejon/ner/data/conll2003/train.jsonl', '/vol/tmp/goldejon/ner/data/conll2003/train.jsonl', id2label)