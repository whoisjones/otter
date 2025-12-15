import glob
import json
from collections import defaultdict
from pathlib import Path
import iso639
from tqdm import tqdm
from wtpsplit import SaT

sat_languages = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kirghiz",
    "la": "Latin",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "ne": "Nepalese",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Panjabi",
    "pl": "Polish",
    "ps": "Pushto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sq": "Albanian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zh": "Chinese",
    "zu": "Zulu",
}


if __name__ == "__main__":
    sat = SaT("sat-12l-sm")
    sat.half().to("cuda")
    
    def remap_char_spans(char_spans, char_start, char_end):
        """Remap char_spans to new character indices for a sentence."""
        remapped = []
        dropped = 0
        for span in char_spans:
            span_start = int(span["start"])
            span_end = int(span["end"])
            # Check if span is fully within sentence boundaries
            if span_start >= char_start and span_end <= char_end:
                remapped.append({
                    "start": span_start - char_start,
                    "end": span_end - char_start,
                    "label": span["tag"],
                })
            elif span_start < char_end and span_end > char_start:
                # Span crosses sentence boundary, drop it
                dropped += 1
        return remapped, dropped
    
    for file in glob.glob("/vol/tmp2/goldejon/multilingual_ner/data/singlelabel/finerweb_merged_jsonl_translated/*.jsonl"):
        language = file.split("/")[-1].split(".")[0]
        language_code = iso639.Language.from_part3(language).part1
        # Determine output path in finerweb_splitted directory
        output_dir = Path('/vol/tmp/goldejon/ner/data/finerweb_translated_splitted')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / file.split("/")[-1]
        if output_path.exists():
            continue
        
        stats = defaultdict(int)
        annotated_sentences = []
        
        with open(file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(tqdm(f, desc=f"Processing {Path(file).name}")):
                sample = json.loads(line)
                text = sample.get("text", "")
                char_spans = sample.get("char_spans", [])
                
                if not text:
                    stats["samples_without_text"] += 1
                    continue
                
                # Split text into sentences using SAT
                try:
                    sentence_splits = sat.split(text)
                except Exception as e:
                    stats["sentence_split_errors"] += 1
                    continue
                
                if not sentence_splits:
                    stats["samples_without_sentences"] += 1
                    continue
                
                sample_sentences = []
                text_pos = 0
                
                for sent_idx, sent_info in enumerate(sentence_splits):
                    # Handle different return formats from SAT
                    if isinstance(sent_info, tuple) and len(sent_info) == 3:
                        sent_text, sent_start_char, sent_end_char = sent_info
                    elif isinstance(sent_info, str):
                        # If only text is returned, find it in original text
                        sent_text = sent_info.strip()
                        sent_start_char = text.find(sent_text, text_pos)
                        if sent_start_char == -1:
                            # Try without strip
                            sent_start_char = text.find(sent_info, text_pos)
                        if sent_start_char == -1:
                            # Fallback: use current position
                            sent_start_char = text_pos
                        sent_end_char = sent_start_char + len(sent_text)
                    else:
                        continue
                    
                    if not sent_text:
                        continue
                    
                    # Remap char_spans
                    remapped_char_spans, dropped_char = remap_char_spans(
                        char_spans, sent_start_char, sent_end_char
                    )
                    
                    stats["spans_dropped"] += dropped_char
                    
                    # Only keep sentences with annotations - skip if no annotations
                    if not remapped_char_spans:
                        stats["sentences_without_annotations"] += 1
                        continue
                    
                    new_sample = {
                        k: v for k, v in sample.items() 
                        if k not in {"tokens", "text", "token_spans", "char_spans"}
                    }
                    new_sample.update({
                        "id": f"{sample.get('id', idx)}-s{sent_idx}",
                        "text": sent_text,
                        "char_spans": remapped_char_spans,
                    })
                    
                    sample_sentences.append(new_sample)
                    stats["sentences_with_annotations"] += 1
                    text_pos = sent_end_char
                
                if not sample_sentences:
                    stats["samples_without_sentences"] += 1
                    continue
                
                stats["input_samples"] += 1
                annotated_sentences.extend(sample_sentences)
        
        # Write output
        with output_path.open("w", encoding="utf-8") as out_f:
            for sample in annotated_sentences:
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        stats["output_sentences"] = len(annotated_sentences)
        
        print(f"\nCompleted processing {Path(file).name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")