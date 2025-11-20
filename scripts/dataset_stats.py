import json
import glob
from collections import defaultdict

def main():
    # compute stats for all languages with number of all samples, avg. text length, avg. types/sample, avg. unique types sample
    language_stats = defaultdict(lambda: {
        "num_samples": 0,
        "text_lengths": [],
        "num_types_per_sample": [],
        "num_unique_types_per_sample": []
    })
    
    for file in glob.glob("/vol/tmp/goldejon/ner/data/finerweb_splitted/*.jsonl"):
        language = file.split("/")[-1].split(".")[0]
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                text = sample["text"]
                char_spans = sample["char_spans"]
                
                # Count number of spans (types) in this sample
                num_spans = len(char_spans)
                
                # Count unique types in this sample
                unique_types = set(span["label"] for span in char_spans)
                num_unique_types = len(unique_types)
                
                # Accumulate stats
                language_stats[language]["num_samples"] += 1
                language_stats[language]["text_lengths"].append(len(text))
                language_stats[language]["num_types_per_sample"].append(num_spans)
                language_stats[language]["num_unique_types_per_sample"].append(num_unique_types)
    
    # Compute and print statistics
    print(f"{'Language':<20} {'Samples':<12} {'Avg Text Length':<18} {'Avg Types/Sample':<18} {'Avg Unique Types/Sample':<25}")
    print("-" * 95)
    
    for language in sorted(language_stats.keys()):
        stats = language_stats[language]
        num_samples = stats["num_samples"]
        avg_text_length = sum(stats["text_lengths"]) / num_samples if num_samples > 0 else 0
        avg_types_per_sample = sum(stats["num_types_per_sample"]) / num_samples if num_samples > 0 else 0
        avg_unique_types_per_sample = sum(stats["num_unique_types_per_sample"]) / num_samples if num_samples > 0 else 0
        
        print(f"{language:<20} {num_samples:<12} {avg_text_length:<18.2f} {avg_types_per_sample:<18.2f} {avg_unique_types_per_sample:<25.2f}")
    
    # Print summary
    # Compute global averages across all samples
    all_text_lengths = []
    all_types_per_sample = []
    all_unique_types_per_sample = []
    total_samples = 0

    for stats in language_stats.values():
        total_samples += stats["num_samples"]
        all_text_lengths.extend(stats["text_lengths"])
        all_types_per_sample.extend(stats["num_types_per_sample"])
        all_unique_types_per_sample.extend(stats["num_unique_types_per_sample"])

    avg_text_length = sum(all_text_lengths) / total_samples if total_samples > 0 else 0
    avg_types_per_sample = sum(all_types_per_sample) / total_samples if total_samples > 0 else 0
    avg_unique_types_per_sample = sum(all_unique_types_per_sample) / total_samples if total_samples > 0 else 0

    print("-" * 95)
    print(f"{'TOTAL':<20} {total_samples:<12} {avg_text_length:<18.2f} {avg_types_per_sample:<18.2f} {avg_unique_types_per_sample:<25.2f}")

def score_distribution():
    from datasets import Dataset
    from collections import Counter
    dataset = Dataset.load_from_disk("/vol/tmp/goldejon/multilingual_ner/data/preference/preference_dataset_4o")
    all_scores = []
    for sample in dataset:
        all_scores.append(sample["score"])
    
    # distribution in percentage
    distribution = Counter(all_scores)
    total = sum(distribution.values())
    for score, count in distribution.items():
        print(f"{score}: {count / total * 100:.2f}%")

if __name__ == "__main__":
    score_distribution()