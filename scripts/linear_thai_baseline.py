from transformers import AutoTokenizer, AutoModelForTokenClassification, MT5ForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset
import evaluate
import numpy as np
import torch

def main():
    # Set seeds
    torch.manual_seed(42)
    model_name = "google/rembert"
    
    # Load dataset and tokenizer
    dataset = load_dataset('pythainlp/thainer-corpus-v2.2', revision='convert/parquet')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_column = "words"
    label_column = "ner"
    
    # Filter empty samples
    def is_not_empty(example):
        return len([w for w in example[text_column] if w.strip()]) > 0
    
    for split in dataset.keys():
        dataset[split] = dataset[split].filter(is_not_empty)
    
    # Get label names first to create proper B->I mapping
    label_names = dataset['train'].features[label_column].feature.names
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    
    # Create B-XXX -> I-XXX mapping
    b_to_i_map = {}
    for label_id, label_name in id2label.items():
        if label_name.startswith('B-'):
            i_label_name = 'I-' + label_name[2:]
            if i_label_name in label2id:
                b_to_i_map[label_id] = label2id[i_label_name]
            else:
                # If I-XXX doesn't exist, keep B-XXX
                b_to_i_map[label_id] = label_id
    
    # Align labels with tokens
    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                new_labels.append(-100)
            else:
                label = labels[word_id]
                # Convert B-XXX to I-XXX if it's a continuation token
                if label in b_to_i_map:
                    label = b_to_i_map[label]
                new_labels.append(label)
        return new_labels
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column], truncation=True, is_split_into_words=True
        )
        all_labels = examples[label_column]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))
        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
    
    # Tokenize dataset
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # Setup model
    num_labels = len(label_names)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
    )
    
    # Setup data collator and metric
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    metric = evaluate.load("seqeval")
    
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        
        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }
    
    # Setup training arguments
    args = TrainingArguments(
        "bert-finetuned-ner",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=5e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        seed=42,
        logging_steps=50,
        lr_scheduler_type="linear",
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
