import torch
from tqdm import tqdm
from pathlib import Path
from itertools import cycle
from collections import defaultdict
import shutil

from .metrics import compute_span_predictions, compute_compressed_span_predictions, add_batch_metrics, finalize_metrics
from .logger import setup_logger


def evaluate(model, dataloader, accelerator):
    """Evaluate the model on a dataset."""
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)

    total_loss = 0.0
    num_batches = 0
    
    metrics_by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            output = model(
                token_input_ids=batch["token_input_ids"],
                token_attention_mask=batch["token_attention_mask"],
                token_token_type_ids=batch.get("token_token_type_ids"),
                type_input_ids=batch["type_input_ids"],
                type_attention_mask=batch["type_attention_mask"],
                type_token_type_ids=batch.get("type_token_type_ids"),
                labels=batch["labels"]
            )

            if output.loss is not None:
                loss = output.loss
                total_loss += loss.item()
                num_batches += 1

            golds = batch['labels']['ner']
            if len(output.span_logits.shape) == 4:
                predictions = compute_span_predictions(
                    span_logits=output.span_logits,
                    start_mask=batch["labels"]["valid_start_mask"],
                    end_mask=batch["labels"]["valid_end_mask"],
                    max_span_width=unwrapped_model.config.max_span_length,
                    id2label=batch["id2label"],
                    threshold=unwrapped_model.config.prediction_threshold
                )
            else:
                predictions = compute_compressed_span_predictions(
                    span_logits=output.span_logits,
                    span_mask=batch["labels"]["valid_span_mask"],
                    span_mapping=batch["labels"]["span_subword_indices"],
                    id2label=batch["id2label"],
                    threshold=unwrapped_model.config.prediction_threshold
                )
            add_batch_metrics(golds, predictions, metrics_by_type)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    metrics = finalize_metrics(metrics_by_type)
    metrics["loss"] = avg_loss
    
    return metrics


def train(model, train_dataloader, eval_dataloader, optimizer, scheduler, accelerator, args):
    logger = setup_logger(args.output_dir)
    model.train()
    total_loss = 0.0
    num_batches = 0
    global_step = 0
    best_f1 = 0.0
    
    save_total_limit = getattr(args, 'save_total_limit', 2)
    best_models = []
    best_checkpoint_path = None
    
    # Early stopping parameters
    patience = args.early_stopping_patience
    patience_counter = 0
    
    train_iterator = cycle(train_dataloader)
    
    progress_bar = tqdm(total=args.max_steps, desc="Training", disable=not accelerator.is_local_main_process)
    
    while global_step < args.max_steps:
        batch = next(train_iterator)
        if not batch:
            continue
        
        optimizer.zero_grad()
        
        output = model(
            token_input_ids=batch["token_input_ids"],
            token_attention_mask=batch["token_attention_mask"],
            token_token_type_ids=batch.get("token_token_type_ids"),
            type_input_ids=batch["type_input_ids"],
            type_attention_mask=batch["type_attention_mask"],
            type_token_type_ids=batch.get("type_token_type_ids"),
            labels=batch["labels"]
        )
        loss = output.loss

        # Backward pass with accelerator (handles mixed precision automatically)
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1
        
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss/num_batches:.4f}"
        })
        
        # Evaluate and save checkpoint every eval_steps
        if global_step % args.eval_steps == 0:
            logger.info(f"\n{'='*50}")
            logger.info(f"Step {global_step}/{args.max_steps}")
            logger.info(f"Training Loss: {total_loss/num_batches:.4f}")
            
            # Evaluate
            if eval_dataloader is not None:
                eval_metrics = evaluate(model, eval_dataloader, accelerator)
            else:
                eval_metrics = {"loss": 0.0, "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0}}
            logger.info(f"Evaluation Loss: {eval_metrics['loss']:.4f}")
            logger.info(f"Precision: {eval_metrics['micro']['precision']:.4f}")
            logger.info(f"Recall: {eval_metrics['micro']['recall']:.4f}")
            logger.info(f"F1 Score: {eval_metrics['micro']['f1']:.4f}")
            
            # Save checkpoint (unwrap model if needed)
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(str(checkpoint_dir))
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # Track checkpoints with their F1 scores
            current_f1 = eval_metrics['micro']['f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_checkpoint_path = checkpoint_dir
                patience_counter = 0
                logger.info(f"New best F1: {best_f1:.4f} at checkpoint {checkpoint_dir}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
            best_models.append((current_f1, checkpoint_dir, global_step))
            best_models.sort(key=lambda x: x[0], reverse=True)
            
            if len(best_models) > save_total_limit:
                _, worst_checkpoint_path, _ = best_models[save_total_limit]
                shutil.rmtree(worst_checkpoint_path)
            
            best_models = best_models[:save_total_limit]
            
            logger.info(f"{'='*50}\n")
            
            # Early stopping check
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} evaluations without improvement.")
                logger.info(f"Best F1 score: {best_f1:.4f}")
                break
            
            # Reset training metrics
            total_loss = 0.0
            num_batches = 0
            model.train()
    
    progress_bar.close()
    
    # Save best checkpoint info and create a "best" checkpoint link
    if best_checkpoint_path is not None and accelerator.is_main_process:
        best_checkpoint_link = Path(args.output_dir) / "best_checkpoint"
        if best_checkpoint_link.exists():
            if best_checkpoint_link.is_symlink():
                best_checkpoint_link.unlink()
            elif best_checkpoint_link.is_dir():
                shutil.rmtree(best_checkpoint_link)
            else:
                best_checkpoint_link.unlink()
        # Create symlink using relative path
        best_checkpoint_link.symlink_to(best_checkpoint_path.relative_to(Path(args.output_dir)))
        logger.info(f"Best checkpoint (F1={best_f1:.4f}) saved at: {best_checkpoint_path}")
        logger.info(f"Symlink created: {best_checkpoint_link} -> {best_checkpoint_path.relative_to(Path(args.output_dir))}")
    
    return global_step, best_checkpoint_path, best_f1

