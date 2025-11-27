import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import shutil
from .metrics import compute_span_predictions, compute_compressed_span_predictions, add_batch_metrics, finalize_metrics
from .logger import setup_logger


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def evaluate(model, dataloader, accelerator):
    """Evaluate the model on a dataset."""
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)

    total_loss = 0.0
    num_batches = 0
    
    metrics_by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader), desc="Evaluating", disable=not accelerator.is_local_main_process)
        for batch in dataloader:
            output = model(
                token_encoder_inputs=batch["token_encoder_inputs"],
                type_encoder_inputs=batch["type_encoder_inputs"],
                labels=batch["labels"]
            )

            if output.loss is not None:
                loss = output.loss
                total_loss += loss.detach().item()
                num_batches += 1

            golds = batch['labels']['ner']
            if len(output.span_logits.shape) == 4:
                predictions = compute_span_predictions(
                    span_logits=output.span_logits.detach(),
                    start_mask=batch["labels"]["valid_start_mask"],
                    end_mask=batch["labels"]["valid_end_mask"],
                    max_span_width=unwrapped_model.config.max_span_length,
                    id2label=batch["id2label"],
                    threshold=unwrapped_model.config.prediction_threshold
                )
            else:
                predictions = compute_compressed_span_predictions(
                    span_logits=output.span_logits.detach(),
                    span_mask=batch["labels"]["valid_span_mask"],
                    span_mapping=batch["labels"]["span_subword_indices"],
                    id2label=batch["id2label"],
                    threshold=unwrapped_model.config.prediction_threshold
                )
            add_batch_metrics(golds, predictions, metrics_by_type)
            
            pbar.update(1)
        pbar.close()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    metrics = finalize_metrics(metrics_by_type)
    metrics["loss"] = avg_loss

    torch.cuda.empty_cache()
    
    return metrics


def train(model, train_dataloader, eval_dataloader, optimizer, scheduler, accelerator, args):
    logger = setup_logger(args.output_dir, is_main_process=accelerator.is_main_process)
    model.train()
    total_loss = 0.0
    num_batches = 0
    global_step = 0
    best_f1 = 0.0
    
    save_total_limit = getattr(args, 'save_total_limit', 2)
    best_models = []
    best_checkpoint_path = None
    
    patience = args.early_stopping_patience
    patience_counter = 0
    
    train_iterator = cycle(train_dataloader)
    
    steps_per_epoch = len(train_dataloader)
    logging_steps = getattr(args, 'logging_steps', 10)
    
    progress_bar = tqdm(total=args.max_steps, desc="Training", disable=not accelerator.is_local_main_process)
    
    while global_step < args.max_steps:
        batch = next(train_iterator)
        if not batch:
            continue
        
        optimizer.zero_grad()
        
        output = model(
            token_encoder_inputs=batch["token_encoder_inputs"],
            type_encoder_inputs=batch["type_encoder_inputs"],
            labels=batch["labels"]
        )
        loss = output.loss

        accelerator.backward(loss)
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        loss_value = loss.detach().item()
        total_loss += loss_value
        num_batches += 1
        global_step += 1

        if hasattr(scheduler, 'get_last_lr'):
            lr = scheduler.get_last_lr()[0] if isinstance(scheduler.get_last_lr(), list) else scheduler.get_last_lr()
        else:
            lr = optimizer.param_groups[0]['lr']
        
        epoch = global_step / steps_per_epoch if steps_per_epoch > 0 else 0.0
        
        if global_step % logging_steps == 0 and accelerator.is_main_process:
            avg_loss = total_loss / num_batches
            metrics = {
                'loss': round(avg_loss, 4),
                'grad_norm': round(grad_norm.item(), 4) if isinstance(grad_norm, torch.Tensor) else round(grad_norm, 4),
                'learning_rate': '{:.2e}'.format(lr),
                'epoch': round(epoch, 2)
            }
            progress_bar.write(str(metrics))
        
        progress_bar.update(1)
        
        if global_step % args.eval_steps == 0:
            if accelerator.is_main_process:
                logger.info(f"\n{'='*50}")
                logger.info(f"Step {global_step}/{args.max_steps}")
                logger.info(f"Training Loss: {total_loss/num_batches:.4f}")
            
            if eval_dataloader is not None:
                eval_metrics = evaluate(model, eval_dataloader, accelerator)
            else:
                eval_metrics = {"loss": 0.0, "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0}}
            
            if accelerator.is_main_process:
                logger.info(f"Evaluation Loss: {eval_metrics['loss']:.4f}")
                logger.info(f"Precision: {eval_metrics['micro']['precision']:.4f}")
                logger.info(f"Recall: {eval_metrics['micro']['recall']:.4f}")
                logger.info(f"F1 Score: {eval_metrics['micro']['f1']:.4f}")
            
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(str(checkpoint_dir))
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            current_f1 = eval_metrics['micro']['f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_checkpoint_path = checkpoint_dir
                patience_counter = 0
                if accelerator.is_main_process:
                    logger.info(f"New best F1: {best_f1:.4f} at checkpoint {checkpoint_dir}")
            else:
                patience_counter += 1
                if accelerator.is_main_process:
                    logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
            best_models.append((current_f1, checkpoint_dir, global_step))
            best_models.sort(key=lambda x: x[0], reverse=True)
            
            if len(best_models) > save_total_limit:
                _, worst_checkpoint_path, _ = best_models[save_total_limit]
                if accelerator.is_main_process:
                    shutil.rmtree(worst_checkpoint_path)
            
            best_models = best_models[:save_total_limit]
            
            if accelerator.is_main_process:
                logger.info(f"{'='*50}\n")
            
            if patience_counter >= patience:
                if accelerator.is_main_process:
                    logger.info(f"Early stopping triggered after {patience} evaluations without improvement.")
                    logger.info(f"Best F1 score: {best_f1:.4f}")
                break
            
            total_loss = 0.0
            num_batches = 0
            model.train()
    
    progress_bar.close()
    
    if best_checkpoint_path is not None and accelerator.is_main_process:
        best_checkpoint_link = Path(args.output_dir) / "best_checkpoint"
        if best_checkpoint_link.exists():
            if best_checkpoint_link.is_symlink():
                best_checkpoint_link.unlink()
            elif best_checkpoint_link.is_dir():
                shutil.rmtree(best_checkpoint_link)
            else:
                best_checkpoint_link.unlink()
        best_checkpoint_link.symlink_to(best_checkpoint_path.relative_to(Path(args.output_dir)))
        logger.info(f"Best checkpoint (F1={best_f1:.4f}) saved at: {best_checkpoint_path}")
        logger.info(f"Symlink created: {best_checkpoint_link} -> {best_checkpoint_path.relative_to(Path(args.output_dir))}")
    
    return global_step, best_checkpoint_path, best_f1

