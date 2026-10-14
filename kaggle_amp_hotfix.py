# Paste into a cell in the existing Kaggle notebook, then rerun the training cell.
# Keep the source file unchanged so completed E0/E1 cache signatures remain valid.
import kaggle_vlm as lab

code = r'''def amp_optimizer_step(optimizer, scaler, params, max_norm=1.):
    """Clear overflowing gradients, lower the scale, and never update bad weights."""
    import torch
    scaler.unscale_(optimizer)
    # Check BEFORE clipping: clipping an infinite gradient can produce NaNs.
    finite = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm) if finite else None
    if not finite or not torch.isfinite(norm).item():
        new_scale = scaler.get_scale() * scaler.get_backoff_factor()
        if not math.isfinite(new_scale) or new_scale <= 0:
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError('AMP scale exhausted; inspect model activations or reduce LR')
        # update(new_scale=...) resets the unscale state for the next retry. This
        # also handles aggregate norm overflow when individual gradients are finite.
        scaler.update(new_scale=new_scale)
        optimizer.zero_grad(set_to_none=True)
        return False, None
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return True, norm.item()


def train(cfg, key):
    import torch
    from torch.utils.data import DataLoader
    output = Path('work') / key
    final = output / 'final'
    signature = fingerprint(cfg)
    if (final / 'run.json').exists():
        run = json.loads((final / 'run.json').read_text(encoding='utf-8'))
        if run.get('fingerprint') != signature:
            raise RuntimeError(f'{final} belongs to another configuration; use a new run name')
        print('Reuse completed training:', key)
        return run
    seed_everything(cfg['seed'])
    rows = selected_train_rows(cfg)
    annotation_stats = audit_annotations(rows, cfg['coord'])
    write_json(output / 'annotation_audit.json', annotation_stats)
    print('Annotation audit:', annotation_stats, flush=True)
    model, processor = load_model(cfg)
    try:
        model, matched = apply_setup(model, cfg)
        processor.tokenizer.padding_side = 'right'
        class Dataset:
            def __len__(self): return len(rows)
            def __getitem__(self, i): return encode_example(processor, rows[i], cfg['coord'])
        loader = DataLoader(Dataset(), batch_size=cfg['batch_size'], shuffle=True,
            num_workers=0, generator=torch.Generator().manual_seed(cfg['seed']),
            collate_fn=lambda b: collate(b, processor.tokenizer.pad_token_id))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=0)
        warmup = max(1, round(cfg['max_steps'] * .05))
        def lr_factor(step):
            if step < warmup: return (step + 1) / warmup
            progress = (step - warmup) / max(1, cfg['max_steps'] - warmup)
            return .5 * (1 + math.cos(math.pi * min(1, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        scaler = torch.amp.GradScaler('cuda', init_scale=4096.)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        iterator = iter(loader)
        history = []
        overflow_events = []
        write_json(output / 'config.json', cfg)
        for step in range(1, cfg['max_steps'] + 1):
            # Keep one effective batch in CPU RAM; retry these same images on
            # overflow rather than losing data or counting a skipped optimizer step.
            micro_batches = []
            for _ in range(cfg['grad_accum']):
                try: batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                micro_batches.append(batch)
            rng_cpu = torch.get_rng_state()
            rng_cuda = torch.cuda.get_rng_state(0)
            for retry in range(9):
                # Reproduce dropout for the same effective batch on each retry.
                torch.set_rng_state(rng_cpu)
                torch.cuda.set_rng_state(rng_cuda, device=0)
                total_loss = 0.
                for cpu_batch in micro_batches:
                    batch = {k: v.to('cuda:0') for k, v in cpu_batch.items()}
                    with torch.autocast('cuda', dtype=torch.float16):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Non-finite loss at step {step}; lower LR and inspect labels')
                    total_loss += loss.detach().float().item() / cfg['grad_accum']
                    scaler.scale(loss / cfg['grad_accum']).backward()
                updated, grad_norm = amp_optimizer_step(optimizer, scaler, params)
                if updated: break
                event = {'step': step, 'retry': retry+1, 'new_scale': scaler.get_scale()}
                overflow_events.append(event)
                print(f'{key} AMP overflow: {event}; retrying the same batch.', flush=True)
                write_json(output / 'overflow.json', overflow_events)
            else:
                raise RuntimeError(f'Persistent non-finite gradients at step {step} after 9 attempts; '
                                   'lower LR and inspect forward/backward activations')
            # Advance the LR schedule only after an actual optimizer update.
            scheduler.step()
            history.append({'step': step, 'loss': total_loss, 'lr': scheduler.get_last_lr()[0],
                            'grad_norm': grad_norm, 'amp_scale': scaler.get_scale()})
            if step == 1 or step % 10 == 0:
                print(f'{key} step={step}/{cfg["max_steps"]} loss={total_loss:.4f}', flush=True)
                write_json(output / 'loss.json', history)
            if step % 100 == 0 and step != cfg['max_steps']:
                # One rolling checkpoint limits disk usage. Interrupted training restarts;
                # optimizer/RNG are intentionally not presented as resumable state.
                save_checkpoint(model, processor, output / 'checkpoint', cfg, step)
        torch.cuda.synchronize()
        run = {'trainable_params': sum(p.numel() for p in params),
               'peak_vram_gb': torch.cuda.max_memory_allocated() / 1024**3,
               'train_minutes': (time.perf_counter() - started) / 60,
               'fingerprint': signature, 'steps': cfg['max_steps'], 'train_images': len(rows),
               'matched_modules': len(matched), 'config': cfg,
               'annotation_audit': annotation_stats,
               'amp_policy': 'retry_same_batch_v1', 'gradient_overflows': len(overflow_events),
               'final_amp_scale': scaler.get_scale(),
               'student_name': STUDENT_NAME, 'student_id': STUDENT_ID}
        save_checkpoint(model, processor, final, cfg, cfg['max_steps'], run)
        write_json(output / 'loss.json', history)
        print(json.dumps(run, ensure_ascii=False, indent=2))
        return run
    finally:
        # Release the model even after OOM; each experiment starts from the same base.
        if 'optimizer' in locals(): del optimizer
        if 'scheduler' in locals(): del scheduler
        if 'iterator' in locals(): del iterator
        if 'params' in locals(): del params
        if 'batch' in locals(): del batch
        if 'micro_batches' in locals(): del micro_batches
        if 'cpu_batch' in locals(): del cpu_batch
        if 'loss' in locals(): del loss
        if 'loader' in locals(): del loader
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
'''
exec(code, lab.__dict__)
print('AMP recovery patched in memory. Rerun the training cell; E2 restarts, E1 is reused.')
