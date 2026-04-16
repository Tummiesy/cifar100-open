# CIFAR-100 Instance-Dependent Open-Set Noise Pipeline

This repo provides a PyTorch prototype to construct **instance-dependent open-set noisy labels** on CIFAR-100.

## Files
- `utils.py`: seed control, class split utilities, IO helpers.
- `datasets.py`: class-filtered dataset wrappers and final merged noisy dataset class.
- `model.py`: CIFAR-adapted ResNet-18 with optional feature output (prototype extension point).
- `train_ref_model.py`: train clean reference model on known classes only.
- `generate_open_set_noise.py`: generate unknown->known noisy labels using `softmax(logits / T)` and construct final dataset.

## Example usage

Train reference model:

```bash
python train_ref_model.py \
  --data_root ./data \
  --output_dir ./outputs/ref_run \
  --seed 42 \
  --num_unknown_classes 20 \
  --batch_size 128 \
  --epochs 100 \
  --lr 0.1
```

Generate instance-dependent open-set noise:

```bash
python generate_open_set_noise.py \
  --data_root ./data \
  --output_dir ./outputs/noise_run \
  --ref_ckpt ./outputs/ref_run/reference_model_best.pth \
  --seed 42 \
  --temperature 1.0 \
  --hardness_mode topk \
  --topk 5000 \
  --open_set_noise_ratio 0.2 \
  --ratio_mode fraction_total
```

Use fixed known classes via `--known_classes_file` in `train_ref_model.py`.
The file can be JSON list or newline-separated class ids.
