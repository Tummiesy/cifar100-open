# CIFAR-100 Instance-Dependent Open/Closed-Set Noise Pipeline

This repo provides a PyTorch prototype to construct **instance-dependent noisy labels** on CIFAR-100.

- **Open-set instance-dependent noise**: unknown-class images are assigned noisy labels in the known-class space using `softmax(logits / T)` from a clean reference model.
- **Closed-set instance-dependent noise**: known-class images are relabeled to other known classes with sample-wise transition distributions that mask out each sample's ground-truth class.

## Files
- `utils.py`: seed control, class split utilities, IO helpers.
- `datasets.py`: class-filtered dataset wrappers and merged noisy dataset builders.
- `model.py`: CIFAR-adapted ResNet-18 with optional feature output (prototype extension point).
- `train_ref_model.py`: train clean reference model on known classes only.
- `generate_open_set_noise.py`: generate unknown->known open-set noisy labels and build open-set final dataset.
- `generate_closed_set_noise.py`: generate known->known closed-set instance-dependent noisy labels.
- `generate_mixed_noise.py`: generate closed-set known noise + open-set unknown noise and merge into one final dataset.

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

Generate instance-dependent **closed-set** noise only:

```bash
python generate_closed_set_noise.py \
  --data_root ./data \
  --output_dir ./outputs/closed_noise_run \
  --ref_ckpt ./outputs/ref_run/reference_model_best.pth \
  --seed 42 \
  --temperature 1.0 \
  --closed_set_noise_rate 0.2
```

Generate instance-dependent **open-set** noise only:

```bash
python generate_open_set_noise.py \
  --data_root ./data \
  --output_dir ./outputs/open_noise_run \
  --ref_ckpt ./outputs/ref_run/reference_model_best.pth \
  --seed 42 \
  --temperature 1.0 \
  --hardness_mode topk \
  --topk 5000 \
  --open_set_noise_ratio 0.2 \
  --ratio_mode fraction_total
```

Generate **mixed** dataset (closed-set known + open-set unknown):

```bash
python generate_mixed_noise.py \
  --data_root ./data \
  --output_dir ./outputs/mixed_noise_run \
  --ref_ckpt ./outputs/ref_run/reference_model_best.pth \
  --seed 42 \
  --num_unknown_classes 20 \
  --temperature 1.0 \
  --closed_set_noise_rate 0.2 \
  --open_set_noise_ratio 0.2 \
  --ratio_mode fraction_total \
  --hardness_mode topk \
  --topk 5000
```

## Closed-set instance-dependent noise details

The closed-set generation follows the per-instance transition concept (Algorithm 1 style):

1. Predict per-instance class probabilities with the clean reference model.
2. Mask the sample ground-truth class so it cannot be selected as a flipped class.
3. Renormalize probabilities over non-ground-truth known classes.
4. Draw sample-wise `q_i` from a clipped normal distribution (`mean=closed_set_noise_rate`, `std=0.1`, clipped to `[0, 1]`).
5. Build per-sample transition distribution:
   - `P(y_i) = 1 - q_i`
   - `P(c != y_i) = q_i * pi_i(c)`
6. Sample final training labels from this transition distribution.

Use fixed known classes via `--known_classes_file` in `train_ref_model.py`.
The file can be JSON list or newline-separated class ids.
