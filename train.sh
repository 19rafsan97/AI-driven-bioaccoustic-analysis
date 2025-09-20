# optional: help fragmentation on 4 GB VRAM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python train_ast.py \
  --data_root "." \
  --spectrograms_dir "data/spectrograms" \
  --metadata_csv "data/metadata.csv" \
  --group_column recording_id \
  --output_dir "results/ast_wmwb" \
  --model_name "MIT/ast-finetuned-audioset-10-10-0.4593" \
  --epochs 25 \
  --target_frames 1024 \
  --batch_size 4 \
  --grad_accum 2 \
  --lr 5e-5 \
  --weight_decay 0.05 \
  --warmup_steps 500 \
  --label_smoothing 0.05 \
  --compute_train_mean_std \
  --specaug \
  --mixup 0.2 \
  --amp \
  --grad_ckpt \
  --use_class_weights \
  --save_predictions
