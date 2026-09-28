device_id=0
inverse_problem="sr_4"
low_idx=1
high_idx=30 # need to run for 10, 20, 30

python scripts/train_inference_model.py \
  inverse_problem="${inverse_problem}" \
  inverse_problem.noise_level=0.05 \
  dataset="imagenet_train" \
  model="hdit_imagenet" \
  inference_model_type="hdit_palette_imagenet" \
  training.train_split=0.01 \
  training.val_split=0.001 \
  training.num_epochs=20 \
  training.print_ratio=True \
  training.batch_size=16 \
  training.lr=0.00001 \
  training.wd=0.0 \
  task.deterministic=False \
  task.low_idx=${low_idx} \
  task.high_idx=${high_idx} \
  device="cuda:${device_id}"
