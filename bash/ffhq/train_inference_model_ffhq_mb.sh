device_id=0
inverse_problem="mb_21_0.9"
low_idx=1
high_idx=30 # need to run for 10, 20, 30

python scripts/train_inference_model.py \
  inverse_problem="${inverse_problem}" \
  inverse_problem.noise_level=0.05 \
  dataset="ffhq_train" \
  model="hdit_ffhq" \
  inference_model_type="hdit_palette_ffhq" \
  training.train_split=1.0 \
  training.val_split=0.005 \
  training.num_epochs=5 \
  training.print_ratio=True \
  training.batch_size=16 \
  training.lr=0.0001 \
  training.wd=0.00001 \
  task.deterministic=False \
  task.low_idx=${low_idx} \
  task.high_idx=${high_idx} \
  device="cuda:${device_id}"
