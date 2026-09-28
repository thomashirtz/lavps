device_id=0
inverse_problem="mb_21_0.9"

python scripts/generate_images.py \
  device=cuda:${device_id} \
  model=hdit_ffhq \
  inverse_problem="${inverse_problem}" \
  dataset="ffhq_val" \
  sampler="mgdm"
