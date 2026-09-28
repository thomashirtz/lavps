device_id=0
inverse_problem="sr_4"

python scripts/generate_images.py \
  device=cuda:${device_id} \
  model=hdit_imagenet \
  inverse_problem="${inverse_problem}" \
  dataset="imagenet_val" \
  sampler="mgdm"
