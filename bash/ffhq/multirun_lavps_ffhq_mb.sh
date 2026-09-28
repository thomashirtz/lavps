device_id=0
inverse_problem="mb_21_0.9"
model_path=""

python scripts/generate_images.py -m \
  device=cuda:${device_id} \
  model=hdit_ffhq \
  inverse_problem="${inverse_problem}" \
  dataset="ffhq_val" \
  sampler.high_idx=30 \
  sampler.parameters.gradient_steps_fn.conditions.0.return=0,1,3,10 \
  sampler.parameters.gradient_steps_fn.conditions.1.return=1,3 \
  sampler.parameters.n_denoising_steps=1 \
  sampler.parameters.lr_fn.conditions.0.return=0.01,0.03 \
  sampler.parameters.initialize_va_cfg.path="${model_path}" \
  sampler.parameters.initialize_va_cfg.model=hdit_palette_ffhq \
  sampler="lavps"
