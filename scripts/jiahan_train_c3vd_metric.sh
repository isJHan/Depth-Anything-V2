

python metric_depth/train.py \
--encoder vits \
--dataset c3vd_flip_and_swap \
--max-depth 100 \
--epochs 10 \
--save-path ./save_checkpoints/vits_c3vd_metric_flip_and_swap \
--pretrained-from checkpoints/depth_anything_v2_metric_hypersim_vits.pth
