

python metric_depth/train.py \
--encoder vits \
--dataset SimCol_flip_and_swap \
--max-depth 200 \
--epochs 10 \
--save-path ./save_checkpoints/vits_simcol_metric_flip_and_swap \
--pretrained-from checkpoints/depth_anything_v2_metric_hypersim_vits.pth
