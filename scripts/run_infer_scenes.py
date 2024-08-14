from path import Path
import os
import argparse

parser = argparse.ArgumentParser(description='Run scenes metric script. Sub-directory names are shown in the scenes.txt file.')
parser.add_argument('--root-path', type=str, default="/home/zanxin/jiahan/.dataset4SCDepth", help='Root path')
parser.add_argument('--save-root-path', type=str, default="/home/zanxin/jiahan/depth-anything-v2-depth/C3VD_metric", help='Save root path')
parser.add_argument('--max-depth', type=int, default=100, help='Max depth')
parser.add_argument('--load-from', type=str, default="checkpoints/depth_anything_v2_metric_hypersim_vitl.pth", help='Load from')
parser.add_argument('--encoder', type=str, default="vitl", help='Encoder')

args = parser.parse_args()

root_path = Path(args.root_path)
save_root_path = Path(args.save_root_path)
max_depth = args.max_depth
load_from = args.load_from
encoder = args.encoder



save_root_path.makedirs_p()

scenes = []
with open(root_path / "scenes.txt", 'r') as f:
    scenes = f.read().splitlines()

print(scenes)

for scene in scenes:
    print("=> Processing", scene)
    scene_path = root_path / scene
    scene_save_path = save_root_path / scene
    scene_save_path.makedirs_p()
    
    cmd = f"""python metric_depth/run.py \
--encoder {encoder} \
--img-path {scene_path} \
--load-from {load_from} \
--max-depth {max_depth} \
--outdir {scene_save_path} \
--save-numpy \
--pred-only \
--grayscale
"""
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        raise Exception("Failed to run the command")
