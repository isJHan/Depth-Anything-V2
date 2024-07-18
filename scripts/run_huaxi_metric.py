from path import Path
import os

import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Run SimCol metric script')
# Add arguments to the parser
parser.add_argument('--root-path', type=str, default="/Disk_2/Jiahan/huaxi_dataset", help='Root path')
parser.add_argument('--save-root-path', type=str, default="/home/zanxin/jiahan/depth-anything-v2-depth/huaxi_metric", help='Save root path')
parser.add_argument('--max-depth', type=int, default=200, help='Max depth')
parser.add_argument('--load-from', type=str, default="checkpoints/depth_anything_v2_metric_hypersim_vitl.pth", help='Load from')
parser.add_argument('--encoder', type=str, default="vitl", help='Encoder')

# Parse the arguments
args = parser.parse_args()

# Convert arguments to Path objects where necessary and assign to variables
root_path = Path(args.root_path)
save_root_path = Path(args.save_root_path)
max_depth = args.max_depth
load_from = args.load_from
encoder = args.encoder


save_root_path.makedirs_p()
scenes = [
    "undist_20240313_175728_测大小-30mm-带蒂",
    "undist_20240313_175415_测大小-18mm-带蒂",
    "undist_20240313_174853_测大小-15mm-带蒂",
    "undist_20240313_181002-测大小-12mm-原装2号-带蒂",
    "undist_20240313_180538-测大小-11mm-原装1号",
    "undist_20240313_181514-测大小-12mm-原装4号-扁平",
    "undist_20240313_181228-测大小-8mm-原装3号",
]

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
    os.system(cmd)
