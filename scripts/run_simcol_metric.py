from path import Path
import os

import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Run SimCol metric script')
# Add arguments to the parser
parser.add_argument('--root-path', type=str, default="/home/zanxin/jiahan/SimCol3D4SC_Depth", help='Root path')
parser.add_argument('--save-root-path', type=str, default="/home/zanxin/jiahan/depth-anything-v2-depth/simcol_metric", help='Save root path')
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
    "SyntheticColon_I/Frames_S15",
    "SyntheticColon_I/Frames_S7",
    "SyntheticColon_I/Frames_S12",
    "SyntheticColon_I/Frames_S13",
    "SyntheticColon_I/Frames_S1",
    "SyntheticColon_I/Frames_S4",
    "SyntheticColon_I/Frames_S3",
    "SyntheticColon_I/Frames_S2",
    "SyntheticColon_I/Frames_S11",
    "SyntheticColon_I/Frames_S14",
    "SyntheticColon_I/Frames_S9",
    "SyntheticColon_I/Frames_S6",
    "SyntheticColon_I/Frames_S8",
    "SyntheticColon_I/Frames_S5",
    "SyntheticColon_I/Frames_S10",
    "SyntheticColon_II/Frames_B12",
    "SyntheticColon_II/Frames_B2",
    "SyntheticColon_II/Frames_B1",
    "SyntheticColon_II/Frames_B7",
    "SyntheticColon_II/Frames_B13",
    "SyntheticColon_II/Frames_B10",
    "SyntheticColon_II/Frames_B4",
    "SyntheticColon_II/Frames_B8",
    "SyntheticColon_II/Frames_B14",
    "SyntheticColon_II/Frames_B5",
    "SyntheticColon_II/Frames_B9",
    "SyntheticColon_II/Frames_B6",
    "SyntheticColon_II/Frames_B3",
    "SyntheticColon_II/Frames_B15",
    "SyntheticColon_II/Frames_B11",
    "SyntheticColon_III/Frames_O3",
    "SyntheticColon_III/Frames_O2",
    "SyntheticColon_III/Frames_O1",
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
