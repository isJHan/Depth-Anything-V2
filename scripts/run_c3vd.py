from path import Path
import os

root_path = Path("/home/zanxin/jiahan/.dataset4SCDepth")
save_root_path = Path("/home/zanxin/jiahan/depth-anything-v2-depth/C3VD")
save_root_path.makedirs_p()
scenes = [
    "scene_cecum_t1_a",
    "scene_cecum_t1_b",
    "scene_cecum_t2_a",
    "scene_cecum_t2_b",
    "scene_cecum_t2_c",
    "scene_cecum_t3_a",
    "scene_cecum_t4_a",
    "scene_cecum_t4_b",
    "scene_trans_t1_a",
    "scene_trans_t1_b",
    "scene_trans_t2_a",
    "scene_trans_t2_b",
    "scene_trans_t2_c",
    "scene_trans_t3_a",
    "scene_trans_t3_b",
    "scene_trans_t4_a",
    "scene_trans_t4_b",
    "scene_sigmoid_t1_a",
    "scene_sigmoid_t2_a",
    "scene_sigmoid_t3_a",
    "scene_sigmoid_t3_b",
    "scene_desc_t4_a"
]

for scene in scenes:
    print("=> Processing", scene)
    scene_path = root_path / scene
    scene_save_path = save_root_path / scene
    scene_save_path.makedirs_p()
    
    cmd = f"""python run.py \
--encoder vitl \
--img-path {scene_path} \
--outdir {scene_save_path} \
--pred-only \
--grayscale
"""
    print(cmd)
    os.system(cmd)
