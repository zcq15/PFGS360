import os
import sys
import subprocess
import shutil
from os.path import join
import yaml
import json

import argparse
import ipdb


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", "-i", dest="input_dir", type=str)
parser.add_argument("--output_dir", "-o", dest="output_dir", type=str, default=None)
parser.add_argument("--width", "-W", dest="image_w", type=int)
parser.add_argument("--height", "-H", dest="image_h", type=int)
parser.add_argument("--threads", "-th", dest="threads", type=int, default=32)
parser.add_argument("--images", "-im", dest="images", type=str, default="images")
parser.add_argument("--stage", "-s", dest="stage", type=str, default="sfm")

args = parser.parse_args()
args.output_dir = args.output_dir or join(os.path.abspath(args.input_dir), "opensfm")

input_dir = os.path.abspath(args.input_dir)
output_dir = os.path.abspath(args.output_dir)
process_dir = output_dir + "_running"

assert os.path.exists(join(input_dir, args.images)), f"Image folder {join(input_dir, args.images)} does not exist!"

if os.path.exists(process_dir):
    print(f"Removing the existing temporary folder {process_dir} ...")
    try:
        shutil.rmtree(process_dir)
    except:
        print(f"The process directory {process_dir} can not be removed, try deleting the folder in docker!")
        pIntrisics = subprocess.Popen(
            [
                "docker",
                "run",
                "-it",
                "--rm",
                "-v",
                f"{process_dir}:/data",
                "opensfm",
                "rm",
                "-rf",
                "/data/*",
            ]
        )
        pIntrisics.wait()
        shutil.rmtree(process_dir)
    if os.path.exists(process_dir):
        print(f"The process directory {process_dir} still can not be removed!")
        exit(-1)

print(f"Copying image folder {join(input_dir, args.images)} to {join(process_dir, 'images')} ...")
os.makedirs(process_dir)
shutil.copytree(join(input_dir, args.images), join(process_dir, "images"))

print(
    f"Preparing OpenSfM's config {join(process_dir, 'config.yaml')} and cameras' config {join(process_dir, 'camera_models_overrides.json')} ..."
)
configs = {
    "processes": args.threads,
}
# configs = {}
cameras = {
    "all": {
        "projection_type": "spherical",
        "width": args.image_w,
        "height": args.image_h,
    }
}
with open(join(process_dir, "config.yaml"), "w", encoding="utf-8") as f:
    yaml.dump(configs, f, indent=4)
with open(join(process_dir, "camera_models_overrides.json"), "w", encoding="utf-8") as f:
    json.dump(cameras, f, indent=4)

print(f"Running OpenSfM on {process_dir} ...")
if args.stage == "sfm":
    pIntrisics = subprocess.Popen(
        [
            "docker",
            "run",
            "-it",
            "--rm",
            "-v",
            f"{process_dir}:/data",
            "opensfm",
            "bin/opensfm_sfm",
            "/data",
        ]
    )
else:
    pIntrisics = subprocess.Popen(
        [
            "docker",
            "run",
            "-it",
            "--rm",
            "-v",
            f"{process_dir}:/data",
            "opensfm",
            "bin/opensfm_run_all",
            "/data",
        ]
    )
pIntrisics.wait()

print(f"Copying OpenSfM's results to {output_dir} ...")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir)
if os.path.exists(join(process_dir, "reconstruction.json")):
    shutil.copyfile(join(process_dir, "reconstruction.json"), join(output_dir, "reconstruction.json"))
if os.path.exists(join(process_dir, "undistorted/depthmaps/merged.ply")):
    shutil.copyfile(join(process_dir, "undistorted/depthmaps/merged.ply"), join(output_dir, "merged.ply"))

print(f"Removing temporary folder {process_dir} ...")
try:
    pIntrisics = subprocess.Popen(
        [
            "docker",
            "run",
            "-it",
            "--rm",
            "-v",
            f"{process_dir}:/data",
            "opensfm",
            "rm",
            "-rf",
            "/data/*",
        ]
    )
    pIntrisics.wait()
except:
    pass
try:
    shutil.rmtree(process_dir)
except:
    pass

if os.path.exists(process_dir):
    print(f"Temporary folder {process_dir} still exists and needs to be deleted manuall!")
