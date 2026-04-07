import os
import sys
import subprocess
import shutil
from os.path import join
import yaml
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", "-i", dest="input_dir", type=str)
parser.add_argument("--output_dir", "-o", dest="output_dir", type=str, default=None)
parser.add_argument("--threads", "-th", dest="threads", type=int, default=64)
parser.add_argument("--images", "-im", dest="images", type=str, default="images")

args = parser.parse_args()
args.output_dir = args.output_dir or join(os.path.abspath(args.input_dir), "openmvg")

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
                "openmvg",
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

os.makedirs(join(process_dir, "matches"))
os.makedirs(join(process_dir, "reconstruction"))

print(f"Running OpenMVG on {process_dir} ...")
print("1. Intrinsics analysis")
pIntrisics = subprocess.Popen(
    [
        "docker",
        "run",
        "-it",
        "--rm",
        "-v",
        f"{process_dir}:/data",
        "openmvg",
        "openMVG_main_SfMInit_ImageListing",
        "-i",
        "/data/images",
        "-o",
        "/data/matches",
        "-f",
        "1",
        "-c",
        "7",
    ]
)
pIntrisics.wait()


print("2. Compute features")
pIntrisics = subprocess.Popen(
    [
        "docker",
        "run",
        "-it",
        "--rm",
        "-v",
        f"{process_dir}:/data",
        "openmvg",
        "openMVG_main_ComputeFeatures",
        "-i",
        "/data/matches/sfm_data.json",
        "-o",
        "/data/matches",
        "-m",
        "SIFT",
        "-p",
        "ULTRA",
        "-n",
        f"{args.threads}",
    ]
)
pIntrisics.wait()


print("3. Compute matching pairs")
pIntrisics = subprocess.Popen(
    [
        "docker",
        "run",
        "-it",
        "--rm",
        "-v",
        f"{process_dir}:/data",
        "openmvg",
        "openMVG_main_PairGenerator",
        "-i",
        "/data/matches/sfm_data.json",
        "-o",
        "/data/matches/pairs.bin",
    ]
)
pIntrisics.wait()

print("4. Compute matches")
pIntrisics = subprocess.Popen(
    [
        "docker",
        "run",
        "-it",
        "--rm",
        "-v",
        f"{process_dir}:/data",
        "openmvg",
        "openMVG_main_ComputeMatches",
        "-i",
        "/data/matches/sfm_data.json",
        "-p",
        "/data/matches/pairs.bin",
        "-o",
        "/data/matches/matches.putative.bin",
    ]
)
pIntrisics.wait()


print("5. Filter matches")
pIntrisics = subprocess.Popen(
    [
        "docker",
        "run",
        "-it",
        "--rm",
        "-v",
        f"{process_dir}:/data",
        "openmvg",
        "openMVG_main_GeometricFilter",
        "-i",
        "/data/matches/sfm_data.json",
        "-m",
        "/data/matches/matches.putative.bin",
        "-g",
        "a",
        "-o",
        "/data/matches/matches.f.bin",
    ]
)
pIntrisics.wait()

print("6. Do Sequential/Incremental reconstruction")
pIntrisics = subprocess.Popen(
    [
        "docker",
        "run",
        "-it",
        "--rm",
        "-v",
        f"{process_dir}:/data",
        "openmvg",
        "openMVG_main_SfM",
        "-s",
        "INCREMENTAL",
        "-i",
        "/data/matches/sfm_data.json",
        "-m",
        "/data/matches",
        "-o",
        "/data/reconstruction",
    ]
)
pIntrisics.wait()


print("7. Colorize Structure")
pIntrisics = subprocess.Popen(
    [
        "docker",
        "run",
        "-it",
        "--rm",
        "-v",
        f"{process_dir}:/data",
        "openmvg",
        "openMVG_main_ComputeSfM_DataColor",
        "-i",
        "/data/reconstruction/sfm_data.bin",
        "-o",
        "/data/reconstruction/colorized.ply",
    ]
)
pIntrisics.wait()


print("8. Convert DataFormat")
pIntrisics = subprocess.Popen(
    [
        "docker",
        "run",
        "-it",
        "--rm",
        "-v",
        f"{process_dir}:/data",
        "openmvg",
        "openMVG_main_ConvertSfM_DataFormat",
        "-i",
        "/data/reconstruction/sfm_data.bin",
        "-o",
        "/data/reconstruction/sfm_data.json",
    ]
)
pIntrisics.wait()

print(f"Copying OpenMVG's results to {output_dir} ...")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir)
if os.path.exists(join(process_dir, "reconstruction", "colorized.ply")):
    shutil.copyfile(join(process_dir, "reconstruction", "colorized.ply"), join(output_dir, "colorized.ply"))
if os.path.exists(join(process_dir, "reconstruction", "sfm_data.bin")):
    shutil.copyfile(join(process_dir, "reconstruction", "sfm_data.bin"), join(output_dir, "sfm_data.bin"))
if os.path.exists(join(process_dir, "reconstruction", "sfm_data.json")):
    shutil.copyfile(join(process_dir, "reconstruction", "sfm_data.json"), join(output_dir, "sfm_data.json"))

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
            "openmvg",
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
