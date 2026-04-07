import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import subprocess
import json
from collections import OrderedDict
import yaml
import argparse
import numpy as np

from os.path import join
import os
import shutil
import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", "--cuda", "-c", "-g", dest="cuda", default="0", type=str)
parser.add_argument("--port", "-p", dest="port", default=None, type=int)
parser.add_argument("--experiment", "-e", dest="experiment", choices=["Egocentric", "Non-Egocentric"], type=str)
parser.add_argument("--suffix", "-s", dest="suffix", default=None, type=str)
parser.add_argument("--model", "-m", dest="model", default="pfgs360", type=str)
parser.add_argument("--datadir", "-d", dest="datadir", default="./datasets/OB3D", type=str)
parser.add_argument("--scene", dest="scene", default=None, type=str, nargs="+")
parser.add_argument("--savedir", dest="savedir", default="outputs", type=str)
parser.add_argument("--dataparser", dest="dataparser", default="ob3d-dataparser", type=str)
parser.add_argument("--stages", dest="stages", nargs="+", default=["train", "render", "evaluate"], type=str)
parser.add_argument("--args", "-a", dest="args", default=None, type=str, nargs="+")
opts = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.cuda)
if opts.port is None:
    opts.port = 7000 + int(opts.cuda)

opts.datadir = os.path.normpath(os.path.abspath(opts.datadir))

if os.path.exists(join(opts.datadir, "Egocentric", "images")) or os.path.exists(
    join(opts.datadir, "Non-Egocentric", "images")
):
    opts.scene = [os.path.basename(opts.datadir)]
    opts.datadir = os.path.dirname(os.path.normpath(opts.datadir))

if opts.scene is None:
    scenes = []
    for scene in sorted(os.listdir(opts.datadir)):
        if os.path.isdir(os.path.join(opts.datadir, scene)):
            scenes.append(scene)
elif isinstance(opts.scene, str):
    scenes = [opts.scene]
else:
    scenes = opts.scene

print(f"run for scenes: {scenes}")

dataset = opts.datadir
outputs = opts.savedir
method = OrderedDict(
    [
        ("--experiment-name", f"{os.path.basename(opts.datadir)}-{opts.experiment}"),
        ("--viewer.websocket-port", opts.port),
    ]
)
if opts.suffix is not None:
    method.update({"--pipeline.suffix": opts.suffix})

args = OrderedDict([])
if opts.args is None:
    pass
elif isinstance(opts.args, str):
    k, v = opts.args.strip().split(":")[:2]
    args.update({f"--pipeline.model.{k}": v})
else:
    for line in opts.args:
        k, v = line.strip().split(":")[:2]
        args.update({f"--pipeline.model.{k}": v})

data = OrderedDict(
    [
        ("dataparser", opts.dataparser),
        ("--trajectory_type", opts.experiment),
    ]
)
opts.experiment = f"{os.path.basename(opts.datadir)}-{opts.experiment}"


def pipeline(scenes, args):

    def export_package(savedir):
        import nerfstudio360

        path = nerfstudio360.__path__[0]
        shutil.copytree(path, savedir, dirs_exist_ok=True)

    scene_index = 0
    for scene in scenes:
        scene_index += 1

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if "train" in opts.stages:
            cmd = "ns-train {}".format(opts.model)
            for k, v in method.items():
                if k.startswith("--"):
                    cmd += " {} {}".format(k, v)
            for k, v in args.items():
                if k.startswith("--"):
                    cmd += " {} {}".format(k, v)

            cmd += " {}".format(data["dataparser"])
            cmd += " --data {}".format(os.path.join(dataset, scene))
            for k, v in data.items():
                if k.startswith("--"):
                    cmd += " {} {}".format(k, v)

            if opts.suffix is None:
                package_savedir = join(outputs, opts.experiment, scene, opts.model, f"nerfstudio360-{timestamp}")
            else:
                package_savedir = join(
                    outputs, opts.experiment, scene, f"{opts.model}-{opts.suffix}", f"nerfstudio360-{timestamp}"
                )
            export_package(package_savedir)

            print(f"training scene {scene_index}/{len(scenes)}: {cmd}")
            try:
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                exit(-1)

        if "render" in opts.stages:
            if opts.suffix is None:
                model_dir = join(outputs, opts.experiment, scene, opts.model)
            else:
                model_dir = join(outputs, opts.experiment, scene, f"{opts.model}-{opts.suffix}")
            cmd = "ns-eval --load-config {} --output-path {} --render-output-path {}".format(
                join(model_dir, "config.yml"),
                join(model_dir, "results.json"),
                join(model_dir, "results"),
            )

            print(f"rendering scene {scene_index}/{len(scenes)}: {cmd}")
            try:
                process = subprocess.Popen(cmd, shell=True)
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                exit(-1)

            try:
                shutil.copytree(
                    join(model_dir, "config.yml"), join(model_dir, f"config-{timestamp}.yml"), dirs_exist_ok=True
                )
                shutil.copytree(
                    join(model_dir, "results.json"), join(model_dir, f"results-{timestamp}.json"), dirs_exist_ok=True
                )
            except:
                pass

    if "evaluate" in opts.stages:
        per_results = OrderedDict()
        mean_results = OrderedDict()

        keys = ["psnr", "ssim", "lpips", "gaussians", "RPE_t", "RPE_r", "ATE", "RT"]

        for key in keys:
            mean_results[key] = 0

        # set_trace()
        for scene in sorted(os.listdir(join(outputs, opts.experiment))):
            if opts.suffix is None:
                model_dir = join(outputs, opts.experiment, scene, opts.model)
            else:
                model_dir = join(outputs, opts.experiment, scene, f"{opts.model}-{opts.suffix}")
            if not os.path.exists(join(model_dir, "results.json")):
                continue
            if scene in scenes:
                per_results[scene] = OrderedDict()
                with open(join(model_dir, "results.json"), "r") as f:
                    meta = json.load(f)
                for key in keys:
                    per_results[scene][key] = meta["results"].get(key, 0)
                    mean_results[key] += meta["results"].get(key, 0)

        for key in keys:
            mean_results[key] /= len(per_results)

        if len(per_results) > 0:
            all_results = OrderedDict([("means", mean_results)] + sorted(per_results.items(), key=lambda t: t[0]))
        else:
            all_results = OrderedDict()

        if opts.suffix is None:
            json_fn = join(outputs, opts.experiment, f"{opts.model}-metrics.json")
        else:
            json_fn = join(outputs, opts.experiment, f"{opts.model}-{opts.suffix}-metrics.json")

        with open(json_fn, "w") as f:
            json.dump(all_results, f, indent=4)

        print(json.dumps(all_results, indent=4))


if __name__ == "__main__":

    pipeline(scenes, args)
