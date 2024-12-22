# multiprocessing render
import json
import multiprocessing
import subprocess
from dataclasses import dataclass
from typing import Optional
import os

import boto3

import argparse

parser = argparse.ArgumentParser(description='distributed rendering')

parser.add_argument('--workers_per_gpu', type=int,
                    help='number of workers per gpu.')
parser.add_argument('--input_models_path', type=str,
                    help='Path to a json file containing a list of 3D object files.')
parser.add_argument('--upload_to_s3', type=bool, default=False,
                    help='Whether to upload the rendered images to S3.')
parser.add_argument('--log_to_wandb', type=bool, default=False,
                    help='Whether to log the progress to wandb.')
parser.add_argument('--num_gpus', type=int, default=-1,
                    help='number of gpus to use. -1 means all available gpus.')
parser.add_argument('--gpu_list',nargs='+', type=int, 
                    help='the avalaible gpus')

parser.add_argument('--mode', type=str, default='render', 
                choices=['render_ortho', 'render_persp'],
                    help='use orthogonal camera or perspective camera')

parser.add_argument('--start_i', type=int, default=0,
                    help='the index of first object to be rendered.')

parser.add_argument('--end_i', type=int, default=-1,
                    help='the index of the last object to be rendered.')

parser.add_argument('--objaverse_root', type=str, default='/ghome/l5/xxlong/.objaverse/hf-objaverse-v1',
                    help='Path to a json file containing a list of 3D object files.')

parser.add_argument('--save_folder', type=str, default=None,
                    help='Path to a json file containing a list of 3D object files.')

parser.add_argument('--blender_install_path', type=str, default=None,
                    help='blender path.')

parser.add_argument('--view_idx', type=int, default=2,
                    help='the number of render views.')

parser.add_argument('--ortho_scale', type=float, default=1.25,
                    help='ortho rendering usage; how large the object is')

parser.add_argument('--random_pose', action='store_true',
                    help='whether randomly rotate the poses to be rendered')

args = parser.parse_args()


view_idx = args.view_idx

VIEWS = ["front", "back", "right", "left", "front_right", "front_left", "back_right", "back_left"]

def check_task_finish(render_dir, view_index):
    print(">>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<")
    print('check into', render_dir)
    files_type = ['rgb', 'normals']
    flag = True
    view_index = "%03d" % view_index
    if os.path.exists(render_dir):
        for t in files_type:
            for face in VIEWS:
                fpath = os.path.join(render_dir, f'{t}_{view_index}_{face}.png')
                # print(fpath)
                if not os.path.exists(fpath):
                    flag = False
    else:
        flag = False

    return flag

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    s3: Optional[boto3.client],
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        view_path = os.path.join(args.save_folder, item.split('/')[-1][:2], item.split('/')[-1][:-4])
        # print("view_path:",view_path)
        if 'render' in args.mode:
            print('view_point:', view_path, check_task_finish(view_path, view_idx))
            if check_task_finish(view_path, view_idx):
                queue.task_done()
                print('========', item, 'rendered', '========')

                continue
            else:
                os.makedirs(view_path, exist_ok = True)

        # Perform some operation on the item
        print(item, gpu)

        if args.mode == 'render_ortho':
            command = (
                f" CUDA_VISIBLE_DEVICES={gpu} "
                f" blenderproc run --custom-blender-path {args.blender_install_path} blenderProc_ortho.py"
                f" --object_path {item} --view {view_idx}"
                f" --output_folder {args.save_folder}"
                f" --ortho_scale {args.ortho_scale} "
            )
            if args.random_pose:
                print("random pose to render")
                command += f" --random_pose"
        elif args.mode == 'render_persp':
            command = (
                f" CUDA_VISIBLE_DEVICES={gpu} "
                f" blenderproc run --custom-blender-path {args.blender_install_path} blenderProc_persp.py"
                f" --object_path {item} --view {view_idx}"
                f" --output_folder {args.save_folder}"
            )
            if args.random_pose:
                print("random pose to render")
                command += f" --random_pose"

        print(command)
        # subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()

import gzip
import urllib
def _load_object_paths() -> dict[str, str]:
    """Load the object paths from the dataset.

    The object paths specify the location of where the object is located
    in the Hugging Face repo.

    Returns:
        A dictionary mapping the uid to the object path.
    """
    object_paths_file = "object-paths.json.gz"
    local_path = "~/.objaverse/hf-objaverse-v1/object-paths.json.gz"
    if not os.path.exists(local_path):
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
        # wget the file and put it in local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        object_paths = json.load(f)
    return object_paths


if __name__ == "__main__":
    # args = tyro.cli(Args)

    s3 = boto3.client("s3") if args.upload_to_s3 else None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start worker processes on each of the GPUs

    for worker_i in range(args.workers_per_gpu):
        for gpu_i in range(args.num_gpus):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, args.gpu_list[gpu_i], s3)
            )
            process.daemon = True
            process.start()
        
    # Add items to the queue
    if args.input_models_path is not None:
        with open(args.input_models_path, "r") as f:
            model_paths = json.load(f)

    args.end_i = len(model_paths) if args.end_i > len(model_paths) else args.end_i
    # multiprocessing render
    import json
    import multiprocessing
    import subprocess
    from dataclasses import dataclass
    from typing import Optional
    import os

    import boto3

    import argparse

    parser = argparse.ArgumentParser(description='distributed rendering')

    parser.add_argument('--workers_per_gpu', type=int,
                        help='number of workers per gpu.')
    parser.add_argument('--input_models_path', type=str,
                        help='Path to a json file containing a list of 3D object files.')
    parser.add_argument('--upload_to_s3', type=bool, default=False,
                        help='Whether to upload the rendered images to S3.')
    parser.add_argument('--log_to_wandb', type=bool, default=False,
                        help='Whether to log the progress to wandb.')
    parser.add_argument('--num_gpus', type=int, default=-1,
                        help='number of gpus to use. -1 means all available gpus.')
    parser.add_argument('--gpu_list', nargs='+', type=int,
                        help='the avalaible gpus')

    parser.add_argument('--mode', type=str, default='render',
                        choices=['render_ortho', 'render_persp'],
                        help='use orthogonal camera or perspective camera')

    parser.add_argument('--start_i', type=int, default=0,
                        help='the index of first object to be rendered.')

    parser.add_argument('--end_i', type=int, default=-1,
                        help='the index of the last object to be rendered.')

    parser.add_argument('--objaverse_root', type=str, default='/ghome/l5/xxlong/.objaverse/hf-objaverse-v1',
                        help='Path to a json file containing a list of 3D object files.')

    parser.add_argument('--save_folder', type=str, default=None,
                        help='Path to a json file containing a list of 3D object files.')

    parser.add_argument('--blender_install_path', type=str, default=None,
                        help='blender path.')

    parser.add_argument('--view_idx', type=int, default=2,
                        help='the number of render views.')

    parser.add_argument('--ortho_scale', type=float, default=1.25,
                        help='ortho rendering usage; how large the object is')

    parser.add_argument('--random_pose', action='store_true',
                        help='whether randomly rotate the poses to be rendered')

    args = parser.parse_args()

    view_idx = args.view_idx

    VIEWS = ["front", "back", "right", "left", "front_right", "front_left", "back_right", "back_left"]


    def check_task_finish(render_dir, view_index):
        print(">>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<")
        print('check into', render_dir)
        files_type = ['rgb', 'normals']
        flag = True
        view_index = "%03d" % view_index
        if os.path.exists(render_dir):
            for t in files_type:
                for face in VIEWS:
                    fpath = os.path.join(render_dir, f'{t}_{view_index}_{face}.png')
                    # print(fpath)
                    if not os.path.exists(fpath):
                        flag = False
        else:
            flag = False

        return flag


    def worker(
            queue: multiprocessing.JoinableQueue,
            count: multiprocessing.Value,
            gpu: int,
            s3: Optional[boto3.client],
    ) -> None:
        while True:
            item = queue.get()
            if item is None:
                break

            view_path = os.path.join(args.save_folder, item.split('/')[-1][:2], item.split('/')[-1][:-4])
            # print(view_path)
            if 'render' in args.mode:
                print('view_point:', view_path, check_task_finish(view_path, view_idx))
                if check_task_finish(view_path, view_idx):
                    queue.task_done()
                    print('========', item, 'rendered', '========')

                    continue
                else:
                    os.makedirs(view_path, exist_ok=True)

            # Perform some operation on the item
            print(item, gpu)

            if args.mode == 'render_ortho':
                command = (
                    f" CUDA_VISIBLE_DEVICES={gpu} "
                    f" blenderproc run --custom-blender-path {args.blender_install_path} blenderProc_ortho.py"
                    f" --object_path {item} --view {view_idx}"
                    f" --output_folder {args.save_folder}"
                    f" --ortho_scale {args.ortho_scale} "
                )
                if args.random_pose:
                    print("random pose to render")
                    command += f" --random_pose"
            elif args.mode == 'render_persp':
                command = (
                    f" CUDA_VISIBLE_DEVICES={gpu} "
                    f" blenderproc run --custom-blender-path {args.blender_install_path} blenderProc_persp.py"
                    f" --object_path {item} --view {view_idx}"
                    f" --output_folder {args.save_folder}"
                )
                if args.random_pose:
                    print("random pose to render")
                    command += f" --random_pose"

            print(command)
            subprocess.run(command, shell=True)

            with count.get_lock():
                count.value += 1

            queue.task_done()


    import gzip
    import urllib


    def _load_object_paths() -> dict[str, str]:
        """Load the object paths from the dataset.

        The object paths specify the location of where the object is located
        in the Hugging Face repo.

        Returns:
            A dictionary mapping the uid to the object path.
        """
        object_paths_file = "object-paths.json.gz"
        local_path = "~/.objaverse/hf-objaverse-v1/object-paths.json.gz"
        if not os.path.exists(local_path):
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
            # wget the file and put it in local_path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(hf_url, local_path)
        with gzip.open(local_path, "rb") as f:
            object_paths = json.load(f)
        return object_paths


    if __name__ == "__main__":
        # args = tyro.cli(Args)

        s3 = boto3.client("s3") if args.upload_to_s3 else None
        queue = multiprocessing.JoinableQueue()
        count = multiprocessing.Value("i", 0)

        # Start worker processes on each of the GPUs

        for worker_i in range(args.workers_per_gpu):
            for gpu_i in range(args.num_gpus):
                worker_i = gpu_i * args.workers_per_gpu + worker_i
                process = multiprocessing.Process(
                    target=worker, args=(queue, count, args.gpu_list[gpu_i], s3)
                )
                process.daemon = True
                process.start()

        # Add items to the queue
        if args.input_models_path is not None:
            with open(args.input_models_path, "r") as f:
                model_paths = json.load(f)

        args.end_i = len(model_paths) if args.end_i > len(model_paths) else args.end_i

        for item in model_paths[args.start_i:args.end_i]:

            if os.path.exists(os.path.join(args.objaverse_root, os.path.basename(item))):
                obj_path = os.path.join(args.objaverse_root, os.path.basename(item))
            elif os.path.exists(os.path.join(args.objaverse_root, item)):
                obj_path = os.path.join(args.objaverse_root, item)
            # else:
            #     obj_path = os.path.join(args.objaverse_root, item[:2], item+".glb")
            else:
                object_paths = _load_object_paths()
                uid = item
                if uid.endswith(".glb"):
                    uid = uid[:-4]
                if uid not in object_paths:
                    # warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                    continue
                obj_path = object_paths[uid]
                obj_path = os.path.join(args.objaverse_root, obj_path)
                # print("object_at:", obj_path)
            queue.put(obj_path)

        # Wait for all tasks to be completed
        queue.join()

        # Add sentinels to the queue to stop the worker processes
        for i in range(args.num_gpus * args.workers_per_gpu):
            queue.put(None)

    for item in model_paths[args.start_i:args.end_i]:

        if os.path.exists(os.path.join(args.objaverse_root, os.path.basename(item))):
            obj_path = os.path.join(args.objaverse_root, os.path.basename(item))
        elif os.path.exists(os.path.join(args.objaverse_root, item)):
            obj_path = os.path.join(args.objaverse_root, item)
        # else:
        #     obj_path = os.path.join(args.objaverse_root, item[:2], item+".glb")
        else:
            object_paths = _load_object_paths()
            uid = item
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                # warnings.warn(f"Could not find object with uid {uid}. Skipping it.")
                continue
            obj_path = object_paths[uid]
            obj_path = os.path.join(args.objaverse_root, obj_path)
            # print("object_at:", obj_path)
        queue.put(obj_path)

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)
