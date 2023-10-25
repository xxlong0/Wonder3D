# Instant Neural Surface Reconstruction

This repository contains a concise and extensible implementation of NeRF and NeuS for neural surface reconstruction based on Instant-NGP and the Pytorch-Lightning framework. **Training on a NeRF-Synthetic scene takes ~5min for NeRF and ~10min for NeuS on a single RTX3090.**

||NeRF in 5min|NeuS in 10 min|
|---|---|---|
|Rendering|![rendering-nerf](https://user-images.githubusercontent.com/19284678/199078178-b719676b-7e60-47f1-813b-c0b533f5480d.png)|![rendering-neus](https://user-images.githubusercontent.com/19284678/199078300-ebcf249d-b05e-431f-b035-da354705d8db.png)|
|Mesh|![mesh-nerf](https://user-images.githubusercontent.com/19284678/199078661-b5cd569a-c22b-4220-9c11-d5fd13a52fb8.png)|![mesh-neus](https://user-images.githubusercontent.com/19284678/199078481-164e36a6-6d55-45cc-aaf3-795a114e4a38.png)|


## Features
**This repository aims to provide a highly efficient while customizable boilerplate for research projects based on NeRF or NeuS.**

- acceleration techniques from [Instant-NGP](https://github.com/NVlabs/instant-ngp): multiresolution hash encoding and fully fused networks by [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn), occupancy grid pruning and rendering by [nerfacc](https://github.com/KAIR-BAIR/nerfacc)
- out-of-the-box multi-GPU and mixed precision training by [PyTorch-Lightning](https://github.com/Lightning-AI/lightning)
- hierarchical project layout that is designed to be easily customized and extended, flexible experiment configuration by [OmegaConf](https://github.com/omry/omegaconf)

**Please subscribe to [#26](https://github.com/bennyguo/instant-nsr-pl/issues/26) for our latest findings on quality improvements!**

## News

🔥🔥🔥 Check out my new project on 3D content generation: https://github.com/threestudio-project/threestudio 🔥🔥🔥

- 06/03/2023: Add an implementation of [Neuralangelo](https://research.nvidia.com/labs/dir/neuralangelo/). See [here](https://github.com/bennyguo/instant-nsr-pl#training-on-DTU) for details.
- 03/31/2023: NeuS model now supports background modeling. You could try on the DTU dataset provided by [NeuS](https://drive.google.com/drive/folders/1Nlzejs4mfPuJYORLbDEUDWlc9IZIbU0C?usp=sharing) or [IDR](https://www.dropbox.com/sh/5tam07ai8ch90pf/AADniBT3dmAexvm_J1oL__uoa) following [the instruction here](https://github.com/bennyguo/instant-nsr-pl#training-on-DTU).
- 02/11/2023: NeRF model now supports unbounded 360 scenes with learned background. You could try on [MipNeRF 360 data](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) following [the COLMAP configuration](https://github.com/bennyguo/instant-nsr-pl#training-on-custom-colmap-data).

## Requirements
**Note:**
- To utilize multiresolution hash encoding or fully fused networks provided by tiny-cuda-nn, you should have least an RTX 2080Ti, see [https://github.com/NVlabs/tiny-cuda-nn#requirements](https://github.com/NVlabs/tiny-cuda-nn#requirements) for more details.
- Multi-GPU training is currently not supported on Windows (see [#4](https://github.com/bennyguo/instant-nsr-pl/issues/4)).
### Environments
- Install PyTorch>=1.10 [here](https://pytorch.org/get-started/locally/) based the package management tool you used and your cuda version (older PyTorch versions may work but have not been tested)
- Install tiny-cuda-nn PyTorch extension: `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
- `pip install -r requirements.txt`


## Run
### Training on NeRF-Synthetic
Download the NeRF-Synthetic data [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and put it under `load/`. The file structure should be like `load/nerf_synthetic/lego`.

Run the launch script with `--train`, specifying the config file, the GPU(s) to be used (GPU 0 will be used by default), and the scene name:
```bash
# train NeRF
python launch.py --config configs/nerf-blender.yaml --gpu 0 --train dataset.scene=lego tag=example

# train NeuS with mask
python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=lego tag=example
# train NeuS without mask
python launch.py --config configs/neus-blender.yaml --gpu 0 --train dataset.scene=lego tag=example system.loss.lambda_mask=0.0
```
The code snapshots, checkpoints and experiment outputs are saved to `exp/[name]/[tag]@[timestamp]`, and tensorboard logs can be found at `runs/[name]/[tag]@[timestamp]`. You can change any configuration in the YAML file by specifying arguments without `--`, for example:
```bash
python launch.py --config configs/nerf-blender.yaml --gpu 0 --train dataset.scene=lego tag=iter50k seed=0 trainer.max_steps=50000
```
### Training on DTU
Download preprocessed DTU data provided by [NeuS](https://drive.google.com/drive/folders/1Nlzejs4mfPuJYORLbDEUDWlc9IZIbU0C?usp=sharing) or [IDR](https://www.dropbox.com/sh/5tam07ai8ch90pf/AADniBT3dmAexvm_J1oL__uoa). In the provided config files we assume using NeuS DTU data. If you are using IDR DTU data, please set `dataset.cameras_file=cameras.npz`. You may also need to adjust `dataset.root_dir` to point to your downloaded data location.
```bash
# train NeuS on DTU without mask
python launch.py --config configs/neus-dtu.yaml --gpu 0 --train
# train NeuS on DTU with mask
python launch.py --config configs/neus-dtu-wmask.yaml --gpu 0 --train
# train NeuS on DTU with mask using tricks from Neuralangelo (experimental)
python launch.py --config configs/neuralangelo-dtu-wmask.yaml --gpu 0 --train
```
Notes:
- PSNR in the testing stage is meaningless, as we simply compare to pure white images in testing.
- The results of Neuralangelo can't reach those in the original paper. Some potential improvements: more iterations; larger `system.geometry.xyz_encoding_config.update_steps`; larger `system.geometry.xyz_encoding_config.n_features_per_level`; larger `system.geometry.xyz_encoding_config.log2_hashmap_size`; adopting curvature loss.

### Training on Custom COLMAP Data
To get COLMAP data from custom images, you should have COLMAP installed (see [here](https://colmap.github.io/install.html) for installation instructions). Then put your images in the `images/` folder, and run `scripts/imgs2poses.py` specifying the path containing the `images/` folder. For example:
```bash
python scripts/imgs2poses.py ./load/bmvs_dog # images are in ./load/bmvs_dog/images
```
Existing data following this file structure also works as long as images are store in `images/` and there is a `sparse/` folder for the COLMAP output, for example [the data provided by MipNeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip). An optional `masks/` folder could be provided for object mask supervision. To train on COLMAP data, please refer to the example config files `config/*-colmap.yaml`. Some notes:
- Adapt the `root_dir` and `img_wh` (or `img_downscale`) option in the config file to your data;
- The scene is normalized so that cameras have a minimum distance `1.0` to the center of the scene. Setting `model.radius=1.0` works in most cases. If not, try setting a smaller radius that wraps tightly to your foreground object.
- There are three choices to determine the scene center: `dataset.center_est_method=camera` uses the center of all camera positions as the scene center; `dataset.center_est_method=lookat` assumes the cameras are looking at the same point and calculates an approximate look-at point as the scene center; `dataset.center_est_method=point` uses the center of all points (reconstructed by COLMAP) that are bounded by cameras as the scene center. Please choose an appropriate method according to your capture.
- PSNR in the testing stage is meaningless, as we simply compare to pure white images in testing.

### Testing
The training procedure are by default followed by testing, which computes metrics on test data, generates animations and exports the geometry as triangular meshes. If you want to do testing alone, just resume the pretrained model and replace `--train` with `--test`, for example:
```bash
python launch.py --config path/to/your/exp/config/parsed.yaml --resume path/to/your/exp/ckpt/epoch=0-step=20000.ckpt --gpu 0 --test
```


## Benchmarks
All experiments are conducted on a single NVIDIA RTX3090.

|PSNR|Chair|Drums|Ficus|Hotdog|Lego|Materials|Mic|Ship|Avg.|
|---|---|---|---|---|---|---|---|---|---|
|NeRF Paper|33.00|25.01|30.13|36.18|32.54|29.62|32.91|28.65|31.01|
|NeRF Ours (20k)|34.80|26.04|33.89|37.42|35.33|29.46|35.22|31.17|32.92|
|NeuS Ours (20k, with masks)|34.04|25.26|32.47|35.94|33.78|27.67|33.43|29.50|31.51|

|Training Time (mm:ss)|Chair|Drums|Ficus|Hotdog|Lego|Materials|Mic|Ship|Avg.|
|---|---|---|---|---|---|---|---|---|---|
|NeRF Ours (20k)|04:34|04:35|04:18|04:46|04:39|04:35|04:26|05:41|04:42|
|NeuS Ours (20k, with masks)|11:25|10:34|09:51|12:11|11:37|11:46|09:59|16:25|11:44|


## TODO
- [✅] Support more dataset formats, like COLMAP outputs and DTU
- [✅] Support simple background model
- [ ] Support GUI training and interaction
- [ ] More illustrations about the framework

## Related Projects
- [ngp_pl](https://github.com/kwea123/ngp_pl): Great Instant-NGP implementation in PyTorch-Lightning! Background model and GUI supported.
- [Instant-NSR](https://github.com/zhaofuq/Instant-NSR): NeuS implementation using multiresolution hash encoding.

## Citation
If you find this codebase useful, please consider citing:
```
@misc{instant-nsr-pl,
    Author = {Yuan-Chen Guo},
    Year = {2022},
    Note = {https://github.com/bennyguo/instant-nsr-pl},
    Title = {Instant Neural Surface Reconstruction}
}
```
