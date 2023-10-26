# Wonder3D
Single Image to 3D using Cross-Domain Diffusion
## [Paper](https://arxiv.org/abs/2310.15008) | [Project page](https://www.xxlong.site/Wonder3D/) | [Hugging Face Demo](https://huggingface.co/spaces/flamehaze1115/Wonder3D-demo)

![](assets/fig_teaser.png)

Wonder3D reconstructs highly-detailed textured meshes from a single-view image in only 2 ∼ 3 minutes. Wonder3D first generates consistent multi-view normal maps with corresponding color images via a cross-domain diffusion model, and then leverages a novel normal fusion method to achieve fast and high-quality reconstruction.

## Schedule
- [x] Inference code and pretrained models.
- [x] Huggingface demo.
- [ ] New model trained on the whole Objaverse dataset.


### Preparation for inference
1. Install packages in `requirements.txt`. 
```angular2html
conda create -n wonder3d
conda activate wonder3d
pip install -r requirements.txt
```
Install tiny-cuda-nn PyTorch extension for mesh extraction: `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`

2. Download the [checkpoints](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xxlong_connect_hku_hk/EgSHPyJAtaJFpV_BjXM3zXwB-UMIrT4v-sQwGgw-coPtIA) and into the root folder.

### Inference
1. Make sure you have the following models.
```bash
Wonder3D
|-- ckpts
    |-- unet
    |-- scheduler.bin
    ...
```
2. Predict foreground mask as the alpha channel. We use [Clipdrop](https://clipdrop.co/remove-background) to segment the foreground object interactively. 
You may also use `rembg` to remove the backgrounds.
```bash
# !pip install rembg
import rembg
result = rembg.remove(result)
result.show()
```
3. Run Wonder3d to produce multiview-consistent normal maps and color images. Then you can check the results in the folder `./outputs`. (we use rembg to remove backgrounds of the results, but the segmemtations are not always perfect.) 
```bash
accelerate launch --config_file 1gpu.yaml test_mvdiffusion_seq.py \
            --config mvdiffusion-joint-ortho-6views.yaml
```
or 
```bash
bash run_test.sh
```

4. Mesh Extraction
```bash
cd ./instant-nsr-pl
bash run.sh output_folder_path scene_name
```

## Citation
If you find this repository useful in your project, please cite the following work. :)
```
@misc{long2023wonder3d,
      title={Wonder3D: Single Image to 3D using Cross-Domain Diffusion}, 
      author={Xiaoxiao Long and Yuan-Chen Guo and Cheng Lin and Yuan Liu and Zhiyang Dou and Lingjie Liu and Yuexin Ma and Song-Hai Zhang and Marc Habermann and Christian Theobalt and Wenping Wang},
      year={2023},
      eprint={2310.15008},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
