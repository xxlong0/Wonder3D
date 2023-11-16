# Wonder3D - Windows
Single Image to 3D using Cross-Domain Diffusion

**We have received numerous requests from non-professionals using Window who are eager to try out our project. 
As a result, we have provided a Windows version and have included detailed setup instructions.**

## Set Up Deep Learning Environment

### Install Conda environment
(If you have Conda installed, skip this step.)

1. Download [Anaconda](https://www.anaconda.com/download#downloads).
2. Install Anaconda using the downloaded installer.
3. Enter the Anaconda PowerShell Prompt


### Install CUDA Toolkit
(If you have CUDA installed, skip this step.)

1. Use  ``nvidia-smi`` to check the version of your CUDA driver. 
2. Download the corresponding [cuda tookit installer](https://developer.nvidia.com/cuda-toolkit-archive) based your driver version.
3. Install CUDA using the downloaded exe file (Default settings are recommended).
4. Confirm that CUDA_PATH is successfully added into System Environment Variable (Done automatically if installation succeeds).

### Install PyTorch 
Find the correct pytorch installation commands [here](https://pytorch.org/get-started/previous-versions/) based on your CUDA version.


Here is an example:
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Set up Wonder3D
1. Install Git.

If you do not have git installed, use ```conda install git``` to install git first.

2. Clone the repository.
```bash
git clone https://github.com/xxlong0/Wonder3D.git -b main-windows
```

3. Install the dependencies.
```bash
cd Wonder3D
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

4. Download the weights of pretrained models.
* Download the [checkpoint](https://connecthkuhk-my.sharepoint.com/personal/xxlong_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxxlong%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fwonder3d%2Fpretrained%2Dweights%2Fckpts&ga=1) of Wonder3D. 
Put it to the ``ckpts`` folder.
```bash
Wonder3D
|-- ckpts
    |-- unet
    |-- scheduler.bin
    ...
```
* Download the [SAM](https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth) model. Put it to the ``sam_pt`` folder.
```
Wonder3D
|-- sam_pt
    |-- sam_vit_h_4b8939.pth
```

5. Run Wonder3D
* Generate multi-view images and normals
```bash
bash run_test.sh
```

Or you can play with the interactive Gradio demo
```
python gradio_app.py
```
Enter the resulting address with your browser and play with the app. You can then download the results.


* Generate mesh
```bash
cd ./instant-nsr-pl
bash run.sh output_folder_path scene_name
```

## Frequently Encountered Issues on Windows.
1. OSError: [Errno 22] Invalid argument | Python\lib\multiprocessing\reduction.py", line 60, in dump
ForkingPickler(file, protocol).dump(obj)

Reduce the num_workers in `instant-nsr-pl/datasets/ortho.py`
```angular2html
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), # set it to a smaller one
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
```

2. subprocess.CalledProcessError: Command ‘[where, cl]‘ returned non-zero exit status 1

You need to download Visual Studio on your windows, and then add the path of the folder that contains ``cl.exe`` to  Windows PATH Environment Variable.
For example:
``` 
C:\XXX\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.28.29333\bin\Hostx64\x64
```