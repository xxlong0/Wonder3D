# Docker setup

This docker setup is tested on Ubunu20.04.

make sure you are under directory yourworkspace/Wonder3D/

run

`docker build --no-cache -t wonder3d/deploy:cuda11.7 -f docker/Dockerfile .`

then run 

`docker run --gpus all -it wonder3d/deploy:cuda11.7 bash`


## Nvidia Container Toolkit setup

You will have trouble enabling gpu for docker if you haven't installed **NVIDIA Container Toolkit** on you local machine before. You can skip this section if you have already installed it. Follow the instruction in this website to install it. 

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

or you can run the following command to install it with apt:

1.Configure the production repository:
   
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

2.Update the packages list from the repository:

`sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list`

3.Install the NVIDIA Container Toolkit packages:

`sudo apt-get install -y nvidia-container-toolkit`

Remember to restart the docker:

`sudo systemctl restart docker`

now you can run the following command:

`docker run --gpus all -it wonder3d/deploy:cuda11.7 bash`


## Install Tiny Cudann

After you start the container, run the following command to install tiny cudann. Somehow this pip installation can not be done during the docker build, so you have to do it manually after the docker is started.

`pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`


Now you should be good to go, good luck and have fun :)
