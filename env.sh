# git clone https://github.com/LStriving/E2E-LOAD.git
git clone https://github.com/LStriving/slowfast.git
git clone https://github.com/facebookresearch/detectron2 
# env
conda create -n e2e-load python=3.10 -y
conda activate e2e-load
python -m pip install -r --no-cache-dir E2E-LOAD/requirements.txt
python -m pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0 \
   --index-url https://download.pytorch.org/whl/cu117
python -m pip install 'git+https://github.com/facebookresearch/fairscale'
python -m pip install "git+https://github.com/facebookresearch/pytorchvideo.git"
python -m pip install 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
conda install ffmpeg==4.2.2 -y

python -m pip install -e ./detectron2
python -m pip install -e ./slowfast
cd slowfast
python setup.py build develop
cd ..

cd E2E-LOAD
# create env file
cp .env.example .env

# set env variable to ~/.bashrc (slowfast/slowfast E2E-LOAD)
# echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)/slowfast:$(pwd)/E2E-LOAD" >> ~/.bashrc
# source ~/.bashrc
