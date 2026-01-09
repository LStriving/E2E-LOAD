cd lyr/E2E-Load
# conda activate e2e-load
# check if model file exists
cd ckpt
if [ ! -f MViTv2_S_16x4_k400_f302660347.pyth ]; then
    echo "Downloading MViTv2_S_16x4_k400_f302660347.pyth model..."
    wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth
else
    echo "Model file already exists. Skipping download."
fi
cd ..

if [ ! -f ../.env ]; then
    echo ".env file not found!"
    echo "Please create a .env file based on .env.example and add your Hugging Face token."
    exit 1
fi
# load .env file
source .env
mkdir -p ./data/Surgery/videos
hf download LStriving/eyes-surgical-videos --repo-type dataset --local-dir ./data/Surgery/videos --token $token 

