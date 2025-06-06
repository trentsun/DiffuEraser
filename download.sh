# 创建权重目录
mkdir -p weights

# 下载 DiffuEraser 模型
huggingface-cli download lixiaowen/diffuEraser --local-dir ./weights/diffuEraser

# 下载 Stable Diffusion v1.5 (仅必要文件)
huggingface-cli download runwayml/stable-diffusion-v1-5 feature_extractor --local-dir ./weights/stable-diffusion-v1-5
huggingface-cli download runwayml/stable-diffusion-v1-5 model_index.json --local-dir ./weights/stable-diffusion-v1-5
huggingface-cli download runwayml/stable-diffusion-v1-5 safety_checker --local-dir ./weights/stable-diffusion-v1-5
huggingface-cli download runwayml/stable-diffusion-v1-5 scheduler --local-dir ./weights/stable-diffusion-v1-5
huggingface-cli download runwayml/stable-diffusion-v1-5 text_encoder --local-dir ./weights/stable-diffusion-v1-5
huggingface-cli download runwayml/stable-diffusion-v1-5 tokenizer --local-dir ./weights/stable-diffusion-v1-5

# 下载 PCM Weights
huggingface-cli download wangfuyun/PCM_Weights --local-dir ./weights/PCM_Weights

# 下载 SD-VAE-FT-MSE
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir ./weights/sd-vae-ft-mse

# 下载 Motion Adapter (可选,用于训练)
huggingface-cli download guoyww/animatediff-motion-adapter-v1-5-2 --local-dir ./weights/animatediff-motion-adapter-v1-5-2