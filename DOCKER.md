# Docker Setup for bsort

This guide explains how to use Docker with the bsort CLI tool.

## Files Created

- `Dockerfile` - CPU-only version (recommended for inference)
- `Dockerfile.gpu` - GPU-enabled version (for training with CUDA)
- `docker-compose.yml` - Simplified Docker workflow
- `.dockerignore` - Optimizes build by excluding unnecessary files

## Quick Start

### Option 1: Using Docker directly

**Build the image:**
```bash
docker build -t bsort:latest .
```

**Run commands:**
```bash
# Show help
docker run bsort:latest

# Train
docker run -v $(pwd)/sample:/data/sample -v $(pwd)/outputs:/outputs \
  bsort:latest train --config /data/sample/config.yaml

# Infer
docker run -v $(pwd)/sample:/data/sample -v $(pwd)/outputs:/outputs \
  bsort:latest infer --config /data/sample/config.yaml --image /data/sample/raw-250110_dc_s001_b2_1.jpg

# Sanity check
docker run -v $(pwd)/sample:/data/sample \
  bsort:latest sanity_check /data/sample --visualize

# Export model
docker run -v $(pwd)/outputs:/outputs \
  bsort:latest export --model /outputs/train/weights/best.pt --output /outputs/best.onnx

# Augment data
docker run -v $(pwd)/sample:/data/sample \
  bsort:latest augment --input-dir /data/sample/train/images --config /data/sample/config.yaml -n 5
```

### Option 2: Using docker-compose

**Build:**
```bash
docker-compose build
```

**Run commands:**
```bash
# Show help
docker-compose run bsort

# Train (using profile)
docker-compose --profile train up bsort-train

# Infer (using profile)
docker-compose --profile infer up bsort-infer

# Custom command
docker-compose run bsort sanity_check /data/sample --visualize
```

## GPU Support

If you have NVIDIA GPU and want to use it for training:

**Build GPU image:**
```bash
docker build -f Dockerfile.gpu -t bsort:gpu .
```

**Run with GPU:**
```bash
docker run --gpus all -v $(pwd)/sample:/data/sample -v $(pwd)/outputs:/outputs \
  bsort:gpu train --config /data/sample/config.yaml
```

**Requirements:**
- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

## Volume Mounts Explained

- `-v $(pwd)/sample:/data/sample` - Mount your sample data directory
- `-v $(pwd)/outputs:/outputs` - Mount outputs directory for trained models
- `-v $(pwd)/models:/models` - Mount models directory for pre-trained weights

## Tips

1. **Permissions**: If you encounter permission issues with output files, add `--user $(id -u):$(id -g)` to docker run
2. **Interactive mode**: Add `-it` flag for interactive terminal: `docker run -it bsort:latest`
3. **Clean up**: Remove unused images with `docker image prune`

## Example Workflow

```bash
# 1. Build image
docker build -t bsort:latest .

# 2. Check your data
docker run -v $(pwd)/sample:/data/sample bsort:latest sanity_check /data/sample -v

# 3. Train model (assuming you have config.yaml)
docker run -v $(pwd)/sample:/data/sample -v $(pwd)/outputs:/outputs \
  bsort:latest train --config /data/sample/config.yaml

# 4. Export to ONNX
docker run -v $(pwd)/outputs:/outputs \
  bsort:latest export --model /outputs/train/weights/best.pt

# 5. Run inference
docker run -v $(pwd)/sample:/data/sample -v $(pwd)/outputs:/outputs \
  bsort:latest infer --config /data/sample/config.yaml --image /data/sample/test.jpg
```
