<div align="center">

# L2P: Unlocking Latent Potential for Pixel Generation

<p>
  <a href="https://nju-pcalab.github.io/projects/L2P/">
    <img src="https://img.shields.io/badge/Project-Page-1f72ff?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2605.12013">
    <img src="https://img.shields.io/badge/arXiv-2605.12013-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://huggingface.co/datasets/zhen-nan/L2P-dataset">
    <img src="https://img.shields.io/badge/Dataset-L2P-ffcc4d?style=for-the-badge&logo=huggingface&logoColor=white" alt="Dataset">
  </a>
  <a href="https://huggingface.co/spaces/multimodalart/z-image-6b-pixel-space">
    <img src="https://img.shields.io/badge/HF-Space-ff9d00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HF Space">
  </a>
</p>

<p>
  <em>An efficient transfer paradigm enabling high-quality, end-to-end pixel-space diffusion with minimal computational overhead and data requirements.</em>
</p>

<div align="center">
<sub>⭐ If L2P helps your research or product, please consider giving the repo a star ⭐</sub>
</div>


</div>

---

## 📰 News

- **\[2026.05.12\]** Technical report released.
- **\[2026.05.22\]** 1K-resolution training code, inference code, weights, and dataset released.
- **\[2026.05.23\]** Online [demo](https://huggingface.co/spaces/multimodalart/z-image-6b-pixel-space). (Thanks to [multimodalart](https://huggingface.co/multimodalart) for the support!)
- **\[2026.07.11\]** 4K/8K/10K UHR training code, inference code, weights, and dataset released (see  [L2P-ZImage-HR](https://github.com/HaojunChen663/PixVerve-95K/tree/main/L2P-ZImage-HR)).
---

## 🗺️ Roadmap

| Status | Item |
| :---: | :--- |
| ✅ | 1K inference code & weights |
| ✅ | Training code |
| ✅ | 4K/8K/10K UHR generation |
| 🛠️ | Compatibility with more LDM model|

---

## 📦 Installation

```bash
git clone https://github.com/TencentYoutuResearch/T2I-L2P.git
cd T2I-L2P
pip install -e .
```


---



## 🎨 Inference
Checkpoint:
| Model | Params | HuggingFace |
|-------|--------|-------------|
|L2P-z-image (1k resolution)       |6B        |[🤗](https://huggingface.co/zhen-nan/L2P)              |

```python
import torch
from diffsynth.pipelines.z_image_L2P import ZImagePipeline, ModelConfig

main_model_path = "/path/model-1k-merge.safetensors"

text_encoder_paths = [
    "/path/Z-Image-Turbo/text_encoder/model-00001-of-00003.safetensors",
    "/path/Z-Image-Turbo/text_encoder/model-00002-of-00003.safetensors",
    "/path/Z-Image-Turbo/text_encoder/model-00003-of-00003.safetensors",
]

tokenizer_path = "/path/Z-Image-Turbo/tokenizer"

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[main_model_path]),
        ModelConfig(path=text_encoder_paths),
    ],
    tokenizer_config=ModelConfig(path=tokenizer_path),
)

prompt = "an origami pig on fire in the middle of a dark room with a pentagram on the floor"

image = pipe(
    prompt=prompt,
    seed=42,
    rand_device="cuda",
    num_inference_steps=30,
    cfg_scale=2.0,
    height=1024,
    width=1024,
)

image.save("example.png")
```

### Gradio Demo

First, install gradio:

```bash
pip install gradio
```

Launch a multi-GPU web UI:

```bash
python app.py
```

The demo auto-detects free GPUs, dispatches each request to an idle device, and exposes a Gradio interface at `http://0.0.0.0:23231`.

---

## 🏋️ Training

The full training pipeline consists of four steps:
**(1)** prepare the Z-Image base weights → **(2)** convert them into a pixel-space initialization → **(3)** launch training → **(4)** merge the trained delta back with the pixel-init weights for inference.

### Step 1 · Prepare Z-Image weights

Download the official **Z-Image-Turbo** checkpoint from Hugging Face:

- 🤗 [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)



### Step 2 · Offline weight conversion (latent → pixel init)

Convert the latent-space DiT weights into a **pixel-space initialization** that L2P can fine-tune from:

```bash
python examples/z_image/L2P_convert_weight.py \
  --latent_ckpt_files \
    /path/to/Z-Image-Turbo/transformer/diffusion_pytorch_model-00001-of-00003.safetensors \
    /path/to/Z-Image-Turbo/transformer/diffusion_pytorch_model-00002-of-00003.safetensors \
    /path/to/Z-Image-Turbo/transformer/diffusion_pytorch_model-00003-of-00003.safetensors \
  --output_path ./pretrain_weight/Z-Image-Pixel-Init/diffusion_pytorch_model.safetensors
```

### Step 3 · Launch training

**Standard training** :

```bash
bash train_run.sh
```

**Low-VRAM training** (single GPU < 24 GB VRAM):

```bash
bash train_run_low_VRAM.sh
```

#### Dataset format

Provide a directory of images plus a CSV metadata file:

```
data/
├── images/                # raw image folder
└── metadata.csv           # columns: file_name, text, ...
```

### Step 4 · Offline weight merge (for inference)

```bash
python merge_weights.py \
  --file_a ./models/train/L2P_Standard/step-xxx.safetensors \
  --file_b ./pretrain_weight/Z-Image-Pixel-Init/diffusion_pytorch_model.safetensors \
  --file_out ./models/train/L2P_Standard/model-merge.safetensors
```

- `--file_a`: trained checkpoint from Step 3
- `--file_b`: pixel-init weights from Step 2
- `--file_out`: merged single-file weight
---

## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@article{chen2026l2p,
  title   = {L2P: Unlocking Latent Potential for Pixel Generation},
  author  = {Chen, Zhennan and Zhu, Junwei and Chen, Xu and Zhang, Jiangning and
             Chen, Jiawei and Zeng, Zhuoqi and Zhang, Wei and Wang, Chengjie and
             Yang, Jian and Tai, Ying},
  journal = {arXiv preprint arXiv:2605.12013},
  year    = {2026}
}

@article{chen2025dip,
  title   = {DiP: Taming Diffusion Models in Pixel Space},
  author  = {Chen, Zhennan and Zhu, Junwei and Chen, Xu and Zhang, Jiangning and
             Hu, Xiaobin and Zhao, Hanzhen and Wang, Chengjie and Yang, Jian and
             Tai, Ying},
  journal = {arXiv preprint arXiv:2511.18822},
  year    = {2025}
}
```

---

## 🙏 Acknowledgements

L2P is built upon the excellent open-source work of
[**DiffSynth-Studio**](https://github.com/modelscope/DiffSynth-Studio),
[**Z-Image**](https://github.com/Tongyi-MAI/Z-Image).

