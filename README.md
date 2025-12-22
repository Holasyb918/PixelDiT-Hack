# PixelDiT: Pixel Diffusion Transformer

[English](#english) | [中文](#中文)

---

## English

### Overview

Unofficial implementation of **PixelDiT** based on the paper:

📄 **[Pixel-level Reasoning Diffusion Model for Multi-Condition Image Generation](https://arxiv.org/abs/2511.20645)**

### Key Features

- **Dual-level Architecture**: Patch-level for global semantics + Pixel-level for texture details
- **Pixel-wise AdaLN**: Per-pixel modulation instead of patch-wise broadcasting
- **Pixel Token Compaction**: Efficient attention via spatial compression
- **MM-DiT Support**: Text-to-image generation with multi-modal attention

### Model Configurations

| Model | N | M | D | D_pix | Params |
|-------|---|---|------|-------|--------|
| PixelDiT-B | 12 | 2 | 768 | 16 | 184M |
| PixelDiT-L | 22 | 4 | 1024 | 16 | 569M |
| PixelDiT-XL | 26 | 4 | 1152 | 16 | 797M |
| PixelDiT-T2I | 14 | 2 | 1536 | 16 | 1311M |

### Quick Start

```python
from pixeldit_model import PixelDiT_B, test_model

# Create model
model = PixelDiT_B(input_size=256, num_classes=1000)

# Or test with parameters
test_model("PixelDiT-B", hidden_size=768, patch_depth=12, pixel_depth=2, num_heads=12)
```

### Requirements

```
torch
einops
timm
numpy
```

### Contact

If you have any questions or suggestions, feel free to reach out!

---

## 中文

### 概述

基于论文的 **PixelDiT** 非官方实现：

📄 **[Pixel-level Reasoning Diffusion Model for Multi-Condition Image Generation](https://arxiv.org/abs/2511.20645)**

### 核心特性

- **双层架构**：Patch 级别处理全局语义 + Pixel 级别处理纹理细节
- **逐像素 AdaLN**：每个像素独立调制，而非 patch 级别广播
- **像素 Token 压缩**：通过空间压缩实现高效注意力
- **MM-DiT 支持**：支持文生图的多模态注意力

### 模型配置

| 模型 | N | M | D | D_pix | 参数量 |
|------|---|---|------|-------|--------|
| PixelDiT-B | 12 | 2 | 768 | 16 | 184M |
| PixelDiT-L | 22 | 4 | 1024 | 16 | 569M |
| PixelDiT-XL | 26 | 4 | 1152 | 16 | 797M |
| PixelDiT-T2I | 14 | 2 | 1536 | 16 | 1311M |

### 快速开始

```python
from pixeldit_model import PixelDiT_B, test_model

# 创建模型
model = PixelDiT_B(input_size=256, num_classes=1000)

# 或使用测试函数
test_model("PixelDiT-B", hidden_size=768, patch_depth=12, pixel_depth=2, num_heads=12)
```

### 依赖

```
torch
einops
timm
numpy
```

### 联系方式

如有任何问题或建议，欢迎随时联系我！

---

## License

MIT

## Citation

```bibtex
@article{pixeldit2024,
  title={Pixel-level Reasoning Diffusion Model for Multi-Condition Image Generation},
  author={...},
  journal={arXiv preprint arXiv:2511.20645},
  year={2024}
}
```
