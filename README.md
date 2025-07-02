# 2025_GOOSE_2D_MaskDINO

This repository contains the implementation for **2D semantic segmentation** in the GOOSE 2025 challenge, based on **MaskDINO**.

## 🔧 Architecture Overview

The model is built on top of the MaskDINO framework, which incorporates transformer-based encoder-decoder design with improved query-based denoising strategies and dynamic anchor boxes.

<p align="center">
  <img src="assets/maskdino++.drawio.png" alt="Model Architecture" width="700"/>
</p>

## 🚀 Features

- Based on **MaskDINO** for high-performance segmentation.
- Integrated improvements from:
  - DINO (DETR with Improved DeNoising)

## 📦 Installation

Please follow the official [MaskDINO installation instructions](https://github.com/IDEA-Research/MaskDINO/blob/main/INSTALL.md).

> 🔧 Note: Make sure to set up the same environment as MaskDINO before training or evaluation.

## 🧠 Model Components

- **Backbone**: Swin-Transformer / ResNet variants
- **Transformer Decoder**: DINO + DAB-DETR + Query Denoising
- **Segmentation Head**: MaskDINO

## 📚 Citation

```bibtex
@misc{zhang2022dino,
  title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
  author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
  year={2022},
  eprint={2203.03605},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@inproceedings{li2022dn,
  title={Dn-detr: Accelerate detr training by introducing query denoising},
  author={Li, Feng and Zhang, Hao and Liu, Shilong and Guo, Jian and Ni, Lionel M and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13619--13627},
  year={2022}
}

@inproceedings{liu2022dabdetr,
  title={{DAB}-{DETR}: Dynamic Anchor Boxes are Better Queries for {DETR}},
  author={Shilong Liu and Feng Li and Hao Zhang and Xiao Yang and Xianbiao Qi and Hang Su and Jun Zhu and Lei Zhang},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=oMI9PjOb9Jl}
}

@inproceedings{li_2024_cvpr_csec,
    title       =   {Color Shift Estimation-and-Correction for Image Enhancement},
    author      =   {Yiyu Li and Ke Xu and Gerhard Petrus Hancke and Rynson W.H. Lau},
    booktitle   =   {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year        =   {2024}
}
