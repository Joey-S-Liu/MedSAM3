# MedSAM3: Delving into Segment Anything with Medical Concepts

<div align="center">

**Anglin Liu**<sup>1,*</sup>, **Rundong Xue**<sup>2,*</sup>, **Xu R. Cao**<sup>3,†</sup>, **Yifan Shen**<sup>3</sup>, **Yi Lu**<sup>1</sup>, **Xiang Li**<sup>3</sup>, **Qianqian Chen**<sup>4</sup>, **Jintai Chen**<sup>1,5,†</sup>

<sup>1</sup> The Hong Kong University of Science and Technology (Guangzhou)  
<sup>2</sup> Xi’an Jiaotong University  
<sup>3</sup> University of Illinois Urbana-Champaign  
<sup>4</sup> Southeast University  
<sup>5</sup> The Hong Kong University of Science and Technology  

<small><sup>*</sup> Equal Contribution &nbsp;&nbsp; <sup>†</sup> Corresponding Author</small>

[![arXiv](https://img.shields.io/badge/arXiv-2511.19046-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2511.19046)
&nbsp;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-ffd21e)](https://huggingface.co/lal-Joey/MedSAM3)

</div>

---

## 📖 Introduction

**MedSAM3** is a pure text-guided (concept-guided) medical image segmentation model. Unlike traditional models that rely on bounding boxes or points, MedSAM3 leverages specific medical concepts to segment targets across a wide range of modalities.

### 🌟 Key Features & Dataset Statistics

We constructed a large-scale dataset uniformly sampled to ensure diversity and robustness. The model covers **diverse medical modalities**:
* **Radiology:** CT, MRI, PET, X-ray
* **Optical/Microscopic:** Microscopy, Histopathology, Dermoscopy, OCT, Cell
* **Video/Procedure:** Ultrasound, Endoscopy, Surgery video

**Dataset Scale:**
* **658,094** Images
* **2,863,974** Instance Annotations
* **330** Unique Medical Text IDs (Concepts)

## 📦 Model & Weights

We adopted a parameter-efficient fine-tuning strategy based on **SAM3** using **LoRA (Low-Rank Adaptation)**.

We are releasing our first version (**v1**) of the LoRA weights.

| Model Version | Base Model | Method | Link |
| :--- | :--- | :--- | :--- |
| **MedSAM3-v1** | SAM3 | LoRA Fine-tuning | [**Download LoRA Weights**](https://huggingface.co/lal-Joey/MedSAM3) |

## 🔗 References

This project is built upon the following excellent open-source projects. Please refer to them for the base environment setup. If you encounter code-related issues, please also refer to the specific instructions and documentation provided by these works:

* **SAM3:** [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)
* **SAM3_LoRA:** [https://github.com/Sompote/SAM3_LoRA](https://github.com/Sompote/SAM3_LoRA)

## 🚀 Inference

Follow these steps to run inference on your medical images.

### 1. Setup
```python
# Clone repository
git clone https://github.com/Joey-S-Liu/MedSAM3.git
cd MedSAM3

# Install dependencies
pip install -e .

# Login to Hugging Face
hf auth login
# Paste your token when prompted
```

### 2. Inference Code
```python
python3 infer_sam.py \
  --config configs/full_lora_config.yaml \
  --image path/to/image.jpg \
  --prompt "skin lesion" \
  --threshold 0.5 \
  --nms-iou 0.5 \
  --output skin_lesion.png
```

### 3. Training Code
```python
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml
```

## ⚠️ Notes & Precautions

1. **Hyperparameter Tuning:** Please flexibly adjust the `threshold` and `nms-iou` parameters according to the specific task type. Different modalities or segmentation targets may require different sensitivity settings (e.g., some tasks achieve optimal results with `threshold=0.8`, while others work best with `threshold=0.5`). We recommend using the visualization outputs from `infer_sam.py` to determine the best settings for your specific task.
2. **Configuration:** Please specify the path to your LoRA weights in the `configs/full_lora_config.yaml` file under the `output_dir` field.
3. **Data Format:** The training data follows the **COCO format**, which is consistent with the standard SAM3 implementation.
4. **Supported Tasks (v1):** The specific list of task categories supported by the current v1 version will be released within a few days. We encourage users to experiment with specific tasks and provide feedback.

## 📧 Contact

If you have any questions regarding this project, please feel free to contact the corresponding authors:

* **Xu R. Cao**: [xucao2@illinois.edu](mailto:xucao2@illinois.edu)
* **Jintai Chen**: [jintaiCHEN@hkust-gz.edu.cn](mailto:jintaiCHEN@hkust-gz.edu.cn)

## 🖊️ Citation

If you find this project useful for your research, please consider citing:

```bibtex
@misc{liu2025medsam3delvingsegmentmedical,
      title={MedSAM3: Delving into Segment Anything with Medical Concepts}, 
      author={Anglin Liu and Rundong Xue and Xu R. Cao and Yifan Shen and Yi Lu and Xiang Li and Qianqian Chen and Jintai Chen},
      year={2025},
      eprint={2511.19046},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.19046}, 
}
