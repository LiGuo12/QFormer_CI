# Mitigating Co-Occurrence Bias in Medical Report Generation via Causal Intervention

## Overview
This repository presents a novel approach to address co-occurrence bias in medical report generation systems through causal intervention. By implementing a front-door causal intervention module based on BLIP-2, our method generates more accurate and unbiased medical reports by explicitly modeling the causal relationships between image features and reports.

### Key Features
- Front-door causal intervention module integrated with BLIP-2 Q-Former
- Enhanced handling of feature correlations in medical imaging
- Improved accuracy and reduced bias in report generation

<div align=center>

<img src="figure.png" width="1024" />

</div>

## Repository Structure
We added a front-door causal intervention module based on the [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) Q-Former. Below is our modified code.
### Model Components
1. **BLIP2 Q-Former with Causal Intervention**
   - Location: `lavis/models/blip2_models/blip2_qformer_ci.py`
   - Implements causal intervention in lines 114-126 and 355-368

2. **Front-Door Causal Intervention Module**
   - Location: `lavis/models/ci_modules/visual_ci.py`
   - Core implementation of the causal intervention mechanism

### Dataset Integration
- **IU-Xray Dataset Handler**
  - Dataset implementation: `lavis/datasets/datasets/iu_xray_dataset.py`
  - Dataset builder: `lavis/datasets/builders/iu_xray_builder.py`

### Training Components
- **Image-Text Pretraining**
  - Location: `lavis/tasks/image_text_pretrain.py`

## Getting Started

### Prerequisites
1. **Datasets**
   - Download IU-Xray dataset: [Download Link](https://drive.google.com/file/d/1DH0E2yhbmmmJhWjq6DGbmYylAOsLhvOH/view?usp=sharing)
   - Download compressed features: [Download Link](https://drive.google.com/file/d/1oP1eSasTT_rZfRIuZjJZ5xUKgYKl3JpJ/view?usp=sharing)

2. **Pre-trained Models**
   - Our Q-Former-CI model: [Download Link](https://drive.google.com/file/d/1YEgYJO5bUJ9L4uWlN7NtkfOFKb3lt_Uc/view?usp=sharing)
   - Medical MAE image encoder: [Download Link](https://drive.google.com/file/d/1eZXcoeMJAVjVJUNio2tCyHgiegaa-Vqr/view)

### Configuration
Update the following paths in `lavis/projects/blip2/train/pretrain_stage1_ci.yaml`:
- Line 10: Path to our pre-trained Q-Former-CI
- Line 12: Path to pre-trained Medical MAE image encoder
- Line 20: Path to compressed dataset features
- Lines 23,25,27: Paths to IU-Xray dataset annotations
- Line 29: Path to IU-Xray dataset image folder

## Evaluation

### Running Evaluation
```bash
torchrun --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_ci.yaml

<div align=center>

| Model             |Year  | B@1   | B@2   | B@3   | B@4   | MTR   | RG-L  |   CD  |
|:-----:            |:---: |:---:  |:---:  |:---:  |:---:  |:---:  |:---:  |:---:  |
| CDGPT2            | 2021 | 0.387 | 0.245 | 0.166 | 0.111 | 0.163 | 0.289 |0.257  |
| MMTN              | 2023 | 0.486 | 0.321 | 0.232 | 0.175 |   -   | 0.375 | 0.361 |
| CvT2DistilGP2     | 2023 | 0.382 | 0.245 | 0.169 | 0.124 | 0.152 | 0.285 | 0.361 |
| METransformer     | 2023 | 0.483 | 0.322 | 0.228 | 0.172 | 0.192 | 0.380 | 0.435 |
| SILC              | 2024 | 0.472 | 0.321 | 0.234 | 0.175 | 0.192 | 0.379 | 0.368 |
| PromptMGR         | 2024 | 0.401 |   -   |   -   | 0.098 | 0.160 | 0.281 |   -   |
| BoostrapLLMs      | 2024 | <ins>0.499</ins> | <ins>0.323</ins> | <ins>0.238</ins> | <ins>0.184</ins> | **0.208** | <ins>0.390</ins> |   -   |
| VLCI              | 2024 | **0.505** | **0.334** | **0.245** | **0.190** | <ins>0.210</ins> | **0.394** | **0.592** |
| Ours              | 2024 | 0.329 | 0.206 | 0.148 | 0.114 | 0.148 | 0.258 | <ins>0.491</ins> |

</div>
