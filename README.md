# Mitigating Co-Occurrence Bias in Medical Report Generation via Causal Intervention
This is the implementation of Mitigating Co-Occurrence Bias in Medical Report Generation via Causal Intervention. This work addresses the challenge of co-occurrence bias in medical report generation systems by implementing a causal intervention approach. This modification helps generate more accurate and unbiased medical reports by explicitly considering causal effect between image features and reports.

<div align=center>

<img src="figure.png" width="1024" />

</div>

## Main Modified Components
### Model
We added a front-door causal intervention (CI) module based on the [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) Q-Former. Below is our modified code.
### 1. BLIP2 Q-Former with Causal Intervention (lines 114-126, 355-368)
**File**: `lavis/models/blip2_models/blip2_qformer_ci.py`
### 2. Front-Door Causal Intervention Module
**File**: `lavis\models\ci_modules\visual_ci.py`

### Dataset
### 3. IU-Xray Dataset
**File**: `lavis\datasets\datasets\iu_xray_dataset.py`

**File**: `lavis\datasets\builders\iu_xray_builder.py`

### Task
### 4. Image-Text Pretrain CE
**File**: `lavis\tasks\image_text_pretrain.py`

## Preparation
1. Datasets:
   
Please download the IU-Xray dataset from [here](https://drive.google.com/file/d/1DH0E2yhbmmmJhWjq6DGbmYylAOsLhvOH/view?usp=sharing).
- `lavis\projects\blip2\train\pretrain_stage1_ci.yaml`: lines 23,25,27 with the paths of the annotations of IU-Xray dataset. line 29 with the path of the image folder.
  
Please download the compressed dataset features from [here](https://drive.google.com/file/d/1oP1eSasTT_rZfRIuZjJZ5xUKgYKl3JpJ/view?usp=sharing).
- `lavis\projects\blip2\train\pretrain_stage1_ci.yaml`: line 20 with the path of compressed dataset features.
2. Models:
  
We provide the well-trained model of Q-Former-CI for inference, and please download from [here](https://drive.google.com/file/d/1YEgYJO5bUJ9L4uWlN7NtkfOFKb3lt_Uc/view?usp=sharing).
- `lavis\projects\blip2\train\pretrain_stage1_ci.yaml`: line 10 with the path of our pre-trained Q-Former-CI.
  
Please download the pre-trained image encoder from repo of [Mediacal_MAE](https://github.com/lambert-x/Medical_MAE) and choose the ckp of ViT-Base/16 with 83.0 mAUC, or [here](https://drive.google.com/file/d/1eZXcoeMJAVjVJUNio2tCyHgiegaa-Vqr/view).
- `lavis\projects\blip2\train\pretrain_stage1_ci.yaml`: line 12 with the path of pre-trained image encoder.

## Evaluation
- For our model on IU-Xray dataset 

```
torchrun --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_ci.yaml
```

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
