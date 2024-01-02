# 🌋 Customizing General-Purpose Foundation Models for Medical Report Generation

PyTroch implementation of our paper:
> **Customizing General-Purpose Foundation Models for Medical Report Generation**
> 
> Bang Yang, Asif Raza, Yuexian Zou, Tong Zhang
>
> [[arXiv]](https://arxiv.org/abs/2306.05642), [[CLEF'23 Working Note]](https://ceur-ws.org/Vol-3497/paper-146.pdf)


## Update Notes
  **[2024-01-02]** Release the code, data, and models
    
## Table of Contents
- [🌋 Customizing General-Purpose Foundation Models for Medical Report Generation](#-customizing-general-purpose-foundation-models-for-medical-report-generation)
  - [Update Notes](#update-notes)
  - [Table of Contents](#table-of-contents)
  - [Environment](#environment)
  - [Getting Started](#getting-started)
    - [Preparation](#preparation)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

## Environment

1. (Optional) Creating conda environment

```bash
conda create -n pclmed python=3.8 -y
conda activate pclmed
```
 
2. Build from source for development

```bash
git clone https://github.com/yangbang18/MLLM-MRG
cd MLLM-MRG

# install PyTorch of a proper version according to your hardware
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -e .
```

## Getting Started

### Preparation
- Automatically download data and checkpoints (**Note:** the `THUDM/chatglm-6b` model we use is from [branch v0.1.0](https://huggingface.co/THUDM/chatglm-6b/tree/v0.1.0)):
```bash
bash projects/clef_2023_caption/prepare_data.sh
bash projects/clef_2023_caption/prepare_checkpoints.sh
```
- Check if you have installed `JAVA` by running "`java -version`". If not, run the following command:
```bash
bash projects/clef_2023_caption/prepare_java.sh
```

### Training
- `frozen` EVA-ViT-G + `trainable` Q-Former + `frozen` language models (**Note:** the below scripts can fit in V100)
```bash
gpus=4
bash run.sh $gpus projects/clef_2023_caption/configs/224_ViTg_FT0_Transformer.yaml
bash run.sh $gpus projects/clef_2023_caption/configs/224_ViTg_FT0_OPT2.7B.yaml
bash run.sh $gpus projects/clef_2023_caption/configs/224_ViTg_FT0_ChatGLM_ptuning0.yaml
bash run.sh $gpus projects/clef_2023_caption/configs/224_ViTg_FT0_ChatGLM_ptuning4.yaml
```
- `trainable` EVA-ViT-G + `trainable` Q-Former + `frozen` ChatGLM-6B (**Note:** batch_size=2 requires 41 GB memory per GPU; batch_size=6 requires 50 GB memory per GPU)
```bash
gpus=4
bash run.sh $gpus projects/clef_2023_caption/configs/Joint_224_ViTg_FT1_ChatGLM_ptuning4.yaml
bash run.sh $gpus projects/clef_2023_caption/configs/Joint_364_ViTg_FT1_ChatGLM_ptuning4.yaml
```
- If you don't have enough GPU memory to conduct joint training, you can use the fine-tuned EVA-ViT-G that is derived from our best-performed model to initialize the vision encoder and keep it frozen:
```bash
gpus=4
bash run.sh $gpus projects/clef_2023_caption/configs/224_ViTg_FT1_ChatGLM_ptuning4.yaml
bash run.sh $gpus projects/clef_2023_caption/configs/364_ViTg_FT1_ChatGLM_ptuning4.yaml
```

### Evaluation
- Given the result file, calculate: `BLEU-{1,2,3,4}`, `METEOR`, `CIDEr`  (supported by the `pycocoevalcap` package), and `ROUGE-{1,2,L}` (supported by the `rouge` package):
```bash
python eval_file.py --pred projects/clef_2023_caption/results/PCLmed_val_predictions.json
```
- Given the result file, additionally calculate `BERTScore`, `BLEURT`, and `CLIPScore` metrics:
```bash
# prepare necessary checkpoints and packages
bash projects/clef_2023_caption/prepare_eval.sh 

python eval_file.py --pred projects/clef_2023_caption/results/PCLmed_val_predictions.json --bert_score --bleurt --clip_score
```
- Given the fine-tuned checkpoint, conduct inference:
```bash
gpus=4

# (optional) download our released checkpoint
ckpt=./data/checkpoints/PCLmed_CLEF23_best.pth
wget "https://s3.openi.org.cn/opendata/attachment/7/4/7410fa4b-2554-4db4-b377-2de49940120d?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20240102%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240102T143435Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22PCLmed_CLEF23_best.pth%22&X-Amz-Signature=c96de4aae3a53f471078c6947fba3a66a461f02dc446c9a13126bc9110a81924" -O $ckpt

# override some arguments in the config by passing --options for inference
bash run.sh $gpus projects/clef_2023_caption/configs/364_ViTg_FT1_ChatGLM_ptuning4.yaml "--options run-evaluate=True model-load_finetuned=True model-finetuned=$ckpt"
```



## Citation
If you find our work and code helpful, please cite the following paper:
```bibtex
@misc{yang2023customizing,
      title={Customizing General-Purpose Foundation Models for Medical Report Generation}, 
      author={Bang Yang and Asif Raza and Yuexian Zou and Tong Zhang},
      year={2023},
      eprint={2306.05642},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
When referring to ImageCLEFmedical 2023 Caption general goals, general results, etc. please cite the following paper:
```
@inproceedings{ImageCLEFmedicalCaptionOverview2023,
      author = {R\"uckert, Johannes and Ben Abacha, Asma and G. Seco de Herrera, Alba and Bloch, Louise and Br\"ungel, Raphael and Idrissi-Yaghir, Ahmad and Sch\"afer, Henning and M\"uller, Henning and Friedrich, Christoph M.},
      title = {Overview of {ImageCLEFmedical} 2023 -- {Caption Prediction and Concept Detection}},
      booktitle = {CLEF2023 Working Notes},
      series = {{CEUR} Workshop Proceedings},
      year = {2023},
      volume = {},
      publisher = {CEUR-WS.org},
      pages = {},
      month = {September 18-21},
      address = {Thessaloniki, Greece}
}
```


## Acknowledgements
Our code is built upon [Salesforce/LAVIS](https://github.com/Salesforce/LAVIS).
