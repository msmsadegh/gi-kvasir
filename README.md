# 🧠 Curriculum-Guided Fine-Tuning for Multimodal VQA in GI Endoscopy
### Team Lama4Vision — MediaEval Medico’25 Challenge

This repository contains the official implementation of **Team Lama4Vision** for  
**Task 1: Visual Question Answering in GI Endoscopy** at the [MediaEval Medico’25 Challenge](https://multimediaeval.github.io/editions/2025/).

Our approach combines:
- **Curriculum-guided fine-tuning** for gradual reasoning learning,
- **Parameter-efficient adaptation (LoRA)** for low-memory training,
- **Clinically safe weak data augmentation** preserving diagnostic semantics,
- **Quantization-aware optimization (QLoRA)** for large multimodal models.

---

## 📘 Abstract

Visual Question Answering (VQA) for gastrointestinal (GI) endoscopy involves reasoning over complex medical imagery and domain-specific clinical language.  
We fine-tune **Qwen2-VL-2B-Instruct** using a curriculum-based training pipeline that progresses from simple to complex question–answer pairs, enhancing model reasoning stability.  
Lightweight **LoRA adapters** allow efficient fine-tuning under single-GPU constraints while maintaining generalization.

Experimental results on the **Kvasir-VQA-x1** dataset demonstrate consistent improvement across BLEU, ROUGE, and METEOR metrics, validating the benefit of curriculum-guided and augmentation-aware multimodal adaptation.

---

## 🧩 Repository Structure

```
├── train_gi_vqa.ipynb         # Main notebook: 3-stage curriculum fine-tuning
├── submission_task1.py        # Script for inference and JSON submission generation
├── config/                    # Hyperparameter and LoRA config files
├── data/                      # Dataset folder (images + Q&A JSON)
├── checkpoints/               # Saved LoRA and final model weights
├── utils/                     # Helper scripts for preprocessing, metrics, and augmentation
└── README.md                  # Documentation
```

---

## ⚙️ Environment Setup

Tested on:  
- Ubuntu 22.04 / CUDA 12.4  
- Python 3.10  
- GPU: NVIDIA A100 (40 GB)  
- Frameworks: PyTorch 2.4+, Transformers 4.43+, Unsloth, PEFT, BitsAndBytes

### Installation
```bash
git clone https://github.com/TeamLama4Vision/medvqa2025.git
cd medvqa2025
conda create -n gi-vqa python=3.10
conda activate gi-vqa
pip install -r requirements.txt
```

### Example `requirements.txt`
```
torch>=2.4.0
transformers>=4.43.0
unsloth
bitsandbytes
accelerate
datasets
peft
wandb
evaluate
```

---

## 🚀 Training Pipeline

The **training notebook (`train_gi_vqa.ipynb`)** implements curriculum-guided fine-tuning over three stages using LoRA and Unsloth.

### Curriculum Schedule
| Stage | Level | Description | Learning Rate | Samples | Epochs |
|:------|:------|:-------------|:---------------|:---------|:-------|
| 1 | L1 | Easy visual Q&A | 5e-5 | 49,360 | 1 |
| 2 | L1+L2 | Moderate reasoning | 3e-5 | 96,458 | 1 |
| 3 | L1+L2+L3 | Hard clinical questions | 2e-5 | 143,594 | 1 |

Training is performed with:
- **Batch size:** 512 (128 × 4 accumulation)
- **Optimizer:** AdamW (8-bit)
- **Scheduler:** Linear warmup (2 steps)
- **Logging:** [Weights & Biases](https://wandb.ai)
- **Seed control:** `random_state=3407`

### Run example (in Colab or local)
```python
!python train_gi_vqa.ipynb --base_model Qwen2-VL-2B-Instruct --lora_rank 16 --epochs 3
```

---

## 🧪 Inference & Submission

Generate submission file using:
```bash
python submission_task1.py   --model_path ./checkpoints/final_model   --input_file ./data/task1_test.json   --output_file ./submission_task1.json
```

The output follows MediaEval Task 1 format:
```json
[
  {
    "image": "kvasir_0001.jpg",
    "question": "How many polyps are visible?",
    "answer": "Two polyps"
  },
  ...
]
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|:--------|:-------------|
| **BLEU-4** | Measures n-gram precision between generated and reference answers |
| **ROUGE-1 / ROUGE-L** | Lexical overlap and longest common subsequence |
| **METEOR** | Considers synonym/stem matches for semantic similarity |

We report results per complexity level (Easy, Moderate, Hard) for transparency and reproducibility.

---

## 📚 Dataset

[Kvasir-VQA-x1 (2025)](https://www.kaggle.com/datasets/msmsadegh/kvasir-vqa-by-category) —  
159,549 Q&A pairs over 6,449 GI endoscopy images.

⚠️ **Note:**  
Original splits contained image-level leakage (~59%).  
Our internal experiments applied leakage analysis to evaluate robustness, with plans for strict non-overlapping splits in future work.

---

## 🔬 Model Resources

- 🧠 **Hugging Face:** [lama4vision/Qwen2-VL-gastro-all](https://huggingface.co/lama4vision/Qwen2-VL-gastro-all)  
- 💻 **Kaggle Notebook:** [msmsadegh/gi-kvasir](https://www.kaggle.com/code/msmsadegh/gi-kvasir)  
- 📦 **Dataset by Category:** [Kvasir-VQA-x1](https://www.kaggle.com/datasets/msmsadegh/kvasir-vqa-by-category)

---

## 🔁 Reproducibility Notes

To reproduce exact results:
```bash
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:16:8
torch.manual_seed(42)
numpy.random.seed(42)
random.seed(42)
```

We also recommend enabling **deterministic CUDA kernels** for exact reproducibility:
```python
torch.use_deterministic_algorithms(True)
```

---

## 🧾 Citation

If you use this work, please cite:

```
Azmoodeh-Kalati, M., Maghareh, M. S., Alavi, S., & Lashgari, R. (2025).
Curriculum-Guided Fine-Tuning for Multimodal VQA in GI Endoscopy (Team Lama4Vision).
In MediaEval 2025 Workshop, CEUR-WS.org.
```

**BibTeX**
```bibtex
@inproceedings{azmoodeh2025lama4vision,
  title={Curriculum-Guided Fine-Tuning for Multimodal VQA in GI Endoscopy (Team Lama4Vision)},
  author={Azmoodeh-Kalati, Mahdi and Maghareh, Mohammad Sadegh and Alavi, Saba and Lashgari, Reza},
  booktitle={MediaEval 2025 Workshop Proceedings},
  year={2025},
  organization={CEUR-WS.org}
}
```

---

## 🧾 Acknowledgments

We sincerely thank the **MediaEval Medico’25 organizers**, especially  
**Sushant Gautam**, **Steven Hicks**, and the **Simula Research Laboratory** team for their support.

This work was conducted collaboratively between:
- 🏛 **Institute of Medical Science and Technology, Shahid Beheshti University**
- 💻 **Department of Computer Science, Amirkabir University of Technology**

---

## 👥 Authors

| Name | Affiliation | Role |
|------|--------------|------|
| **Mahdi Azmoodeh-Kalati** | SBU | Model training, experiments, presentation |
| **Mohammad Sadegh Maghareh** | AUT | Code development, dataset processing, documentation |
| **Saba Alavi** | SBU | Data analysis, evaluation pipeline |
| **Prof. Reza Lashgari** | SBU | Supervision and review |

---

## 📬 Contact

For inquiries or collaboration:
- **Mohammad Sadegh Maghareh** — [msmsadegh@gmail.com](mailto:msmsadegh@gmail.com)  
- **Mahdi Azmoodeh-Kalati** — [mahdiazmoodeh95@gmail.com](mailto:mahdiazmoodeh95@gmail.com)

---

## 🧩 License
This repository is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
You are free to share, remix, and adapt with proper credit to the authors and the MediaEval 2025 organizers.
