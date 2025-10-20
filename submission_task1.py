from datasets import load_dataset, Image as HfImage
from transformers import AutoProcessor
import torch
import json
import time
from tqdm import tqdm
import subprocess
import platform
import sys

from evaluate import load

bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")


ds = load_dataset("SimulaMet/Kvasir-VQA-x1")["test"]
ds_shuffled = ds.shuffle(seed=42) # Shuffle with fixed seed for reproducibility
val_dataset = ds_shuffled.select(range(1500)) # Select first 1500 after shuffle

val_dataset = val_dataset.cast_column("image", HfImage())
predictions = []  # List to store predictions

gpu_name = torch.cuda.get_device_name(
    0) if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_mem(): return torch.cuda.memory_allocated(device) / \
    (1024 ** 2) if torch.cuda.is_available() else 0


initial_mem = get_mem()

# ✏️✏️--------EDIT SECTION 1: SUBMISISON DETAILS and MODEL LOADING --------✏️✏️#

SUBMISSION_INFO = {
    # 🔹 TODO: PARTICIPANTS MUST ADD PROPER SUBMISSION INFO FOR THE SUBMISSION 🔹
    # This will be visible to the organizers
    # DONT change the keys, only add your info
    "Participant_Names": "Mahdi Azmoodeh-Kalati, Mohammad Sadegh Maghareh, Saba Alavi, Reza Lashgari",
    "Affiliations": "Shahid Beheshti University, Amirkabir University of Technology",
    "Contact_emails": ["m.azmudehkalati@alumni.sbu.ac.ir", "maghareh@aut.ac.ir", "Sabaalavi13@gmail.com", "r_lashgari@sbu.ac.ir"],
    # But, the first email only will be used for correspondance
    "Team_Name": "Lama4vision",
    "Country": "Iran",
    "Notes_to_organizers": '''
        We fine-tuned the Qwen2-VL-2B-Instruct-bnb-4bit model using a parameter-efficient LoRA (PEFT) approach.
        Our goal was to adapt large multimodal language model to gastrointestinal (GI) endoscopy question answering tasks within the MediaEval 2025 MedVQA challenge.

        To improve robustness and reasoning ability, we applied several training strategies:
        1. **Data Augmentation:** Introduced visual augmentations to increase resilience to illumination changes, artifacts, and noise in endoscopic images.
        2. **Curriculum Learning:** Employed a three-stage incremental training pipeline (easy → moderate → hard question levels) based on annotated complexity in the Kvasir-VQA-x1 dataset, to gradually build reasoning capability.
        3. **Layer-Selective Fine-Tuning:** Enabled LoRA adapters on attention and MLP modules, and selectively fine-tuned vision, language, and cross-attention layers to balance efficiency and performance.

        This fine-tuning strategy significantly enhanced the model’s understanding of domain-specific visual and linguistic patterns, improving answer consistency on complex clinical queries.

        We observed:
        • LoRA fine-tuning boosted performance compared to zero-shot baselines.  
        • Curriculum learning reduced catastrophic forgetting and improved handling of complex reasoning.  
        • Data redundancy and augmentation improved generalization under noisy or artifact-heavy conditions.

        Informal note: Our experiments were conducted on NVIDIA A100 40 GB GPUs. We greatly appreciate the organizers’ efforts in curating the dataset and evaluation framework. Team Lama4Vision enjoyed participating and exploring how large vision-language models can support medical reasoning in GI endoscopy.

        '''
}
# 🔹 TODO: PARTICIPANTS MUST LOAD THEIR MODEL HERE, EDIT AS NECESSARY FOR YOUR MODEL 🔹
# can add necessary library imports here
from unsloth import FastVisionModel

model_hf, processor = FastVisionModel.from_pretrained(
    "mahdiazmoodeh95/Qwen2-VL-gastro-all",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)
FastVisionModel.for_inference(model_hf)

# 🏁----------------END  SUBMISISON DETAILS and MODEL LOADING -----------------🏁#

start_time, post_model_mem = time.time(), get_mem()
total_time, final_mem = round(
    time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
model_mem_used = round(post_model_mem - initial_mem, 2)

for idx, ex in enumerate(tqdm(val_dataset, desc="Validating")):
    question = ex["question"]
    image = ex["image"].convert(
        "RGB") if ex["image"].mode != "RGB" else ex["image"]
    # you have access to 'question' and 'image' variables for each example

# ✏️✏️___________EDIT SECTION 2: ANSWER GENERATION___________✏️✏️#
    # 🔹 TODO: PARTICIPANTS CAN MODIFY THIS TOKENIZATION STEP IF NEEDED 🔹
    instruction = f"""You are an expert gastroenterologist. Your task is to analyze the image and answer the user's question.
    **Constraint:** Your entire response must be a single, concise sentence. Do not elaborate or provide additional context.
    **Question:** {question}"""
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt = True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")

    # 🔹 TODO: PARTICIPANTS CAN MODIFY THE GENERATION AND DECODING METHOD HERE 🔹
    outputs = model_hf.generate(
        **inputs,
        max_new_tokens=100,   
        do_sample=False,
    )
    
    raw_answer = processor.batch_decode(outputs, skip_special_tokens=True,skip_prompt = True)[0]  
    answer = raw_answer.split("assistant")[1]

    # make sure 'answer' variable will hold answer (sentence/word) as str
# 🏁________________ END ANSWER GENERATION ________________🏁#

# ⛔ DO NOT EDIT any lines below from here, can edit only upto decoding step above as required. ⛔
    # Ensures answer is a string
    assert isinstance(
        answer, str), f"Generated answer at index {idx} is not a string"
    # Appends prediction
    predictions.append(
        {"index": idx, "img_id": ex["img_id"], "question": ex["question"], "answer": answer})

# Ensure all predictions match dataset length
assert len(predictions) == len(
    val_dataset), "Mismatch between predictions and dataset length"

total_time, final_mem = round(
    time.time() - start_time, 4), round(get_mem() - post_model_mem, 2)
model_mem_used = round(post_model_mem - initial_mem, 2)

# caulcualtes metrics
references = [[e] for e in val_dataset['answer']]
preds = [pred['answer'] for pred in predictions]

bleu_result = bleu.compute(predictions=preds, references=references)
rouge_result = rouge.compute(predictions=preds, references=references)
meteor_result = meteor.compute(predictions=preds, references=references)
bleu_score = round(bleu_result['bleu'], 4)
rouge1_score = round(float(rouge_result['rouge1']), 4)
rouge2_score = round(float(rouge_result['rouge2']), 4)
rougeL_score = round(float(rouge_result['rougeL']), 4)
meteor_score = round(float(meteor_result['meteor']), 4)

public_scores = {
    'bleu': bleu_score,
    'rouge1': rouge1_score,
    'rouge2': rouge2_score,
    'rougeL': rougeL_score,
    'meteor': meteor_score
}
print("✨Public scores: ", public_scores)

# Saves predictions to a JSON file

output_data = {"submission_info": SUBMISSION_INFO, "public_scores": public_scores,
               "predictions": predictions, "total_time": total_time, "time_per_item": total_time / len(val_dataset),
               "memory_used_mb": final_mem, "model_memory_mb": model_mem_used, "gpu_name": gpu_name,
               "debug": {
                   "packages": json.loads(subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=json"])),
                   "system": {
                       "python": platform.python_version(),
                       "os": platform.system(),
                       "platform": platform.platform(),
                       "arch": platform.machine()
                   }}}


with open("predictions_1.json", "w") as f:
    json.dump(output_data, f, indent=4)
print(f"Time: {total_time}s | Mem: {final_mem}MB | Model Load Mem: {model_mem_used}MB | GPU: {gpu_name}")
print("✅ Scripts Looks Good! Generation process completed successfully. Results saved to 'predictions_1.json'.")
print("Next Step:\n 1) Upload this submission_task1.py script file to HuggingFace model repository.")
print('''\n 2) Make a submission to the competition:\n Run:: medvqa validate_and_submit --competition=medico-2025 --task=1 --repo_id=...''')
