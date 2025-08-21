# -*- coding: utf-8 -*-
"""
Created on Mon May 12 20:15:16 2025

@author: henry
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
from datasets import load_dataset
import torch
from transformers import TrainingArguments, Trainer
from transformers import BitsAndBytesConfig
import json
from datasets import Dataset

# 配置量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True   # <== CPU 分流
)


# 分詞器
model_path = "codellama/CodeLlama-7b-hf"#"./codellama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 加載數據集
with open("E:/python/code_llama/labeled_analysis_output.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]
    
dataset = []
for item in lines:
    text = item["prompt"].strip() + "\n\n" + item["completion"].strip() + tokenizer.eos_token
    dataset.append({"text": text})
    
dataset = Dataset.from_list(dataset)
    
# Tokenize 數據
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_test_split = tokenized_datasets.train_test_split(test_size=0.3, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# 加載基礎模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True
)

# 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# 準備 PEFT 模型
model = get_peft_model(model, lora_config)
model = PeftModel.from_pretrained(model, "E:/python/code_llama/results", is_trainable=True)
# 訓練參數
training_args = TrainingArguments(
    output_dir="E:/python/code_llama/results/scaff_2",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
)

# 自定義 DataCollator
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # 對於因果語言模型，不使用掩碼語言模型
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# 開始訓練
trainer.train()
trainer.save_model(output_dir="E:/python/code_llama/results/scaff_2")

#%%

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig
from peft import PeftModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model_path = "./codellama/CodeLlama-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
# 加載 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,device_map={"": 0}, 
quantization_config=bnb_config)

# 設定 padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 設定 pad_token 為 eos_token

# 移動模型到 GPU（如果可用）
model = PeftModel.from_pretrained(model, "E:/python/code_llama/lora_stage_3")
print(model.peft_config)
model.to(device)

# 測試輸入
prompt = """
Analyze the following ESG disclosure. Does it demonstrate potential greenwashing? Provide reasoning and compare with known greenwashing indicators.\n\nicies and Environmental Action Plan ・Organizing qualitative information such as ・Energy/Global warming 32〜35 （Environmental Philosophy, Policies and the Toyota Environmental Action Plan/Review of the Fourth Toyota Environmental Action Plan/The Fifth Toyota quantitative data and progress reports to ・Recycling of resources 36〜39 Environmental Action Plan） ・Ic no cm lup dl ie nm g e pn ut b t lh ice “ s cr ie ea nd tii fin cg ” e vs ie dc et nio cn e and data on the ・ ・S Au tmbs ota spn hce es ri c", "completion": 
"""
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).to(device)

# 產生回應
with torch.no_grad():
    output = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=8,
    top_k=50,
    eos_token_id=tokenizer.eos_token_id,
    num_beams=1,         # 更 deterministic
    do_sample=False,     # 不用 sampling 來避免亂輸出
    temperature=0.1,
    top_p=1.0,
    early_stopping=True,
)

# 解碼輸出
print(tokenizer.decode(output[0], skip_special_tokens=True).strip())