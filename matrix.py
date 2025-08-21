from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# 載入模型與 tokenizer，只需載入一次

uni_prompt = """
Analyze the following ESG disclosure. Does it demonstrate potential greenwashing?

erformance crite- ria to other sustainable development criteria (HSE, CSR, HR VARIABLE COMPENSATION ALIGNED WITH THE COMPANY’S and diversity) in the determination of the Chairman and CEO’s STRATEGIC OBJECTIVES variable compensation. The Oil & Gas growth criterion in this calculation was replaced by two criteria concerning his steer- ANNUAL VARIABLE PORTION ing of the transformation and profitable growth in renewables Chairman & CEO Senior Executives: and electricity. The granting of performance 

"completion":
"""
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #model_path = "E:/python/code_llama/codellama/CodeLlama-7b-hf"
    model_path = "codellama/CodeLlama-7b-Instruct-hf"
    #lora_path = "E:/python/code_llama/lora_stage_3"
    lora_path = "E:/python/code_llama/results/regular"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map={"": 0}, quantization_config=bnb_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, lora_path)
    model.to(device)
    print(model.peft_config)
    model.eval()  # 確保在評估模式

    # ******************* 新增：在 load_model 內部進行一次測試推論 *******************
    test_prompt = uni_prompt
    test_inputs = tokenizer(
        test_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    #print(f"DEBUG (load_model internal): input_ids = {test_inputs['input_ids']}")
    print(f"DEBUG (load_model internal): Raw prompt repr: {repr(test_prompt)}") # 確保 test_prompt 來自於原始定義的長字串
    with torch.no_grad():
        test_output = model.generate(
            test_inputs["input_ids"],
            attention_mask=test_inputs["attention_mask"],
            max_new_tokens=8,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
        )
    test_decoded_output = tokenizer.decode(
        test_output[0], skip_special_tokens=True).strip()
    print(f"DEBUG: Test generation inside load_model: {test_decoded_output}")
    # ********************************************************************************

    return tokenizer, model, device

# 單次推論函數：輸入 prompt、回傳生成文字


def generate_response(prompt, tokenizer, model, device):
    # 應該是 False
    print(f"DEBUG: Model in eval mode at start of generate_response: {model.training}")
    raw_prompt = prompt
    # 暫時移除字串清理，確認是否為其導致的問題
    # cleaned_prompt = raw_prompt.strip().replace("\r", "").replace("\u2028", "").replace("\u200b", "")
    # formatted_prompt = cleaned_prompt
    formatted_prompt = raw_prompt  # 暫時使用原始提示
    print(f"DEBUG (generate_response): Formatted prompt repr: {repr(formatted_prompt)}")
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    #print(f"DEBUG (generate_response): input_ids = {inputs['input_ids']}")
    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=8,
            # 移除衝突的參數，或根據需求調整
            # 保持 do_sample=False 和 num_beams=1 進行確定性生成
            do_sample=False,
            num_beams=1,
            # 移除 temperature 和 early_stopping，因為它們在當前設定下無效
            # temperature=0.1,
            # top_p=1.0,
            # top_k=50, # top_k 和 top_p 在 do_sample=False 時也無效
            eos_token_id=tokenizer.eos_token_id,
            # early_stopping=True, # 移除
        )

    decoded_output = tokenizer.decode(
        output[0], skip_special_tokens=True).strip()
    return decoded_output


# 初始化一次模型
tokenizer, model, device = load_model()

# 測試用一筆 prompt
sample_prompt = """
Analyze the following ESG disclosure. Does it demonstrate potential greenwashing?

erformance crite- ria to other sustainable development criteria (HSE, CSR, HR VARIABLE COMPENSATION ALIGNED WITH THE COMPANY’S and diversity) in the determination of the Chairman and CEO’s STRATEGIC OBJECTIVES variable compensation. The Oil & Gas growth criterion in this calculation was replaced by two criteria concerning his steer- ANNUAL VARIABLE PORTION ing of the transformation and profitable growth in renewables Chairman & CEO Senior Executives: and electricity. The granting of performance 

"completion":
"""
# 呼叫推論
response = generate_response(uni_prompt, tokenizer, model, device)
print("模型回應：", response)

#%%
import json
import torch
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path, adapter_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        quantization_config=bnb_config
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)

    return tokenizer, model, device

def generate_response(prompt, tokenizer, model, device):
    # 補上結尾提示（如果缺少）
    if not prompt.rstrip().endswith('"completion":'):
        prompt = prompt.rstrip() + '\n\n"completion":'

    print(f"\n[DEBUG] Prompt repr: {repr(prompt)}")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=8,
            top_k=50, # 根據您之前的測試，這裡的參數應該是有效的組合
            num_beams=1,
            do_sample=False,
            temperature=0.1,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True, # 如果 num_beams=1, early_stopping=True 可能會觸發警告，但對結果影響不大
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_sample(jsonl_path, tokenizer, model, device, sample_count=20, output_csv="results_output.csv"):
    y_true = []
    y_pred = []
    records = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_count:
                break

            entry = json.loads(line)
            prompt = entry.get('prompt')
            label = entry['completion'].strip()

            response = generate_response(prompt, tokenizer, model, device)

            if 'High Risk' in response:
                prediction = 'High Risk'
            elif 'Low Risk' in response:
                prediction = 'Low Risk'
            else:
                print(f"[SKIPPED] Unrecognized response: {response}")
                continue

            y_true.append(label)
            y_pred.append(prediction)

            # 儲存結果
            records.append({
                "prompt": prompt,
                "label": label,
                "prediction": prediction,
                "raw_response": response
            })

    # 匯出 CSV
    with open(output_csv, "w", newline='', encoding='utf-8') as csvfile:
        fieldnames = ["prompt", "label", "prediction", "raw_response"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print(f"\n✅ 已將推論結果匯出至：{output_csv}")

    # 繪製混淆矩陣
    labels = ['High Risk', 'Low Risk']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels)

    print("\n=== Classification Report ===")
    print(report)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Sample 20)")
    plt.tight_layout()
    plt.show()

# === 主程式 ===
if __name__ == "__main__":
    model_path = "E:/python/code_llama/codellama/CodeLlama-7b-hf"
    adapter_path = "E:/python/code_llama/results/regular"
    #adapter_path = "E:/python/code_llama/lora_stage_3"
    jsonl_path = "E:/python/code_llama/testing_data.jsonl"

    tokenizer, model, device = load_model(model_path, adapter_path)
    evaluate_sample(jsonl_path, tokenizer, model, device, sample_count=10, output_csv="results_output.csv")


#%%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- 定義模型期望的 Prompt 後綴 ---
# 確保這個後綴與您在 load_model 內部測試時使用的 test_prompt 的結尾完全一致
# 根據您之前提供的 repr 輸出，這是 load_model 內部測試能正常運作的後綴部分
PROMPT_SUFFIX = '\xa0\n\n"completion":\n'

# --- 載入模型與 tokenizer 函數 ---
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

   # model_path = "E:/python/code_llama/codellama/CodeLlama-7b-hf"
    model_path = "codellama/CodeLlama-7b-Instruct-hf"
    #lora_path = "E:/python/code_llama/lora_stage_3"
    #lora_path = "E:/python/code_llama/results/regular"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": 0}, quantization_config=bnb_config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #model = PeftModel.from_pretrained(model, lora_path)
    model.to(device)
    model.eval()
    return tokenizer, model, device
    # 內部測試推論 (保持用於驗證模型載入後的狀態)
    # 這個 test_prompt 應該包含完整的後綴，與模型訓練時的格式一致
"""    test_prompt = 
Analyze the following ESG disclosure. Does it demonstrate potential greenwashing?

erformance crite- ria to other sustainable development criteria (HSE, CSR, HR VARIABLE COMPENSATION ALIGNED WITH THE COMPANY’S and diversity) in the determination of the Chairman and CEO’s STRATEGIC OBJECTIVES variable compensation. The Oil & Gas growth criterion in this calculation was replaced by two criteria concerning his steer- ANNUAL VARIABLE PORTION ing of the transformation and profitable growth in renewables Chairman & CEO Senior Executives: and electricity. The granting of performance\xa0

"completion":
# 確保這裡的字串與 PROMPT_SUFFIX 銜接正確，並能正常生成
    test_inputs = tokenizer(
        test_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        test_output = model.generate(
            test_inputs["input_ids"],
            attention_mask=test_inputs["attention_mask"],
            max_new_tokens=8,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
        )
    test_decoded_output = tokenizer.decode(test_output[0], skip_special_tokens=True).strip()
    print(f"DEBUG: Test generation inside load_model: {test_decoded_output}")
""" 
   # return tokenizer, model, device

# --- 單次推論函數 ---
def generate_response(prompt_text, tokenizer, model, device):
    # 此處 formatted_prompt 會包含從 JSONL 讀取的內容加上 PROMPT_SUFFIX
    formatted_prompt = prompt_text 

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=8,
            do_sample=False, # 保持確定性生成
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return decoded_output

# --- 從模型回應中提取標籤的函數 ---
def extract_model_label(model_response):
    response_lower = model_response.lower()
    if "high risk" in response_lower:
        return "High Risk"
    elif "low risk" in response_lower:
        return "Low Risk"
    else:
        return None # 返回 None 表示無法識別的標籤

# --- 主程序 ---
if __name__ == "__main__":
    tokenizer, model, device = load_model()

    jsonl_file_path = "E:/python/code_llama/testing_data.jsonl"
    output_csv_path = "E:/python/code_llama/instruct_model_inference_results.csv"

    results = [] # 用於儲存所有處理結果
    true_labels_for_cm = [] # 僅用於混淆矩陣的真實標籤
    predicted_labels_for_cm = [] # 僅用於混淆矩陣的模型預測標籤

    # 明確定義我們關心的類別
    target_classes = ["Low Risk", "High Risk"] 

    print(f"\n--- Starting Inference from {jsonl_file_path} ---")

    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    prompt_from_file = data.get("prompt")
                    completion = data.get("completion") # 獲取正確標籤

                    if prompt_from_file is None or completion is None:
                        print(f"Warning: Line {line_num + 1} incomplete (missing prompt/completion). Skipping.")
                        continue

                    # *** 關鍵修改：在這裡補上 PROMPT_SUFFIX ***
                    # 注意：如果您的 JSONL 檔案中的 prompt 已經以換行符結尾，您可能需要調整 PROMPT_SUFFIX 或添加一個條件判斷
                    # 但基於您之前的描述，它缺少了整個 \xa0\n\n"completion":\n'
                    full_prompt_for_inference = prompt_from_file + PROMPT_SUFFIX
                    
                    # 可選：檢查補齊後的 prompt repr，確保格式正確
                   # print(f"DEBUG: Full prompt for inference repr: {repr(full_prompt_for_inference)}")

                    model_response = generate_response(full_prompt_for_inference, tokenizer, model, device)
                    model_label = extract_model_label(model_response)

                    results.append({
                        "Prompt": prompt_from_file, # 記錄原始 prompt
                        "Actual_Completion": completion,
                        "Model_Predicted_Label": model_label if model_label else "N/A",
                        "Model_Full_Response": model_response
                    })

                    if model_label in target_classes and completion in target_classes:
                        true_labels_for_cm.append(completion)
                        predicted_labels_for_cm.append(model_label)
                    else:
                        reason = ""
                        if model_label not in target_classes:
                            reason += f"Model output '{model_label}' is not an expected label. "
                        if completion not in target_classes:
                            reason += f"Actual completion '{completion}' is not an expected label. "
                        print(f"Line {line_num + 1}: Excluded from confusion matrix. Reason: {reason.strip()} (Full Response: '{model_response}')")

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_num + 1}: {e}. Line content: {line.strip()}")
                except Exception as e:
                    print(f"An error occurred while processing line {line_num + 1}: {e}")
                    results.append({
                        "Prompt": prompt_from_file if 'prompt_from_file' in locals() else None,
                        "Actual_Completion": completion if 'completion' in locals() else None,
                        "Model_Predicted_Label": "Error",
                        "Model_Full_Response": f"Error: {e}"
                    })


    except FileNotFoundError:
        print(f"Error: The file '{jsonl_file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\n--- Inference Complete ---")

    # --- 匯出結果到 CSV ---
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False, encoding='utf-8-sig') 
    print(f"All inference results saved to '{output_csv_path}'")

    # --- 繪製混淆矩陣 ---
    if len(true_labels_for_cm) > 0 and len(predicted_labels_for_cm) > 0:
        print("\n--- Confusion Matrix (2x2, Excluding unexpected responses) ---")

        cm = confusion_matrix(true_labels_for_cm, predicted_labels_for_cm, labels=target_classes)
        
        cm_df = pd.DataFrame(cm, index=[f'實際: {l}' for l in target_classes],
                             columns=[f'預測: {l}' for l in target_classes])

        print(cm_df)

        plt.figure(figsize=(6, 5)) 
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16}) 
        plt.title('混淆矩陣', fontsize=18)
        plt.xlabel('預測標籤', fontsize=14)
        plt.ylabel('實際標籤', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12, rotation=0) 
        plt.tight_layout()
        plt.show()

        print("\n--- Classification Report ---")
        print(classification_report(true_labels_for_cm, predicted_labels_for_cm, labels=target_classes, zero_division=0))

    else:
        print("\n沒有足夠的有效資料來計算混淆矩陣 (模型沒有為任何樣本生成預期的標籤，或實際標籤不在目標類別中)。")

    print("\n--- 腳本執行完成 ---")
    
#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 解決中文字型問題（若需要顯示中文標籤）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 1. 讀取混淆矩陣 CSV（你上傳的圖片對應的原始 CSV）
csv_path = 'E:/python/code_llama/model_inference_results.csv'  # ← 替換為實際路徑（你上傳的CSV名稱）
df = pd.read_csv(csv_path, index_col=0)

# 2. 轉換DataFrame中的值為整數（或浮點數）
df = df.apply(pd.to_numeric)

# 3. 取出混淆矩陣與標籤
cm = df.values
labels = df.columns.tolist()
true_labels = df.index.tolist()

# 4. 繪圖
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
cax = ax.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(cax)

ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(true_labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(true_labels)

# 顯示數字
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(int(cm[i, j]), 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14)

# 儲存與顯示圖片
plt.tight_layout()
plt.savefig('confusion_matrix_manual.png', dpi=300)
plt.show()
