# -*- coding: utf-8 -*-
"""
Created on Thu May  8 13:15:22 2025

@author: henry
"""
"""
import pdfplumber
import os
import json
from tqdm import tqdm
import textstat
import re

# ---
## 設定路徑和輸出檔案
# ---
high_risk_path = "D:/中正授權軟體/ESG report/HIGH RISK"
low_risk_path = "D:/中正授權軟體/ESG report/LOW RISK"

# 新增：用於初步提取結果的輸出檔案
raw_extracted_output_file = "D:/中正授權軟體/ESG report/extracted_data_raw.jsonl"

# 最終帶有相似度篩選結果的輸出檔案
final_output_file = "D:/中正授權軟體/ESG report/greenwashing_curriculum_pbl_dataset.jsonl"

data_entries = [] # 用於儲存所有處理後的資料

# ---
## 輔助函數
# ---

# 文字前處理：移除換行符並去除首尾空白
def clean_text(text):
    # 移除多餘的空白，確保句子間只有一個空格
    cleaned = text.replace("\n", " ").strip()
    return re.sub(r'\s+', ' ', cleaned) # 將多個空白替換為單一空白

# 漸進式學習分級規則：根據 Flesch-Kincaid Grade Level 評分
def assign_level(chunk):
    score = textstat.flesch_kincaid_grade(chunk)
    if score <= 12:
        return 1
    elif score <= 15:
        return 2
    else:
        return 3

# PBL 問題模板生成：根據文本內容和難度等級生成不同的問題
def generate_pbl_prompt(chunk, level, label):
    question = ""
    if level == 1:
        question = "Identify whether this ESG statement might be misleading."
    elif level == 2:
        question = "Examine whether the ESG claims in this report align with measurable actions."
    else:  # level 3
        question = "Analyze the following ESG disclosure. Does it demonstrate potential greenwashing?"
    return f"{question}\n\n{chunk}"

# ---
## 資料提取與格式化核心函數
# ---

def extract_and_format(pdf_folder, label):
   
    print(f"\n--- 處理 '{label.upper()}' 資料夾 ---")
    
    # 檢查資料夾是否存在
    if not os.path.exists(pdf_folder):
        print(f"錯誤：資料夾 '{pdf_folder}' 不存在。跳過此資料夾。")
        return

    files_in_folder = os.listdir(pdf_folder)
    pdf_files = [f for f in files_in_folder if f.endswith(".pdf")]

    # 檢查資料夾是否有 PDF 檔案
    if not pdf_files:
        print(f"警告：資料夾 '{pdf_folder}' 中沒有找到任何 PDF 檔案。跳過此資料夾。")
        return

    for filename in tqdm(pdf_files, desc=f"Processing {label.upper()}"):
        filepath = os.path.join(pdf_folder, filename)
        
        # 從檔案名稱提取公司名稱 (假設檔名格式為 CompanyName_Year.pdf)
        # 如果檔名格式不同，請調整此處
        company_name = filename.replace(".pdf", "").rsplit('_', 1)[0] 

        full_text = ""
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n"
                    else:
                        print(f"警告：檔案 '{filename}' 的第 {page.page_number} 頁未提取到文本。")

            cleaned_text = clean_text(full_text)
            
            if not cleaned_text.strip():
                print(f"警告：檔案 '{filename}' 提取並清理後的文本為空。跳過此檔案。")
                continue
            
            # 將長文本分割成 500 字元的分段
            chunks = [cleaned_text[i:i+500] for i in range(0, len(cleaned_text), 500)]
            
            if not chunks:
                print(f"警告：檔案 '{filename}' 未能生成任何分段。跳過此檔案。")
                continue
            
            print(f"檔案 '{filename}' ({company_name}) 產生了 {len(chunks)} 個分段。")

            for chunk in chunks:
                if chunk.strip(): # 確保分段內容不為空
                    level = assign_level(chunk)
                    prompt = generate_pbl_prompt(chunk, level, label)
                    completion = " High Risk" if label == "high" else " Low Risk"
                    data_entries.append({
                        "prompt": prompt,
                        "completion": completion,
                        "level": level,
                        "source": label.upper(),
                        "company_name": company_name # 添加公司名稱
                    })
                else:
                    print(f"警告：檔案 '{filename}' 中有一個空的分段被跳過。")

        except Exception as e:
            print(f"警告：處理檔案 '{filename}' 時發生錯誤：{e}")
            continue

# ---
## 執行資料提取
# ---

# 提取 HIGH RISK 資料
extract_and_format(high_risk_path, label="high")

# 提取 LOW RISK 資料
extract_and_format(low_risk_path, label="low")

# ---
## 輸出初步提取結果到檔案
# ---
print(f"\n--- 輸出初步提取結果到 '{raw_extracted_output_file}' ---")
try:
    with open(raw_extracted_output_file, "w", encoding="utf-8") as f:
        for entry in data_entries:
            # 為了輸出到初步檔案，可以保留 company_name 以便檢查
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✅ 共產出 {len(data_entries)} 筆原始資料，儲存在 {raw_extracted_output_file}")
except Exception as e:
    print(f"錯誤：寫入原始提取檔案 '{raw_extracted_output_file}' 時發生錯誤：{e}")


# ---
# -*- coding: utf-8 -*-

Created on Thu May  8 13:15:22 2025

@author: henry
"""
#%%
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import os
import json
from tqdm import tqdm
import collections
import re # 用於分詞

# ---
## 設定檔案路徑
# ---
raw_extracted_output_file = "D:/中正授權軟體/ESG report/extracted_data_raw.jsonl"
final_output_file = "D:/中正授權軟體/ESG report/greenwashing_curriculum_pbl_dataset.jsonl"

# ---
## Jaccard 相似度輔助函數
# ---

def tokenize_text_for_jaccard(text):
    """
    將文本分詞並轉換為小寫的詞彙集合。
    移除標點符號和單個字母（除非是數字）。
    """
    # 移除標點符號，將非字母數字的字符替換為空格
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # 按空格分詞並轉小寫，過濾掉空字符串和單個字母（避免單個字母對相似度影響過大）
    tokens = [word for word in text.lower().split() if len(word) > 1 or word.isdigit()]
    return set(tokens)

def jaccard_similarity(set1, set2):
    """
    計算兩個集合的 Jaccard 相似度。
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# ---
## 讀取初步提取的資料
# ---
print(f"--- 讀取初步提取的資料 '{raw_extracted_output_file}' ---")
data_entries = []
try:
    if not os.path.exists(raw_extracted_output_file):
        print(f"錯誤：找不到初步提取的檔案 '{raw_extracted_output_file}'。請先運行提取程式。")
        exit()

    with open(raw_extracted_output_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                entry = json.loads(line.strip())
                data_entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"警告：無法解析 '{raw_extracted_output_file}' 第 {i+1} 行：'{line.strip()}'。錯誤訊息：{e}")
except Exception as e:
    print(f"錯誤：讀取檔案 '{raw_extracted_output_file}' 時發生錯誤：{e}")
    exit()

print(f"✅ 成功讀取 {len(data_entries)} 筆資料，準備進行相似度篩選。")



## 同一檔案內部相似度比對與篩除 (使用 Jaccard 相似度)

print("\n--- 進行同一檔案內部相似度比對與篩除 (Jaccard) ---")

# 推薦閾值：0.8，可調整
INTERNAL_SIMILARITY_THRESHOLD = 0.8 

# 將數據按 company_name 和 source 分組
grouped_by_company = collections.defaultdict(list)
for entry in data_entries:
    company_key = (entry.get('company_name', 'UNKNOWN_COMPANY'), entry.get('source', 'UNKNOWN_SOURCE')) 
    grouped_by_company[company_key].append(entry)

new_data_entries_after_internal_filter = []
total_internal_removed_count = 0 # 總計內部篩除數量

# 用於紀錄每個公司/來源組合篩除的數量
internal_removed_counts_by_company = collections.defaultdict(int) 

for company_key, entries_for_company in tqdm(grouped_by_company.items(), desc="Internal Filtering by Company"):
    current_company_name, current_source_type = company_key
    
    if len(entries_for_company) <= 1:
        # 如果只有一個或沒有段落，無需比對
        new_data_entries_after_internal_filter.extend(entries_for_company)
        continue

    # 為每個段落預先分詞
    tokenized_prompts = []
    original_indices_map = [] # 映射分詞後的索引回原始 data_entries 的索引
    for i, entry in enumerate(entries_for_company):
        prompt = entry.get('prompt', '')
        if prompt.strip():
            tokenized_prompts.append(tokenize_text_for_jaccard(prompt))
            original_indices_map.append(i) # 記錄原始索引
        else:
            # 如果 prompt 為空，直接保留，不參與相似度計算
            new_data_entries_after_internal_filter.append(entry)

    # 追蹤當前公司/來源組合中已保留的段落的原始索引
    retained_original_indices = set(original_indices_map)
    
    # 記錄當前公司/來源組合內部篩除的數量
    current_internal_removed_count = 0

    # 執行檔案內部的相似度比對
    for i in range(len(tokenized_prompts)):
        current_original_idx_i = original_indices_map[i]
        if current_original_idx_i not in retained_original_indices:
            continue 
        
        set_i = tokenized_prompts[i]
        
        for j in range(i + 1, len(tokenized_prompts)):
            current_original_idx_j = original_indices_map[j]
            if current_original_idx_j not in retained_original_indices:
                continue 

            set_j = tokenized_prompts[j]

            similarity = jaccard_similarity(set_i, set_j)
            
            if similarity >= INTERNAL_SIMILARITY_THRESHOLD:
                retained_original_indices.discard(current_original_idx_j) 
                current_internal_removed_count += 1
                # print(f"  內部篩除: '{current_company_name}' ({current_source_type}) - 段落 {current_original_idx_i} 與 {current_original_idx_j} 相似 ({similarity:.2f})")

    # 將保留的段落添加到新的列表中
    for original_idx in sorted(list(retained_original_indices)):
        new_data_entries_after_internal_filter.append(entries_for_company[original_idx])
    
    # 打印當前公司/來源組合的篩除數量
    if current_internal_removed_count > 0:
        print(f"  --> 公司 '{current_company_name}' ({current_source_type}): 內部篩除 {current_internal_removed_count} 筆資料。")
    
    total_internal_removed_count += current_internal_removed_count

print(f"總計因檔案內部相似性篩除 {total_internal_removed_count} 筆資料。")
print(f"檔案內部篩選後，剩餘 {len(new_data_entries_after_internal_filter)} 筆資料。")

# 將篩選後的數據賦值回 data_entries，以便後續的跨類別篩選使用
data_entries = new_data_entries_after_internal_filter



## 跨類別相似度篩選：以篩除 HIGH RISK 為主 (使用 Jaccard 相似度)

# 推薦閾值：0.75，可調整
CROSS_SIMILARITY_THRESHOLD = 0.75

# 分離高風險和低風險資料 (這裡的 source 欄位必須與 extracted_data_raw.jsonl 中的實際值一致)
high_risk_data = [entry for entry in data_entries if entry.get('source') == 'HIGH']
low_risk_data = [entry for entry in data_entries if entry.get('source') == 'LOW']

print("\n--- 進行跨類別相似度篩選 (Jaccard) ---")
print(f"篩選前 HIGH RISK 資料量: {len(high_risk_data)}")
print(f"篩選前 LOW RISK 資料量: {len(low_risk_data)}")

# 預先分詞所有低風險資料的 prompt (作為比較基準)
tokenized_low_risk_prompts = []
for entry in low_risk_data:
    prompt = entry.get('prompt', '')
    if prompt.strip():
        tokenized_low_risk_prompts.append(tokenize_text_for_jaccard(prompt))
    else:
        tokenized_low_risk_prompts.append(set()) # 如果為空，給一個空集合

filtered_high_risk_data = []
removed_counts_by_company = collections.defaultdict(lambda: collections.defaultdict(int)) # 記錄每個公司篩除的資料量

# 迭代高風險資料，檢查與低風險資料的相似度
for high_risk_idx, high_risk_entry in tqdm(enumerate(high_risk_data), total=len(high_risk_data), desc="Filtering HIGH RISK"):
    high_risk_prompt = high_risk_entry.get('prompt', '')
    high_risk_company = high_risk_entry.get('company_name', 'UNKNOWN')
    
    # 如果 prompt 為空，則直接保留，不參與相似度比較
    if not high_risk_prompt.strip():
        filtered_high_risk_data.append(high_risk_entry)
        continue

    tokenized_high_risk_prompt = tokenize_text_for_jaccard(high_risk_prompt)
    
    is_similar_to_low_risk = False
    
    # 與所有低風險資料進行比較
    for low_risk_idx, low_risk_entry in enumerate(low_risk_data):
        low_risk_company = low_risk_entry.get('company_name', 'UNKNOWN')

        # 只比較來自不同公司的資料
        if high_risk_company != low_risk_company:
            # 計算 Jaccard 相似度
            similarity_score = jaccard_similarity(tokenized_high_risk_prompt, tokenized_low_risk_prompts[low_risk_idx])
            
            if similarity_score >= CROSS_SIMILARITY_THRESHOLD:
                is_similar_to_low_risk = True
                # 記錄被篩除的資料及其來源
                removed_counts_by_company[high_risk_company]['HIGH'] += 1 
                break # 找到一個相似的就停止
    
    if not is_similar_to_low_risk:
        filtered_high_risk_data.append(high_risk_entry)

print(f"篩選後 HIGH RISK 資料量: {len(filtered_high_risk_data)}")

print("\n--- 按公司和來源統計篩除的資料量 ---")
total_removed_cross = 0
for company, source_counts in removed_counts_by_company.items():
    for source, count in source_counts.items(): # 這裡的 source 會是 'HIGH'
        print(f"公司 '{company}' ({source}): 篩除 {count} 筆資料。")
        total_removed_cross += count

print(f"總計篩除 {total_removed_cross} 筆 HIGH RISK 資料，因為它們與來自不同公司的 LOW RISK 資料過於相似。")


# 將篩選後的資料合併回 final_data_entries (現在是保留 low_risk_data + 篩選後的 high_risk_data)
final_data_entries = low_risk_data + filtered_high_risk_data

# ---
## 輸出最終結果
# ---

# 輸出為 JSONL 格式
try:
    with open(final_output_file, "w", encoding="utf-8") as f:
        for entry in final_data_entries:
            # 為了輸出到最終檔案，移除 'company_name' 欄位，因為它只是用於篩選
            temp_entry = entry.copy()
            if 'company_name' in temp_entry:
                del temp_entry['company_name']
            f.write(json.dumps(temp_entry, ensure_ascii=False) + "\n")
    print(f"\n✅ 最終共產出 {len(final_data_entries)} 筆資料，儲存在 {final_output_file}")
except Exception as e:
    print(f"錯誤：寫入最終檔案 '{final_output_file}' 時發生錯誤：{e}")