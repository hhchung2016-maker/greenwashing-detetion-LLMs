# -*- coding: utf-8 -*-
"""
Created on Sat May 10 18:41:35 2025

@author: henry
"""
import json
import random
import textstat
import ast



def calculate_average_readability(data_list, num_samples=100):
    """
    從資料列表中隨機抽取指定數量的樣本，並計算平均 Flesch Reading Ease 和 Flesch-Kincaid Grade Level。

    Args:
        data_list (list): 包含文本資料的列表，每個元素應該是包含 'prompt' 或 'completion' 鍵的字典。
        num_samples (int): 要隨機抽取的樣本數量。

    Returns:
        tuple: 包含平均 Flesch Reading Ease 和平均 Flesch-Kincaid Grade Level 的元組，
               如果抽取的樣本數量不足，則返回 None。
    """
    if not data_list:
        print("資料列表為空。")
        return None

    samples = random.sample(data_list, min(num_samples, len(data_list)))

    flesch_scores = []
    kincaid_scores = []

    for item in samples:
        text = ""
        if 'prompt' in item:
            text += item['prompt'] + " "
        if 'completion' in item:
            text += item['completion']

        if text:
            flesch_scores.append(textstat.flesch_reading_ease(text))
            kincaid_scores.append(textstat.flesch_kincaid_grade(text))

    if flesch_scores:
        avg_flesch = sum(flesch_scores) / len(flesch_scores)
        avg_kincaid = sum(kincaid_scores) / len(kincaid_scores)
        return avg_flesch, avg_kincaid
    else:
        return None


def calculate_average_readability_txt(data_list, num_samples=100):
    """
    從資料列表中隨機抽取指定數量的樣本，並計算平均 Flesch Reading Ease 和 Flesch-Kincaid Grade Level。

    Args:
        data_list (list): 包含文本資料的列表，每個元素應該是包含 'prompt' 或 'completion' 鍵的字典。
        num_samples (int): 要隨機抽取的樣本數量。

    Returns:
        tuple: 包含平均 Flesch Reading Ease 和平均 Flesch-Kincaid Grade Level 的元組，
               如果抽取的樣本數量不足，則返回 None。
    """
    if not data_list:
        print("資料列表為空。")
        return None

    samples = random.sample(data_list, min(num_samples, len(data_list)))

    flesch_scores = []
    kincaid_scores = []

    for text in samples:
        if text:
            flesch_scores.append(textstat.flesch_reading_ease(text))
            kincaid_scores.append(textstat.flesch_kincaid_grade(text))

    if flesch_scores:
        avg_flesch = sum(flesch_scores) / len(flesch_scores)
        avg_kincaid = sum(kincaid_scores) / len(kincaid_scores)
        return avg_flesch, avg_kincaid
    else:
        return None
# 讀取 010_generated_prompts_completions.txt 檔案並提取 completion (使用 ast.literal_eval 逐行處理)
file_path_txt = '002_generated_prompts_completions.txt'
raw_text = []
try:
    with open(file_path_txt, "r", encoding="utf-8") as f:
            try:
                data_list = ast.literal_eval(f.read())
            except Exception as e:
                print(f"❌ 無法解析: {e}")
            for item in data_list:
              #  print("item:"+str(item))
                raw_text.append(item.get('completion').strip())
except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{file_path_txt}'")

# 計算 010_generated_prompts_completions.txt 中 completion 的平均可讀性
#print("raw_text:"+str(raw_text))
if raw_text:
    avg_readability_completion = calculate_average_readability_txt(raw_text)
    print("raw_text:"+str(avg_readability_completion))
    if avg_readability_completion:
        avg_flesch_completion, avg_kincaid_completion = avg_readability_completion
        print(f"檔案 '{file_path_txt}' 中隨機 100 筆 'completion' 的平均 Flesch Reading Ease 分數：{avg_flesch_completion:.2f}")
        print(f"檔案 '{file_path_txt}' 中隨機 100 筆 'completion' 的平均 Flesch-Kincaid Grade Level 分數：{avg_kincaid_completion:.2f}")
    else:
        print(f"檔案 '{file_path_txt}' 無有效 'completion' 資料可計算可讀性")
else:
    print(f"檔案 '{file_path_txt}' 無包含 'completion' 的有效資料。")
print("-" * 30)

# 讀取 greenwashing_prompt_completion.json 檔案
file_path_json = 'greenwashing_prompt_completion.json'
data_from_json = []
try:
    with open(file_path_json, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data_from_json.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"警告：無法解析 JSON 檔案中的行：'{line.strip()}'")
except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{file_path_json}'")
    data_from_json = []

# 計算 greenwashing_prompt_completion.json 的平均可讀性
if data_from_json:
    avg_readability_json = calculate_average_readability(data_from_json)
    if avg_readability_json:
        avg_flesch_json, avg_kincaid_json = avg_readability_json
        print(f"檔案 '{file_path_json}' 的隨機 100 筆資料平均 Flesch Reading Ease 分數：{avg_flesch_json:.2f}")
        print(f"檔案 '{file_path_json}' 的隨機 100 筆資料平均 Flesch-Kincaid Grade Level 分數：{avg_kincaid_json:.2f}")
    else:
        print(f"檔案 '{file_path_json}' 無有效資料可計算可讀性。")
else:
    print(f"檔案 '{file_path_json}' 無資料可供分析。")
    
    #%%
    
    
import textstat

text = "A consulting firm claims to use sustainable practices, but is later found to have falsified environmental impact reports.\n\nCreate a detailed blog post of approximately 1000 words discussing the impact of social media on mental health"

# 常用指標
flesch_score = textstat.flesch_reading_ease(text)
flesch_kincaid = textstat.flesch_kincaid_grade(text)
gunning_fog = textstat.gunning_fog(text)

print(f"Flesch Reading Ease: {flesch_score}")
print(f"Flesch-Kincaid Grade Level: {flesch_kincaid}")
print(f"Gunning Fog Index: {gunning_fog}")