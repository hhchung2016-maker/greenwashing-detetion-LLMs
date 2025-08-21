
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np # 確保 numpy 被導入以處理 nan

# --- 設定 CSV 檔案路徑 ---
csv_file_path = "oldlevel3_model_inference_results.csv" # 替換成您的 CSV 檔案路徑

# --- 定義您關心的目標類別 ---
target_classes = ["Low Risk", "High Risk"] 

# --- NEW: 設定 Matplotlib 字體以支援中文 ---
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS'] # 嘗試多個字體，優先使用找到的第一個
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示為方塊的問題

print(f"--- 正在從 '{csv_file_path}' 讀取資料 ---")

try:
    df_results = pd.read_csv(csv_file_path)

    # --- 清理數據 ---
    df_results['Actual_Completion'] = df_results['Actual_Completion'].astype(str).str.strip()
    df_results['Model_Predicted_Label'] = df_results['Model_Predicted_Label'].astype(str).str.strip()
    # 將 'nan' 字串替換為真正的 NaN，以便後續處理
    df_results['Model_Predicted_Label'] = df_results['Model_Predicted_Label'].replace('nan', np.nan) 
    
    # 確保 Model_Full_Response 也是字串類型，並處理可能的 NaN
    df_results['Model_Full_Response'] = df_results['Model_Full_Response'].astype(str).fillna('') # 將 NaN 填充為空字串，避免錯誤

    print("--- 根據 'Model_Full_Response' 調整 'Model_Predicted_Label' ---")
    
    # 創建一個新的列表來儲存調整後的標籤
    adjusted_predicted_labels = []
    
    # 遍歷每一行數據來調整標籤
    for index, row in df_results.iterrows():
        full_response = row['Model_Full_Response']
        original_label = row['Model_Predicted_Label'] # 保留原始標籤以備不變動時使用
        
        # 將全響應轉為小寫，方便不區分大小寫匹配
        full_response_lower = full_response.lower() 
        
        if "high r" in full_response_lower:
            adjusted_predicted_labels.append("High Risk")
        elif "low r" in full_response_lower:
            adjusted_predicted_labels.append("Low Risk")
        else:
            # 如果沒有匹配到 "high" 或 "low"，則使用原始的 Model_Predicted_Label (即使它是 NaN)
            # 這裡需要注意：如果原始標籤是 NaN，且沒有匹配到 High/Low，它會保持 NaN
            adjusted_predicted_labels.append(original_label) 
            
    # 將調整後的標籤賦值回 DataFrame 的 'Model_Predicted_Label' 欄位
    df_results['Model_Predicted_Label'] = adjusted_predicted_labels
    
    # 在這裡，我們再次確保 NaN 被正確處理，即使經過上面的調整
    df_results['Model_Predicted_Label'] = df_results['Model_Predicted_Label'].replace('nan', np.nan)

    # 篩選出只包含目標類別的數據
    # 這一步確保只有 'Low Risk' 和 'High Risk' 的標籤（實際和預測）會被納入計算
    filtered_df = df_results[
        (df_results['Actual_Completion'].isin(target_classes)) &
        (df_results['Model_Predicted_Label'].isin(target_classes) & df_results['Model_Predicted_Label'].notna())
    ].copy() # 使用 .copy() 避免 SettingWithCopyWarning

    true_labels = filtered_df['Actual_Completion'].tolist()
    predicted_labels = filtered_df['Model_Predicted_Label'].tolist()

    print(f"\n共讀取 {len(df_results)} 條記錄。")
    print(f"其中 {len(filtered_df)} 條記錄被納入混淆矩陣計算 (排除非預期標籤)。")

    # --- 繪製混淆矩陣 ---
    if len(true_labels) > 0 and len(predicted_labels) > 0:
        print("\n--- 混淆矩陣 (2x2) ---")

        # 由於我們只關注 'Low Risk' 和 'High Risk'，所以 confusion_matrix 的 labels 參數很重要
        cm = confusion_matrix(true_labels, predicted_labels, labels=target_classes)
        
        # 為了報告和視覺化，創建一個 DataFrame
        cm_df = pd.DataFrame(cm, index=[f'Actual: {l}' for l in target_classes],
                             columns=[f'Predicted: {l}' for l in target_classes])

        print(cm_df) # 再次確認控制台輸出完整

        # --- 關鍵調整：強制數字顏色為黑色，增大圖形尺寸和DPI ---
        plt.figure(figsize=(8, 7), dpi=300) 
        sns.heatmap(cm_df, annot=True, fmt='d',
                    cmap='Blues', 
                    cbar=False,
                    linewidths=0.8, 
                    linecolor='black', 
                    annot_kws={"size": 18, "color": "black"}) 

        plt.title('confusion matrix ', fontsize=20) 
        plt.xlabel('Predicted label', fontsize=16) 
        plt.ylabel('Actual label', fontsize=16)
        plt.xticks(fontsize=14) 
        plt.yticks(fontsize=14, rotation=0) 
        plt.tight_layout()

        # --- 將圖形保存到文件 ---
        output_image_path = "confusion_matrix_output.png"
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight') 
        print(f"\n混淆矩陣圖片已保存至: {output_image_path}")

        plt.show() # 顯示圖形視窗

        print("\n--- 分類報告 ---")
        # 確保 classification_report 也是針對我們關心的 target_classes
        print(classification_report(true_labels, predicted_labels, labels=target_classes, zero_division=0))

    else:
        print("\n沒有足夠的有效資料來計算混淆矩陣 (CSV 中沒有符合目標類別的記錄)。")

except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{csv_file_path}'。請檢查路徑是否正確。")
except KeyError as e:
    print(f"錯誤：CSV 檔案中缺少必要的欄位。請確認存在 '{e}' 欄位。")
except Exception as e:
    print(f"發生意外錯誤：{e}")

print("\n--- 腳本執行完成 ---")
#%%
"""
import pandas as pd

# 讀取CSV檔案
df = pd.read_csv("level3_model_inference_results.csv",encoding='unicode_escape')
#print(df["Model_Predicted_Label"])
# 篩選出無法回答的樣本
na_df = df[df["Model_Predicted_Label"]== "NaN"]
print(len(na_df))
# 定義每個level的關鍵詞對應
def detect_level(prompt):
    if "Identify whether this ESG statement might be misleading" in prompt:
        return "Level 1"
    elif "Examine whether the ESG claims in this report align with measurable actions" in prompt:
        return "Level 2"
    elif "Analyze the following ESG disclosure. Does it demonstrate potential greenwashing" in prompt:
        return "Level 3"
    else:
        return "Unknown"

# 新增欄位以標註 Prompt Level
na_df["Prompt_Level"] = na_df["Prompt"].apply(detect_level)

# 統計各 Level 的 N/A 數量
na_counts = na_df["Prompt_Level"].value_counts()

# 顯示結果
print("無法回答的樣本數量（依不同提問級別）:")
print(na_counts)
"""
