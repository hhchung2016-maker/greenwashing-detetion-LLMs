# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:05:55 2025

@author: henry
"""
import os
import json
import matplotlib.pyplot as plt

# 訓練階段資料夾（可依實際修改）
stages = ["1", "2", "3"]
base_path = "./fine-tuned/"

# 儲存每個階段的資料
all_steps = []
all_train_losses = []
all_eval_losses = []

for stage in stages:
    path = os.path.join(base_path, stage, "trainer_state.json")
    with open(path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    steps = []
    train_loss = []
    eval_loss = []

    for entry in log_data["log_history"]:
        if "loss" in entry:
            steps.append(entry["step"])
            train_loss.append(entry["loss"])
        elif "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])

    all_steps.append(steps[:len(train_loss)])
    all_train_losses.append(train_loss)
    all_eval_losses.append(eval_loss)

# 繪圖
plt.figure(figsize=(10, 6))
for i, stage in enumerate(stages):
    plt.plot(all_steps[i], all_train_losses[i], label=f"Level {stage} - Train Loss")
    if all_eval_losses[i]:
        # 簡單方式：將 eval_loss 等距畫在後面 (近似)
        eval_steps = all_steps[i][-len(all_eval_losses[i]):]
        plt.plot(eval_steps, all_eval_losses[i], linestyle='--', label=f"Level {stage} - Eval Loss")

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss Across Levels")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
import json
import matplotlib.pyplot as plt

log_path = "D:/中正授權軟體/ESG report/fine-tuned/trainer_state.json"

with open(log_path, "r", encoding="utf-8") as f:
    logs = json.load(f)

log_history = logs["log_history"]

steps = []
train_loss = []
eval_loss = []

for entry in log_history:
    if "loss" in entry:
        steps.append(entry["step"])
        train_loss.append(entry["loss"])
    elif "eval_loss" in entry:
        eval_loss.append(entry["eval_loss"])

plt.plot(steps[:len(train_loss)], train_loss, label="Train Loss")
if eval_loss:
    eval_steps = steps[-len(eval_loss):]
    plt.plot(eval_steps, eval_loss, label="Eval Loss")

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
