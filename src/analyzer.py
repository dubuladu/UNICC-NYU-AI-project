from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np

# 模型路径
model_path = "XLM_final_model"
misinfo_model_path = "misinfo_model"  # Add misinfo model path

# 加载本地 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

# 加载错误信息检测模型
misinfo_tokenizer = AutoTokenizer.from_pretrained(misinfo_model_path, use_fast=False, local_files_only=True)
misinfo_model = AutoModelForSequenceClassification.from_pretrained(misinfo_model_path, local_files_only=True)
misinfo_model.eval()

# ✅ 强制设置标签映射
model.config.id2label = {0: "NON-TOXIC", 1: "TOXIC", 2: "SEVERE-TOXIC"}
label_map = model.config.id2label

# ✅ 错误信息模型标签映射
misinfo_model.config.id2label = {0: "FACTUAL", 1: "MISINFORMATION"}
misinfo_label_map = misinfo_model.config.id2label

# 分析文章文本，返回每段的分析结果（包含毒性和错误信息分析）
def analyze_article(article_text):
    paragraphs = [para.strip() for para in article_text.split('\n') if para.strip()]
    results = []
    THRESHOLD = 0.5  # ✅ 你可以自己调这个置信度阈值

    for idx, para in enumerate(paragraphs):
        # 毒性分析
        inputs = tokenizer(
            para,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        none_toxic_prob = float(probs[0])
        toxic_prob = float(probs[1])
        severe_toxic_prob = float(probs[2])

        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])

        # ✅ 加入置信度阈值判断
        if confidence < THRESHOLD:
            label = "UNCERTAIN"
        else:
            label = label_map[max_idx]
            
        # 错误信息分析
        misinfo_inputs = misinfo_tokenizer(
            para,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            misinfo_outputs = misinfo_model(**misinfo_inputs)
            misinfo_probs = F.softmax(misinfo_outputs.logits, dim=1).cpu().numpy()[0]
            
        factual_prob = float(misinfo_probs[0])
        misinfo_prob = float(misinfo_probs[1])
        
        misinfo_max_idx = int(np.argmax(misinfo_probs))
        misinfo_confidence = float(misinfo_probs[misinfo_max_idx])
        
        # 错误信息置信度阈值判断
        if misinfo_confidence < THRESHOLD:
            misinfo_label = "UNCERTAIN"
        else:
            misinfo_label = misinfo_label_map[misinfo_max_idx]

        results.append({
            "paragraph": idx + 1,
            "text": para,  # 添加这一行
            "preview": para[:100].replace('\n', ' '),
            "label": label,
            "confidence": confidence,
            "none_toxic_prob": none_toxic_prob,
            "toxic_prob": toxic_prob,
            "severe_toxic_prob": severe_toxic_prob,
            # ✅ 新增：前端展示使用
            "display_prob": confidence,
            # ✅ 新增：错误信息分析结果
            "misinfo_label": misinfo_label,
            "misinfo_confidence": misinfo_confidence,
            "factual_prob": factual_prob,
            "misinfo_prob": misinfo_prob,
            "misinfo_display_prob": misinfo_confidence
        })

    return results