import ssl
import certifi

def create_verified_context(*args, **kwargs):
    context = ssl.create_default_context(*args, **kwargs)
    context.load_verify_locations(cafile=certifi.where())
    return context

ssl._create_default_https_context = create_verified_context

import os
import cv2
import torch
import whisper
import pytesseract
import numpy as np
import stanza
import filetype
import re
import torch.nn.functional as F
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import openai
from openai import OpenAI

pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
article_model_path = "ar_model"  
misinfo_model_path = "misinfo_model"

# 设置OpenAI API
client = OpenAI(api_key="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")  # 请替换为实际的API密钥

for lang in ['en', 'zh', 'fr', 'es', 'ru', 'ar']:
    stanza.download(lang, verbose=False)

lang_map = {
    "en": "en", "zh-cn": "zh", "zh": "zh", "fr": "fr",
    "es": "es", "ru": "ru", "ar": "ar"
}

def detect_language(text):
    try:
        lang = detect(text)
        return lang_map.get(lang, "en")
    except:
        return "en"

def load_stanza_tokenizer(lang_code):
    return stanza.Pipeline(lang=lang_code, processors='tokenize')

def get_sentences_by_language(text, lang_code, segments=None):
    if lang_code in ["zh", "ar", "ja", "ko"]:
        return [seg["text"].strip() for seg in segments if seg.get("text", "").strip()]
    elif lang_code in ["en", "fr", "es"]:
        nlp = load_stanza_tokenizer(lang_code)
        doc = nlp(text)
        return [s.text for s in doc.sentences if s.text.strip()]
    else:
        pattern = r'(?<=[.!?])\s+'
        return [s.strip() for s in re.split(pattern, text) if s.strip()]

# 毒性模型加载 - 使用article_final_model
print("🔄 Loading article final model...")
tokenizer = AutoTokenizer.from_pretrained(article_model_path, local_files_only=True, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(article_model_path, local_files_only=True)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# 修正标签映射
model.config.id2label = {0: "NON-TOXIC", 1: "TOXIC", 2: "SEVERE-TOXIC"}
label_map = model.config.id2label

# 错误信息模型加载
print("🔄 Loading misinformation model...")
misinfo_tokenizer = AutoTokenizer.from_pretrained(misinfo_model_path, local_files_only=True, use_fast=False)
misinfo_model = AutoModelForSequenceClassification.from_pretrained(misinfo_model_path, local_files_only=True)
misinfo_model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# 错误信息模型标签映射
misinfo_model.config.id2label = {0: "FACTUAL", 1: "MISINFORMATION"}
misinfo_label_map = misinfo_model.config.id2label

# GPT摘要生成函数
def generate_summary_with_gpt(text, lang_code='en'):
    """使用GPT生成摘要"""
    if not text or len(text.strip()) < 20:
        return "文本内容不足，无法生成摘要" if lang_code == 'zh' else "Insufficient text content for summary"
        
    # 限制文本长度
    max_text_length = 10000
    if len(text) > max_text_length:
        text_length = len(text)
        first_part = text[:int(max_text_length * 0.75)]
        last_part = text[text_length - int(max_text_length * 0.25):]
        text = first_part + "\n...[内容省略]...\n" + last_part
    
    prompt_templates = {
        'en': "Summarize the following article in 3-5 sentences, highlighting the main points and key information. IMPORTANT: Your summary must be in English:",
        'zh': "请用中文总结以下文章的主要内容，限制在3到5句话内，突出重点和关键信息。重要：你的摘要必须是中文：",
        'fr': "Résumez l'article suivant en 3-5 phrases, en soulignant les points principaux et les informations clés. IMPORTANT: Votre résumé doit être en français:",
        'ru': "Кратко изложите следующую статью в 3-5 предложениях, выделив основные моменты и ключевую информацию. ВАЖНО: Ваше резюме должно быть на русском языке:",
        'ar': "لخص المقال التالي في 3-5 جمل، مع التركيز على النقاط الرئيسية والمعلومات الأساسية. مهم: يجب أن يكون ملخصك باللغة العربية:",
        'es': "Resume el siguiente artículo en 3-5 frases, destacando los puntos principales y la información clave. IMPORTANTE: Tu resumen debe estar en español:"
    }
    
    template = prompt_templates.get(lang_code, prompt_templates['en'])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"你是一个精通多语言的摘要专家。请使用{lang_code}语言回答。"},
                {"role": "user", "content": template + f"\n{text}"}
            ],
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT摘要生成失败: {e}")
        return "摘要生成失败" if lang_code == 'zh' else "Failed to generate summary"

# GPT毒性原因分析函数
def analyze_toxic_reason_with_gpt(sentence, lang_code='en'):
    """使用GPT分析毒性内容的具体原因"""
    prompt_templates = {
        'en': "Explain briefly why the following statement is toxic. Choose from: racism, xenophobia, sexism, religious discrimination, ageism, homophobia, cultural oppression, classism, linguistic discrimination, other.",
        'zh': "请简要解释以下句子为什么具有毒性。从以下类别中选择：种族歧视、仇外情绪、性别歧视、宗教歧视、年龄歧视、性取向歧视、文化压迫、社会阶层歧视、语言歧视、其他。",
        'fr': "Expliquez brièvement pourquoi cette déclaration est toxique. Choisissez parmi : racisme, xénophobie, sexisme, discrimination religieuse, âgisme, homophobie, oppression culturelle, classisme, discrimination linguistique, autre.",
        'ru': "Кратко объясните, почему это утверждение токсично. Выберите из: расизм, ксенофобия, сексизм, религиозная дискриминация, эйджизм, гомофобия, культурное угнетение, классовая дискриминация, языковая дискриминация, другое.",
        'ar': "اشرح بإيجاز سبب اعتبار هذا التصريح سامًا. اختر من: العنصرية، كراهية الأجانب، التمييز الجنسي، التمييز الديني، التمييز العمري، رهاب المثلية، القمع الثقافي، التمييز الطبقي، التمييز اللغوي، أخرى.",
        'es': "Explica brevemente por qué esta declaración es tóxica. Elige entre: racismo, xenofobia, sexismo, discriminación religiosa, discriminación por edad, homofobia, opresión cultural, clasismo, discriminación lingüística, otro."
    }
    
    template = prompt_templates.get(lang_code, prompt_templates['en'])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"你是一个专门分析有害内容的专家。请使用{lang_code}语言回答。"},
                {"role": "user", "content": template + f"\n{sentence}"}
            ],
            timeout=15
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT毒性分析失败: {e}")
        return "无法分析" if lang_code == 'zh' else "Analysis failed"

def predict_toxicity(sentences):
    results = []
    device = model.device
    THRESHOLD = 0.5

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        max_index = int(np.argmax(probs))
        confidence = float(probs[max_index])

        # 先定义 label 变量
        if confidence < THRESHOLD:
            label = "UNCERTAIN"
        else:
            label = label_map[max_index]

        results.append({
            "text": sentence,
            "label": label,  # 现在 label 已经定义了
            "confidence": confidence,
            "display_prob": confidence,
            "probabilities": {
                "non_toxic": float(probs[0]),
                "toxic": float(probs[1]),
                "severe_toxic": float(probs[2])
            }
        })
    return results

def predict_misinformation(results):
    device = misinfo_model.device
    THRESHOLD = 0.5

    for i, result in enumerate(results):
        sentence = result["text"]
        inputs = misinfo_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = misinfo_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        factual_prob = float(probs[0])
        misinfo_prob = float(probs[1])
        
        misinfo_max_idx = int(np.argmax(probs))
        misinfo_confidence = float(probs[misinfo_max_idx])
        
        if misinfo_confidence < THRESHOLD:
            misinfo_label = "UNCERTAIN"
        else:
            misinfo_label = misinfo_label_map[misinfo_max_idx]
            
        results[i]["misinfo_label"] = misinfo_label
        results[i]["misinfo_confidence"] = misinfo_confidence
        results[i]["misinfo_display_prob"] = misinfo_confidence
        results[i]["probabilities"]["factual"] = factual_prob
        results[i]["probabilities"]["misinformation"] = misinfo_prob
        
    return results

def transcribe_audio(input_path):
    model = whisper.load_model("small")
    result = model.transcribe(input_path)
    return result["text"], result.get("segments", [])

def extract_ocr_from_video(video_path, frame_skip=30, batch_size=5):
    print("📺 Extracting OCR from video with enhanced cleaning...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    buffer = []
    ocr_texts = []
    seen_texts = set()

    def is_valid_text(t):
        if len(t) < 10:
            return False
        if sum(c.isalpha() for c in t) / (len(t) + 1e-5) < 0.4:
            return False
        if re.search(r'[~@<>{}=*\/|_\[\]^]', t):
            return False
        return True

    def is_duplicate(t):
        for seen in seen_texts:
            if t.lower() in seen or seen in t.lower():
                return True
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            text = pytesseract.image_to_string(frame, lang='eng').strip()
            if is_valid_text(text):
                buffer.append((count / fps, text))
        count += 1

        if len(buffer) >= batch_size:
            start_time = buffer[0][0]
            texts = [b[1] for b in buffer]
            merged = " ".join(texts).strip()
            lang = detect(merged) if len(merged) > 10 else "unknown"

            if is_valid_text(merged) and lang in lang_map and not is_duplicate(merged):
                cleaned = f"[{start_time:.2f}s] {merged}"
                ocr_texts.append(cleaned)
                seen_texts.add(merged.lower())
            buffer = []

    cap.release()
    return "\n".join(ocr_texts)

def analyze_multimodal_stepwise(file_path, task_id, task_progress):
    all_results = []
    full_text = ""  # 用于存储完整文本以生成摘要
    detected_type = filetype.guess(file_path)
    task_progress[task_id] = 10

    if not detected_type:
        return [{"error": "File type could not be detected."}], None
    
    mime_type = detected_type.mime

    if mime_type.startswith("audio"):
        task_progress[task_id] = 20
        text, segments = transcribe_audio(file_path)
        full_text = text

        task_progress[task_id] = 40
        lang = detect_language(text)
        sentences = get_sentences_by_language(text, lang, segments)

        task_progress[task_id] = 60
        results = predict_toxicity(sentences)
        
        task_progress[task_id] = 80
        results = predict_misinformation(results)
        
        # 对有毒性的句子分析原因
        for result in results:
            if result["label"] in ["TOXIC", "SEVERE-TOXIC"]:
                result["toxic_reason"] = analyze_toxic_reason_with_gpt(result["text"], lang)
            else:
                result["toxic_reason"] = None

        all_results += results
        task_progress[task_id] = 90

    elif mime_type.startswith("video"):
        task_progress[task_id] = 20
        text, segments = transcribe_audio(file_path)
        full_text = text

        task_progress[task_id] = 35
        lang = detect_language(text)
        sentences1 = get_sentences_by_language(text, lang, segments)
        
        results1 = predict_toxicity(sentences1)
        task_progress[task_id] = 50
        
        results1 = predict_misinformation(results1)
        
        # 对音频部分的有毒性句子分析原因
        for result in results1:
            if result["label"] in ["TOXIC", "SEVERE-TOXIC"]:
                result["toxic_reason"] = analyze_toxic_reason_with_gpt(result["text"], lang)
            else:
                result["toxic_reason"] = None
                
        task_progress[task_id] = 60
        all_results += results1

        ocr_text = extract_ocr_from_video(file_path)
        full_text += "\n" + ocr_text
        task_progress[task_id] = 70

        ocr_lang = detect_language(ocr_text)
        sentences2 = get_sentences_by_language(ocr_text, ocr_lang)
        
        results2 = predict_toxicity(sentences2)
        task_progress[task_id] = 80
        
        results2 = predict_misinformation(results2)
        
        # OCR部分不分析毒性原因
        for result in results2:
            result["toxic_reason"] = None
            
        task_progress[task_id] = 90
        all_results += results2

    else:
        return [{"error": "Unsupported file type. Only audio/video supported."}], None

    # 生成摘要
    summary = generate_summary_with_gpt(full_text, detect_language(full_text)) if full_text else None
    
    return all_results, summary