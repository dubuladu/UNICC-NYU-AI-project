from flask import Flask, render_template, request, jsonify
import os, uuid, threading
from multimodal_analyzer import analyze_multimodal_stepwise

app = Flask(__name__)

# ✅ 用于记录每个分析任务的进度和结果
task_progress = {}
task_results = {}

@app.route("/")
def home():
    return render_template("index.html")

# ✅ 分析文本 - 更新以包含错误信息分析
@app.route("/analyze_text", methods=["POST"])
def analyze_text():
    from analyzer import analyze_article
    user_text = request.form["text_input"]
    
    # 创建一个唯一的任务ID
    task_id = str(uuid.uuid4())
    
    # 分析文本
    results = analyze_article(user_text)
    
    # 将结果存储在task_results中
    task_results[task_id] = results
    
    # 将task_id传递给模板
    return render_template("result.html", results=results, task_id=task_id)

# ✅ 启动音频/视频分析任务
@app.route("/start_analysis", methods=["POST"])
def start_analysis():
    file = request.files["media_file"]
    task_id = str(uuid.uuid4())

    # 保存上传文件
    filename = f"{task_id}_{file.filename}"
    save_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(save_path)

    # 初始化进度
    task_progress[task_id] = 0

    # 异步处理分析任务
    threading.Thread(target=process_file_with_progress, args=(save_path, task_id)).start()

    return jsonify({"task_id": task_id})

# ✅ 实时查询进度
@app.route("/check_progress/<task_id>")
def check_progress(task_id):
    progress = task_progress.get(task_id, 0)
    return jsonify({"progress": progress})

# ✅ 展示分析结果页面
@app.route("/result/<task_id>")
def show_result(task_id):
    results = task_results.get(task_id, [{"error": "No result found."}])
    return render_template("multimodal_result.html", results=results)

# ✅ 导出毒性内容或错误信息内容
@app.route("/export_problematic/<task_id>/<filter_type>")
def export_problematic(task_id, filter_type):
    results = task_results.get(task_id, [])
    if not results:
        return jsonify({"error": "No results found for this task."})
        
    # 根据不同的过滤类型导出不同的内容
    # filter_type可以是：toxic, misinfo, toxic_misinfo
    filtered_results = []
    
    if filter_type == "toxic":
        filtered_results = [r for r in results if r.get("label") in ["TOXIC", "SEVERE-TOXIC"]]
    elif filter_type == "misinfo":
        filtered_results = [r for r in results if r.get("misinfo_label") == "MISINFORMATION"]
    elif filter_type == "toxic_misinfo":
        filtered_results = [r for r in results if r.get("label") in ["TOXIC", "SEVERE-TOXIC"] and r.get("misinfo_label") == "MISINFORMATION"]
    
    return jsonify({
        "count": len(filtered_results),
        "results": filtered_results
    })

# ✅ 后台分析函数（在新线程中运行）
def process_file_with_progress(path, task_id):
    try:
        task_progress[task_id] = 5  # 上传完成
        from multimodal_analyzer import analyze_multimodal_stepwise
        results = analyze_multimodal_stepwise(path, task_id, task_progress)
        task_results[task_id] = results
        task_progress[task_id] = 100
    except Exception as e:
        task_results[task_id] = [{"error": str(e)}]
        task_progress[task_id] = 100


if __name__ == "__main__":
    print("✅ Flask running at http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=True)