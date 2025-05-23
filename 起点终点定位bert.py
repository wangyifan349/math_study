import os
import numpy as np
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer

# -------------------- 1. 模型下载或加载 --------------------
def download_or_load_bert_model(model_name, local_dir):
    if not os.path.isdir(local_dir) or len(os.listdir(local_dir)) == 0:
        print("本地模型目录 {} 不存在或为空，正在下载模型...".format(local_dir))
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        os.makedirs(local_dir, exist_ok=True)
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
        print("模型已保存到本地目录 {}".format(local_dir))
    else:
        print("检测到本地模型目录 {}，跳过下载".format(local_dir))
    return local_dir

def load_sentence_transformer_model(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    print("加载语义嵌入模型 {}".format(model_name))
    model = SentenceTransformer(model_name)
    return model

# -------------------- 2. 设定配置 --------------------
BERT_MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"
LOCAL_BERT_DIR = "./bert_qa_model"
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("程序启动中...")

local_bert_dir = download_or_load_bert_model(BERT_MODEL_NAME, LOCAL_BERT_DIR)
qa_pipeline = pipeline("question-answering", model=local_bert_dir, tokenizer=local_bert_dir)
st_model = load_sentence_transformer_model(ST_MODEL_NAME)

# -------------------- 3. 知识库准备 --------------------
knowledge_docs = []

knowledge_docs.append("Paramecium is a genus of unicellular ciliates, commonly studied in biology due to their complex cell structure.")
knowledge_docs.append("DNA carries genetic instructions vital for the growth and function of living organisms.")
knowledge_docs.append("Photosynthesis converts light energy into chemical energy by plants.")
knowledge_docs.append("RNA plays roles in protein synthesis.")

print("计算知识库文本的语义向量...")
doc_embeddings = st_model.encode(knowledge_docs, convert_to_numpy=True, normalize_embeddings=True)

# -------------------- 4. 语义检索函数 --------------------
def semantic_search(question, top_k=3):
    q_embedding = st_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    similarity_scores = np.dot(doc_embeddings, q_embedding)

    sorted_indices = np.argsort(similarity_scores)

    # 获取top_k索引，降序排列
    topk_indices = []
    total_len = len(similarity_scores)
    for i in range(1, top_k+1):
        topk_indices.append(sorted_indices[total_len - i])

    # 准备结果列表
    results = []
    for idx in topk_indices:
        score = similarity_scores[idx]
        doc = knowledge_docs[idx]
        results.append((idx, score, doc))
    return results

# -------------------- 5. 问答函数 --------------------
def answer_with_position(question, top_k=3):
    retrieved = semantic_search(question, top_k)
    
    # 拼接上下文字符串（用换行连接）
    context = ""
    for item in retrieved:
        context += item[2] + "\n"
    context = context.strip()  # 移除末尾多余换行

    result = qa_pipeline(question=question, context=context)

    answer = result.get('answer', '')
    score = result.get('score', 0.0)
    start = result.get('start', -1)
    end = result.get('end', -1)

    return {
        "answer": answer,
        "score": score,
        "start": start,
        "end": end,
        "context": context,
        "retrieved_docs": retrieved
    }

# -------------------- 6. 交互问答 --------------------
print("\n离线语义检索 + BERT问答系统，输入 exit、退出 或 quit 结束")
print("-" * 60)

while True:
    question = input("请输入问题：").strip()
    if question.lower() == 'exit' or question == '退出' or question.lower() == 'quit':
        print("程序退出，感谢使用！")
        break

    if question == "":
        print("请输入有效问题！")
        continue

    result = answer_with_position(question)

    print("\n检索到的相关文档（按相关度排序）：")
    for item in result['retrieved_docs']:
        idx, score, doc = item
        print("[相似度 {:.4f}] {}".format(score, doc))

    print("\nBERT回答：{}".format(result['answer']))
    print("置信度：{:.4f}".format(result['score']))
    print("答案在上下文中的字符范围：[起点: {}, 终点: {}]".format(result['start'], result['end']))

    if 0 <= result['start'] < result['end'] and result['end'] <= len(result['context']):
        answer_slice = result['context'][result['start']:result['end']]
        print("上下文切片：{}".format(answer_slice))
    print("-" * 60)
