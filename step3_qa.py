"""
第三步：问答主程序
流程：用户提问 -> 问题向量化 -> FAISS检索最相关的几个内容块 -> 拼接prompt -> 调用GLM生成回答
运行前：确保 step2_build_index.py 已经跑完（data/faiss.index 和 data/chunk_meta.json 存在）
运行：python step3_qa.py
"""

import json
import numpy as np
import faiss
from zhipuai import ZhipuAI

from config import ZHIPU_API_KEY

INDEX_PATH = "data/faiss.index"
META_PATH = "data/chunk_meta.json"

EMBEDDING_MODEL = "embedding-3"
CHAT_MODEL = "glm-4-flash"
TOP_K = 5  # 检索几个最相关的内容块

client = ZhipuAI(api_key=ZHIPU_API_KEY)

# 加载索引和元数据（只需要加载一次）
print("加载索引中...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, encoding="utf-8") as f:
    chunk_meta = json.load(f)
print(f"索引加载完成，共 {len(chunk_meta)} 个内容块\n")


def get_embedding(text: str):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def retrieve(question: str, top_k: int = TOP_K, over_fetch: int = 15):
    """把问题向量化，去FAISS里检索候选内容块，再按口径做重排序：
    如果用户问题没有明确提到'母公司'，就把标记为'母公司'口径的候选降权，
    避免仅因为向量相似度差一点点就选错口径——这个逻辑不依赖LLM是否遵守prompt指令，
    而是在检索阶段就用规则锁定，更可靠。"""
    q_vec = np.array([get_embedding(question)], dtype="float32")
    faiss.normalize_L2(q_vec)

    scores, indices = index.search(q_vec, over_fetch)

    prefer_consolidated = "母公司" not in question

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunk_meta[idx]
        scope = chunk.get("scope", "未知")

        adjusted_score = float(score)
        if prefer_consolidated and scope == "母公司":
            adjusted_score *= 0.7  # 降权，除非用户明确要母公司口径

        candidates.append({
            "page": chunk["page"],
            "type": chunk["type"],
            "content": chunk["content"],
            "scope": scope,
            "score": adjusted_score
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def build_prompt(question: str, retrieved_chunks):
    """把检索到的内容拼成prompt，要求模型只根据这些内容回答"""
    context_parts = []
    for i, c in enumerate(retrieved_chunks, start=1):
        context_parts.append(f"[资料{i}，来自第{c['page']}页]\n{c['content']}")
    context = "\n\n".join(context_parts)

    prompt = f"""你是一个财报分析助手。请仅根据下面提供的资料回答用户问题。
如果资料中没有足够信息回答问题，请明确说"根据提供的资料无法回答这个问题"，不要编造。
回答时请注明信息来自第几页。

如果资料中同一指标（如总资产、净利润、现金流量净额等）同时出现"[口径：合并]"和"[口径：母公司]"
两种标注的数据，除非用户在问题中明确要求母公司口径，否则默认使用"[口径：合并]"标注的数据回答，
并在回答中注明使用的是合并口径数据。

【重要】如果资料中已经直接给出了比例、差值、增长率等数据（比如表格里的"变动比例(%)"这一列），
必须直接引用这个现成数值，禁止自己重新计算，因为你的心算容易出错。
只有当资料中确实没有现成的比例/差值数据、必须自己计算时，才允许计算，
但必须先完整写出算式和用到的两个原始数字（比如"84,788,592,077.34 - 76,298,833,033.93 = "），
再给出计算结果，方便核对，不要只报一个数字不写过程。

【参考资料】
{context}

【用户问题】
{question}

【回答】"""
    return prompt


def answer_question(question: str):
    retrieved = retrieve(question)
    prompt = build_prompt(question, retrieved)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # 财报问答要求准确，温度调低减少发散
    )

    answer = response.choices[0].message.content
    return answer, retrieved


if __name__ == "__main__":
    print("=== 财报智能问答系统（输入 exit 退出）===\n")
    while True:
        question = input("请输入问题：").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue

        answer, retrieved = answer_question(question)

        print("\n--- 回答 ---")
        print(answer)

        print("\n--- 检索到的参考资料（用于核对）---")
        for c in retrieved:
            print(f"  [第{c['page']}页, 相似度{c['score']:.3f}] {c['content'][:60]}...")
        print()
