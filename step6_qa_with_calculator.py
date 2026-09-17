"""
第六步：给问答系统加一个"计算器工具"，解决LLM口算出错的问题
原理（Function Calling / 工具调用）：
1. 把问题+检索资料发给模型，同时告诉模型"如果需要做加减乘除，必须调用calculate工具，不能自己心算"
2. 如果模型决定要算数，它不会直接输出数字，而是返回一个"工具调用请求"（比如：减法，84788592077.34 - 76298833033.93）
3. 我们在Python里真正执行这个减法，把精确结果传回给模型
4. 模型基于这个"保证正确"的计算结果生成最终回答

运行前：确保 step2_build_index.py 已经跑完
运行：python step6_qa_with_calculator.py
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
TOP_K = 5

client = ZhipuAI(api_key=ZHIPU_API_KEY)

print("加载索引中...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, encoding="utf-8") as f:
    chunk_meta = json.load(f)
print(f"索引加载完成，共 {len(chunk_meta)} 个内容块\n")

# ---------- 工具定义 ----------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "对两个数字执行精确的加减乘除运算。任何涉及计算的地方（比如求差值、求比例）都必须调用这个工具，禁止自己心算，因为财务数据精度要求高。",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "运算类型：add=加法，subtract=减法(a-b)，multiply=乘法，divide=除法(a/b)"
                    },
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"}
                },
                "required": ["operation", "a", "b"]
            }
        }
    }
]


def calculate(operation: str, a: float, b: float):
    """真正执行计算的Python函数，保证精确"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        return a / b if b != 0 else None
    else:
        return None


# ---------- 检索逻辑（复用之前的口径重排序）----------

def get_embedding(text: str):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def retrieve(question: str, top_k: int = TOP_K, over_fetch: int = 15):
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
            adjusted_score *= 0.7
        candidates.append({
            "page": chunk["page"],
            "content": chunk["content"],
            "scope": scope,
            "score": adjusted_score
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def build_context(retrieved_chunks):
    parts = []
    for i, c in enumerate(retrieved_chunks, start=1):
        parts.append(f"[资料{i}，来自第{c['page']}页]\n{c['content']}")
    return "\n\n".join(parts)


SYSTEM_PROMPT = """你是一个财报分析助手。请仅根据提供的资料回答用户问题。
如果资料中没有足够信息回答问题，请明确说"根据提供的资料无法回答这个问题"，不要编造。
回答时请注明信息来自第几页。

如果资料中同一指标同时出现"[口径：合并]"和"[口径：母公司]"两种数据，除非用户明确要求母公司口径，
否则默认使用"[口径：合并]"的数据，并注明。

【重要】如果回答问题需要做任何加减乘除运算（比如求差值、算比例、算增长量），
你必须调用calculate工具来计算，不允许自己心算或直接输出算出来的数字，
因为财务数据要求精确到小数点后两位，人工心算容易出错。"""


def answer_question(question: str):
    retrieved = retrieve(question)
    context = build_context(retrieved)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】\n{question}"}
    ]

    # 第一轮：模型可能会请求调用calculate工具
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.1,
    )

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)

    calc_log = []  # 记录发生过的计算，方便调试和展示

    if tool_calls:
        # 把模型的这条消息（包含工具调用请求）加入对话历史
        messages.append(message)

        for call in tool_calls:
            args = json.loads(call.function.arguments)
            result = calculate(args["operation"], args["a"], args["b"])
            calc_log.append(f"{args['a']} {args['operation']} {args['b']} = {result}")

            # 把计算结果作为工具返回结果加入对话，供模型生成最终回答
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({"result": result}, ensure_ascii=False)
            })

        # 第二轮：模型基于精确计算结果，生成最终回答
        final_response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.1,
        )
        final_answer = final_response.choices[0].message.content
    else:
        final_answer = message.content

    return final_answer, retrieved, calc_log


if __name__ == "__main__":
    print("=== 财报智能问答系统（带精确计算器，输入 exit 退出）===\n")
    while True:
        question = input("请输入问题：").strip()
        if question.lower() in ("exit", "quit", "q"):
            break
        if not question:
            continue

        answer, retrieved, calc_log = answer_question(question)

        print("\n--- 回答 ---")
        print(answer)

        if calc_log:
            print("\n--- 本次调用的精确计算 ---")
            for log in calc_log:
                print(f"  {log}")

        print("\n--- 检索来源 ---")
        for c in retrieved:
            print(f"  第{c['page']}页 [{c['scope']}]")
        print()
