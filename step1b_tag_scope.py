"""
后处理脚本：给每个内容块打上"口径"标签（合并 / 母公司 / 未知）
原理：财报里"合并资产负债表""母公司资产负债表"这类标题出现后，
后续几页的表格数据都属于该口径，直到出现下一个标题为止。
按页码顺序扫描文本内容，维护一个"当前口径"状态，依次给每页打标签。

运行前：确保 data/parsed_content.json 已存在（即step1已跑完）
运行：python step1b_tag_scope.py
"""

import json
from collections import Counter

INPUT_PATH = "data/parsed_content.json"
OUTPUT_PATH = "data/parsed_content_tagged.json"

CONSOLIDATED_KEYWORDS = ["合并资产负债表", "合并利润表", "合并现金流量表", "合并股东权益变动表"]
PARENT_KEYWORDS = ["母公司资产负债表", "母公司利润表", "母公司现金流量表", "母公司股东权益变动表"]


def detect_scope(text: str):
    for kw in CONSOLIDATED_KEYWORDS:
        if kw in text:
            return "合并"
    for kw in PARENT_KEYWORDS:
        if kw in text:
            return "母公司"
    return None


def tag_scopes():
    with open(INPUT_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    pages = sorted(set(c["page"] for c in chunks))

    current_scope = "未知"
    page_scope = {}

    for p in pages:
        # 用该页的文本内容（不含表格）来判断本页是否出现了新的口径标题
        page_texts = [c["content"] for c in chunks if c["page"] == p and c["type"] == "text"]
        combined = "\n".join(page_texts)
        detected = detect_scope(combined)
        if detected:
            current_scope = detected
        page_scope[p] = current_scope

    for c in chunks:
        c["scope"] = page_scope[c["page"]]
        # 把口径信息拼进表格内容开头，让embedding能感知到这个上下文
        if c["type"] == "table" and c["scope"] != "未知":
            c["content"] = f"[口径：{c['scope']}] " + c["content"]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    scope_counts = Counter(c["scope"] for c in chunks if c["type"] == "table")
    print("表格口径分布：", dict(scope_counts))
    print(f"已保存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    tag_scopes()
