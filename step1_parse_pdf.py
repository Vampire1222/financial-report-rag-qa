"""
第一步：把财报PDF解析成文本
- 普通段落直接提取
- 表格单独提取并转成"字段：值"的可读格式（避免数字挤在一起看不懂）
运行前：pip install -r requirements.txt
运行：python step1_parse_pdf.py
"""

import pdfplumber
import json

PDF_PATH = "data/yili_2024.pdf"
OUTPUT_PATH = "data/parsed_content.json"


def table_to_readable_text(table):
    """
    把pdfplumber提取的表格（list of list）转成更可读的文本。
    策略：假设第一行是表头，后面每一行按 "表头: 值" 拼接。
    这是最简单的做法，financial表格结构复杂时可能需要更细致处理，
    但作为第一版够用——这也是你可以在项目里写"识别表格解析局限性"的地方。
    """
    if not table or len(table) < 2:
        return ""

    header = table[0]
    lines = []
    for row in table[1:]:
        parts = []
        for h, v in zip(header, row):
            if h and v:
                parts.append(f"{h.strip()}: {v.strip()}")
        if parts:
            lines.append("；".join(parts))
    return "\n".join(lines)


def parse_pdf(pdf_path):
    all_chunks = []  # 每个元素是 {"page": 页码, "type": "text"/"table", "content": 内容}

    with pdfplumber.open(pdf_path) as pdf:
        print(f"总页数：{len(pdf.pages)}")

        for page_num, page in enumerate(pdf.pages, start=1):
            # 提取表格（先提取表格，避免表格内容被当成普通文本重复提取）
            tables = page.extract_tables()
            table_texts = []
            for table in tables:
                readable = table_to_readable_text(table)
                if readable.strip():
                    all_chunks.append({
                        "page": page_num,
                        "type": "table",
                        "content": readable
                    })
                    table_texts.append(readable)

            # 提取普通文本
            text = page.extract_text() or ""
            if text.strip():
                all_chunks.append({
                    "page": page_num,
                    "type": "text",
                    "content": text.strip()
                })

            if page_num % 20 == 0:
                print(f"已处理 {page_num} 页...")

    return all_chunks


if __name__ == "__main__":
    chunks = parse_pdf(PDF_PATH)
    print(f"\n解析完成，共提取 {len(chunks)} 个内容块")

    text_count = sum(1 for c in chunks if c["type"] == "text")
    table_count = sum(1 for c in chunks if c["type"] == "table")
    print(f"其中：文本块 {text_count} 个，表格块 {table_count} 个")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n已保存到 {OUTPUT_PATH}")
    print("\n--- 抽样看几条结果 ---")
    for c in chunks[:3]:
        print(f"[第{c['page']}页 / {c['type']}]")
        print(c["content"][:200])
        print("---")
