"""
查找包含关键财务指标（营业收入、净利润等）的表格块，看解析质量
运行：python check_key_table.py
"""
import json

with open("data/parsed_content.json", encoding="utf-8") as f:
    data = json.load(f)

tables = [c for c in data if c["type"] == "table"]

keywords = ["营业收入", "净利润", "总资产"]
matched = [t for t in tables if any(k in t["content"] for k in keywords)]

print(f"共找到 {len(matched)} 个包含关键财务指标的表格块\n")

for t in matched[:5]:
    print(f"[第{t['page']}页]")
    print(t["content"][:500])
    print("=" * 40)
