"""
临时查看脚本：抽样看几条表格类型的解析结果
运行：python check_tables.py
"""
import json

with open("data/parsed_content.json", encoding="utf-8") as f:
    data = json.load(f)

tables = [c for c in data if c["type"] == "table"]
print(f"共 {len(tables)} 个表格块\n")

for t in tables[5:8]:
    print(f"[第{t['page']}页]")
    print(t["content"][:300])
    print("---")
