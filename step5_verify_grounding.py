"""
第五步：数据溯源校验
对每道题的回答，提取里面的具体数字（带小数点/逗号/百分号的，通常是财务数据），
去检索到的原文里做字符串匹配，检查这个数字是不是真的"有据可查"。
这是比人工判断更严谨的幻觉检测方式。
运行前：确保 step3_qa.py 在同一目录下
运行：python step5_verify_grounding.py
"""

import re
from step3_qa import answer_question

QUESTIONS_PATH = "questions.txt"
RESULT_PATH = "data/grounding_check.txt"

# 匹配形如 8.24 / 115,393,310,976.69 / -18.94 / 249,462.64 的数字
NUMBER_PATTERN = re.compile(r'-?\d[\d,]*\.?\d*%?')


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def extract_numbers(text):
    """只提取'像财务数据'的数字：必须带逗号、小数点或百分号，
    过滤掉纯整数（比如'第18页'里的18、序号1234这种无意义数字）"""
    candidates = NUMBER_PATTERN.findall(text)
    meaningful = []
    for c in candidates:
        if ("," in c or "." in c or "%" in c):
            digits_only = re.sub(r"[^\d]", "", c)
            if len(digits_only) >= 2:
                meaningful.append(c)
    return meaningful


def is_grounded(number, context):
    """去掉百分号和负号后，检查这个数字是否作为子串出现在检索到的原文里"""
    clean = number.replace("%", "").lstrip("-")
    return clean in context


def run_check():
    questions = load_questions(QUESTIONS_PATH)
    report_lines = []
    total_numbers = 0
    grounded_numbers = 0

    for i, q in enumerate(questions, start=1):
        answer, retrieved = answer_question(q)
        context = "\n".join(c["content"] for c in retrieved)

        numbers = extract_numbers(answer)
        report_lines.append(f"【第{i}题】{q}")

        if not numbers:
            report_lines.append("（回答中未提取到具体数字，跳过校验）")
        else:
            for n in numbers:
                grounded = is_grounded(n, context)
                total_numbers += 1
                if grounded:
                    grounded_numbers += 1
                status = "✅ 可在检索资料中找到" if grounded else "⚠️ 未在检索资料中找到（疑似编造/计算得出）"
                report_lines.append(f"  {n}  →  {status}")

        report_lines.append("-" * 50)

    if total_numbers > 0:
        ratio = grounded_numbers / total_numbers * 100
        summary = (
            f"\n总计检测数字 {total_numbers} 个，"
            f"其中 {grounded_numbers} 个可在检索资料中直接找到\n"
            f"数据可溯源比例：{ratio:.1f}%"
        )
    else:
        summary = "\n未提取到任何可检测的具体数字"

    report_lines.append(summary)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"\n结果已保存到 {RESULT_PATH}")


if __name__ == "__main__":
    run_check()
