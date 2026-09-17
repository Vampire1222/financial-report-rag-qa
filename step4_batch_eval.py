"""
第四步：批量效果评测
一次性把questions.txt里的所有问题跑一遍，打印每条的回答，
同时把结果存成一个文件，方便你之后人工核对打分、统计准确率。
运行前：确保 step3_qa.py 在同一目录下（会复用它里面的检索+生成逻辑）
运行：python step4_batch_eval.py
"""

from step3_qa import answer_question

QUESTIONS_PATH = "questions.txt"
RESULT_PATH = "data/eval_results.txt"


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_batch_eval():
    questions = load_questions(QUESTIONS_PATH)
    print(f"共 {len(questions)} 道题，开始批量测试...\n")
    print("=" * 60)

    lines = []  # 用来存结果文件

    for i, q in enumerate(questions, start=1):
        answer, retrieved = answer_question(q)

        pages = "、".join(f"第{c['page']}页" for c in retrieved)

        print(f"\n【第{i}题】{q}")
        print(f"回答：{answer}")
        print(f"检索来源：{pages}")
        print("-" * 60)

        lines.append(f"【第{i}题】{q}")
        lines.append(f"回答：{answer}")
        lines.append(f"检索来源：{pages}")
        lines.append("是否正确：____（人工填写：对/错/部分对）")
        lines.append("备注：____")
        lines.append("-" * 60)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n全部测试完成，结果已保存到 {RESULT_PATH}")
    print("打开这个文件，对照财报原文逐条填写'是否正确'，最后统计一下正确率")


if __name__ == "__main__":
    run_batch_eval()
