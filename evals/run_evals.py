"""Evaluation: does the agent choose the right tools and give the right answer?

This is what the Rabobank role calls "evaluation": instead of trying a few
questions by hand, you keep a fixed set of test questions with expectations
and run them after every change (new prompt, new model, new tool).

    python -m evals.run_evals
"""
import logging

from app.agent import run_agent

logging.basicConfig(level=logging.WARNING)

CASES = [
    {
        "question": "How many days per week can I work from home?",
        "expect_tools": ["search_documents"],
        "expect_in_answer": ["three", "3"],  # any of these is fine
    },
    {
        "question": "How many documents has Jan de Vries uploaded?",
        "expect_tools": ["query_records"],
        "expect_in_answer": ["2", "two"],
    },
    {
        "question": "Which documents are still in draft?",
        "expect_tools": ["query_records"],
        "expect_in_answer": ["DOC-004", "AI"],
    },
    {
        "question": "Summarise what the documents written by Sofia Martins say about reporting deadlines.",
        "expect_tools": ["query_records", "search_documents"],  # multi-step: find docs, then search in them
        "expect_in_answer": ["hour"],
    },
    {
        "question": "What is the company's policy on bringing dogs to the office?",
        "expect_tools": ["search_documents"],
        "expect_in_answer": ["don't know", "do not know", "not", "no information"],  # should NOT make things up
    },
]


def main() -> None:
    passed = 0
    for case in CASES:
        result = run_agent(case["question"])
        used = [t["tool"] for t in result["trace"]]
        answer = (result["answer"] or "").lower()

        tools_ok = all(tool in used for tool in case["expect_tools"])
        answer_ok = any(word.lower() in answer for word in case["expect_in_answer"])
        ok = tools_ok and answer_ok and result["stopped"] == "answered"
        passed += ok

        print(f"{'PASS' if ok else 'FAIL'}  {case['question']}")
        print(f"      tools used: {used}  (expected {case['expect_tools']})")
        if not ok:
            print(f"      answer: {result['answer']}")
    print(f"\n{passed}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
