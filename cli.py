"""Ask the agent a question from the terminal and see every step it takes.

    python cli.py "How many documents did Jan de Vries upload?"
    python cli.py            (interactive mode)
"""
import json
import logging
import sys

from app.agent import run_agent

logging.basicConfig(level=logging.INFO, format="%(message)s")


def ask(question: str) -> None:
    result = run_agent(question)
    print("\n--- steps ---")
    for item in result["trace"]:
        print(f"[{item['step']}] {item['tool']}({json.dumps(item['arguments'], ensure_ascii=False)}) "
              f"{'ok' if item['ok'] else 'ERROR'} {item['duration_ms']} ms")
    print(f"\n--- answer ({result['steps']} model calls, {result['stopped']}) ---\n{result['answer']}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        while (q := input("Question (empty to quit): ").strip()):
            ask(q)
