"""Issue #60 re-verification (faithful langchain repro, single model).

Mirrors the exact reporter script in issue #60 (ChatUpstage + bind_tools +
.stream) so the result is stated in the same terms the issue was filed in.
Driven by env MODEL. Appends a markdown report to $GITHUB_STEP_SUMMARY.

Run:  MODEL=solar-pro2 UPSTAGE_API_KEY=... python langchain_repro.py
"""

from __future__ import annotations

import os

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_upstage import ChatUpstage

MODEL = os.environ.get("MODEL", "solar-pro2")
PROMPT = "아마존과 구글의 주가는 어떻게 되나?"

_summary_lines: list[str] = []


def emit(line: str = "") -> None:
    print(line)
    _summary_lines.append(line)


@tool(parse_docstring=True)
def get_stock_price(ticker: str) -> float:
    """Retrieves the stock price for a specific ticker.

    Args:
        ticker: The stock ticker symbol (e.g. 'AAPL' for Apple Inc.).
    """
    import random

    return random.random() * 1000


def is_doubled(s: str) -> bool:
    if not s:
        return False
    parts = s.split(" ")
    if len(parts) >= 2 and parts[-1] and parts[-1] == parts[-2]:
        return True
    if len(s) % 2 == 0 and s[: len(s) // 2] == s[len(s) // 2 :]:
        return True
    return False


def main() -> int:
    emit(f"## `{MODEL}` — issue #60 langchain `ChatUpstage.stream` repro")
    emit("")
    tool_llm = ChatUpstage(model=MODEL).bind_tools([get_stock_price])

    chunks: list[str] = []
    tool_call_chunks = 0
    for m in tool_llm.stream([HumanMessage(PROMPT)]):
        if getattr(m, "tool_call_chunks", None):
            tool_call_chunks += 1
        if m.content:
            chunks.append(m.content if isinstance(m.content, str) else str(m.content))

    total = len(chunks)
    doubled = sum(1 for c in chunks if is_doubled(c))
    ratio = (doubled / total * 100) if total else 0.0
    reproduces = ratio >= 50 and total >= 4
    verdict = "REPRODUCES (doubling)" if reproduces else "clean"

    emit("### S2 — langchain stream with bind_tools")
    emit(f"- tool_call chunks: **{tool_call_chunks}**")
    emit(f"- content chunks: **{total}**, doubled: **{doubled}** ({ratio:.0f}%)")
    emit(f"- verdict: **{verdict}**")
    emit("")
    emit("Full `.content` stream:")
    emit("```")
    emit("".join(chunks) if chunks else "(no content streamed)")
    emit("```")
    emit("")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(_summary_lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
