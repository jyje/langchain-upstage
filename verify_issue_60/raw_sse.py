"""Issue #60 re-verification (raw SSE, single model).

Reproduces https://github.com/langchain-ai/langchain-upstage/issues/60 at the
raw OpenAI-compatible SSE level, where the original defect lived (the server's
`delta.content` chunks arrived doubled, e.g. ' The The', 'getget').

Two scenarios, driven by env MODEL:
  S1 — stream WITH tools, exact Korean prompt from the issue (server-level repro)
  S3 — feed the tool result back and stream a natural-language answer while tools
       stay bound (decisive: forces streamed content in a tool context, so a
       lingering doubling defect could not hide behind "no content emitted")

Prints every inference's full reconstructed content, plus a per-chunk doubling
ratio and verdict. Appends a markdown report to $GITHUB_STEP_SUMMARY when set.

Run:  MODEL=solar-pro2 UPSTAGE_API_KEY=... python raw_sse.py
"""

from __future__ import annotations

import json
import os

import requests

BASE_URL = "https://api.upstage.ai/v1/solar"
API_KEY = os.environ["UPSTAGE_API_KEY"]
MODEL = os.environ.get("MODEL", "solar-pro2")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Retrieves the stock price for a specific ticker",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g. 'AAPL').",
                    }
                },
                "required": ["ticker"],
            },
        },
    }
]
# Exact prompt from issue #60.
PROMPT = "아마존과 구글의 주가는 어떻게 되나?"

_summary_lines: list[str] = []


def emit(line: str = "") -> None:
    print(line)
    _summary_lines.append(line)


def is_doubled(s: str) -> bool:
    """True if a chunk looks like one token emitted twice (the #60 symptom)."""
    if not s:
        return False
    parts = s.split(" ")
    if len(parts) >= 2 and parts[-1] and parts[-1] == parts[-2]:
        return True
    if len(s) % 2 == 0 and s[: len(s) // 2] == s[len(s) // 2 :]:
        return True
    return False


def post(payload: dict, stream: bool):
    return requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        stream=stream,
        timeout=90,
    )


def stream_content(payload: dict) -> tuple[list[str], bool, str]:
    """Return (content_chunks, saw_tool_calls, resolved_model)."""
    resp = post(payload, stream=True)
    resp.raise_for_status()
    chunks: list[str] = []
    saw_tools = False
    resolved = payload["model"]
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        resolved = obj.get("model", resolved)
        for choice in obj.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("tool_calls"):
                saw_tools = True
            c = delta.get("content")
            if c:
                chunks.append(c)
    return chunks, saw_tools, resolved


def report(scenario: str, chunks: list[str], extra: str = "") -> bool:
    total = len(chunks)
    doubled = sum(1 for c in chunks if is_doubled(c))
    ratio = (doubled / total * 100) if total else 0.0
    reproduces = ratio >= 50 and total >= 4
    verdict = "REPRODUCES (doubling)" if reproduces else "clean"
    emit(f"### {scenario}")
    if extra:
        emit(extra)
    emit(f"- content chunks: **{total}**, doubled: **{doubled}** ({ratio:.0f}%)")
    emit(f"- verdict: **{verdict}**")
    emit("")
    emit("Full inference output:")
    emit("```")
    emit("".join(chunks) if chunks else "(no content streamed)")
    emit("```")
    emit("")
    return reproduces


def main() -> int:
    emit(f"## `{MODEL}` — issue #60 raw-SSE re-verification")
    emit("")

    # S1 — exact repro: stream with tools bound.
    chunks, saw_tools, resolved = stream_content(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "tools": TOOLS,
            "stream": True,
        }
    )
    emit(f"resolved model: `{resolved}`")
    emit("")
    r1 = report(
        "S1 — stream WITH tools (exact issue repro)",
        chunks,
        extra=f"- tool_calls streamed: **{'yes' if saw_tools else 'no'}**",
    )

    # S3 — decisive: get tool calls, feed results back, stream the answer.
    t1 = post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "tools": TOOLS,
        },
        stream=False,
    )
    t1.raise_for_status()
    msg = t1.json()["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []
    r3 = False
    if not tool_calls:
        emit("### S3 — content-with-tools round-trip")
        emit("- model did not request a tool call; scenario not applicable")
        emit("")
    else:
        messages = [
            {"role": "user", "content": PROMPT},
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            },
        ]
        for tc in tool_calls:
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": "123.45"}
            )
        chunks3, _, _ = stream_content(
            {"model": MODEL, "messages": messages, "tools": TOOLS, "stream": True}
        )
        r3 = report(
            "S3 — stream CONTENT while tools bound (tool-result round-trip)",
            chunks3,
            extra=f"- tool_calls in turn 1: **{len(tool_calls)}**",
        )

    overall = "REPRODUCES" if (r1 or r3) else "NOT REPRODUCED"
    emit(f"## Overall for `{resolved}`: **{overall}** (as of check date)")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(_summary_lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
