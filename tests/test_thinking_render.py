"""Rendering of Qwen3.6 thinking blocks: gray reasoning, hidden markers."""

from wren.cli import _styled_chunks, _styled_final_text


def test_thinking_disabled_passes_through():
    out = list(_styled_chunks(["plain ", "text"], False))
    assert all(style is None for _, style in out)
    assert "".join(t for t, _ in out) == "plain text"
    assert _styled_final_text("plain text", False).plain == "plain text"


def test_stream_dims_reasoning_and_hides_end_marker():
    chunks = [
        "The user is asking a ",
        "question.\n</think>\n",
        "Here is the answer.",
    ]
    out = list(_styled_chunks(chunks, True))
    assert "</think>" not in "".join(t for t, _ in out)
    styles = [s for _, s in out]
    assert "dim" in styles and None in styles
    assert styles.index("dim") < styles.index(None)


def test_stream_handles_marker_split_across_chunks():
    out = list(_styled_chunks(["reasoning text </", "think>\nANSWER"], True))
    assert "</think>" not in "".join(t for t, _ in out)
    assert "".join(t for t, _ in out) == "reasoning text \nANSWER"


def test_stream_drops_leading_thinking_marker():
    out = list(_styled_chunks(["<think>\nreasoning\n</think>\nANS"], True))
    assert "<think>" not in "".join(t for t, _ in out)
    assert "".join(t for t, _ in out) == "\nreasoning\n\nANS"


def test_final_text_strips_markers_and_dims_reasoning():
    t = _styled_final_text("reasoning here\n</think>\nAnswer.", True)
    assert t.plain == "reasoning here\n\nAnswer."
    # reasoning span is dim; the answer carries no style
    assert [(s.start, s.end, s.style) for s in t.spans] == [(0, 15, "dim")]
