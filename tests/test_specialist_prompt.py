from app.decomposer import classify_segment, decompose, specialist_prompt
from app.schemas import PromptPart
from app.synthesizer import clean_specialist_text, stitch


def test_multi_part_specialist_prompt_is_isolated():
    part = PromptPart(
        id="p1",
        text="1) In 3 bullets, what is an API gateway?",
        task_type="qa",
    )
    siblings = [
        "p2 code: Write a Python palindrome function",
        "p3 debug: Fix first([])",
    ]
    prompt = specialist_prompt(
        part,
        original=(
            "Do EVERY numbered section separately.\n\n"
            "1) In 3 bullets, what is an API gateway?\n\n"
            "2) Write a Python function is_palindrome...\n\n"
            "3) Fix this bug..."
        ),
        sibling_summaries=siblings,
    )
    assert "Original user request" not in prompt
    assert "Write a Python palindrome" not in prompt
    assert "Fix first([])" not in prompt
    assert "Other parts of this request" not in prompt
    assert "API gateway" in prompt
    assert "not part of this slice" in prompt.lower()  # instruction forbids it
    assert "TASK:" in prompt


def test_single_part_specialist_prompt_uses_body():
    part = PromptPart(id="p1", text="What is HTTP?", task_type="qa")
    prompt = specialist_prompt(part, original="What is HTTP?", sibling_summaries=[])
    assert "What is HTTP?" in prompt
    assert "TASK:" not in prompt


def test_clean_specialist_text_strips_refusal_boilerplate():
    messy = (
        "An API gateway is a front door for APIs.\n"
        "This section is for code and not part of the qa slice.\n"
        "It handles auth and routing.\n"
        "3 This section is for debugging and not part of the qa slice.\n"
    )
    cleaned = clean_specialist_text(messy)
    assert "API gateway" in cleaned
    assert "auth and routing" in cleaned
    assert "not part of the qa slice" not in cleaned.lower()


def test_clean_strips_handled_separately_walkthrough():
    messy = (
        "1\n"
        "An API gateway is an entry point.\n"
        "2\n"
        "This section is for code and will be handled separately.\n"
        "3\n"
        "This section is for debugging and will be handled separately.\n"
        "11\n"
        "Wrong parseInt answer should also be removed if above pattern...\n"
    )
    # Lines 2/3 refusals + bare indexes drop; keep real gateway answer.
    cleaned = clean_specialist_text(
        "An API gateway is an entry point.\n"
        "2\n"
        "This section is for code and will be handled separately.\n"
        "3\n"
        "This section is for debugging and will be handled separately.\n"
    )
    assert "API gateway" in cleaned
    assert "handled separately" not in cleaned.lower()
    assert "this section is for" not in cleaned.lower()


def test_decompose_drops_instruction_preamble():
    prompt = (
        "Do EVERY numbered section separately. Keep the headers.\n\n"
        "1) In 3 bullets, what is an API gateway?\n\n"
        "2) Write a Python function is_palindrome.\n"
    )
    parts = decompose(prompt)
    assert not any("Do EVERY numbered section" in p.text for p in parts)
    assert any("API gateway" in p.text for p in parts)


def test_stitch_drops_refusal_lines():
    parts = [
        PromptPart(id="p1", text="What is an API gateway?", task_type="qa"),
        PromptPart(id="p2", text="Write SQL", task_type="code"),
    ]
    from app.schemas import PartOutput

    outputs = [
        PartOutput(
            part_id="p1",
            task_type="qa",
            text=(
                "Gateway answers here.\n"
                "This section is for code and not part of the qa slice."
            ),
            model="m",
            provider="p",
            success=True,
        ),
        PartOutput(
            part_id="p2",
            task_type="code",
            text="SELECT 1;",
            model="m",
            provider="p",
            success=True,
        ),
    ]
    combined, method = stitch(parts, outputs)
    assert method == "stitch"
    assert "Gateway answers here" in combined
    assert "SELECT 1" in combined
    assert "not part of the qa slice" not in combined.lower()


def test_classifier_cues_for_demo_sections():
    assert classify_segment("Find the capacity. A tank is 3/4 full. 20 liters.")[0] == "math"
    assert classify_segment("Why does ['10','10','10'].map(parseInt) return [10, NaN, 2]?")[0] == "debug"
    assert (
        classify_segment(
            "Write SQL to find the top 5 models by request count from table request_logs(model, id)."
        )[0]
        == "code"
    )
    assert (
        classify_segment(
            "Friendly 4-sentence welcome for a capstone viva evaluator watching Routing Theater."
        )[0]
        == "chat"
    )
    assert (
        classify_segment(
            'Invent a 1-paragraph product myth about "Tiny Aya" models as elemental spirits.'
        )[0]
        == "creative"
    )


def test_numbered_decompose_keeps_gateway_slice():
    prompt = (
        "Do EVERY numbered section separately. Keep the headers.\n\n"
        "1) In 3 bullets, what is an API gateway?\n\n"
        "2) Write a Python function `is_palindrome(s: str) -> bool` with type hints.\n\n"
        "3) Fix this bug and explain it:\ndef first(xs):\n    return xs[0]\n"
    )
    parts = decompose(prompt)
    assert len(parts) >= 3
    assert any("API gateway" in p.text for p in parts)
    assert any("is_palindrome" in p.text for p in parts)
