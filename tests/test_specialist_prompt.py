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


DEMO_20 = """Do EVERY numbered section separately. Keep the headers.

1) : In 3 bullets, what is an API gateway?

2) : Write a Python function `is_palindrome(s: str) -> bool` with type hints and a doctest.

3) Fix this bug and explain it:
def first(xs):
    return xs[0]
print(first([]))

4)A tank is 3/4 full. 20 liters are removed and it becomes 2/3 full. Find the capacity. Show steps.

5) There are 5 houses in a row. The green house is left of the white. Who owns the fish?

6)  Summarize in 2 sentences: Cloudflare Workers AI and Groq are free-tier endpoints.

7) Write an 8-line free-verse poem titled "Orchestra of Models".

8) Translate to Hindi, French, Spanish, and Arabic:
"Route each subtask to the best free model."

9)Return JSON with keys name, city, stack from:
"Ananya Rao in Hyderabad builds FastAPI + React orchestration demos."

10)  Compare Cloudflare vs Groq vs Cohere for a student demo. Markdown table.

11Friendly 4-sentence welcome for a capstone viva evaluator watching Routing Theater.

12): Describe what a dashboard chart comparing routed vs baseline cost in INR would show.

13) : Write SQL to find the top 5 models by request count from table request_logs(model, id).

14) : Why does `['10','10','10'].map(parseInt)` return [10, NaN, 2]? Fix it.

15) Solve 2x+5=17 and check the answer.

16) REASONING: Why can a weaker free model beat a larger model on translation but lose on multi-step math?

17) TRANSLATION (Arabic focus): Translate to Arabic only: "Command models handle multilingual chat well."

18)  Invent a 1-paragraph product myth about "Tiny Aya" models as elemental spirits.

19) From this log, extract model, provider, latency_ms as JSON:
"model=cf-gpt-oss-120b provider=cloudflare latency_ms=2200 success=true"

20) Rank these for coding specialty: Qwen2.5 Coder, Kimi K2.7 Code, North Mini Code, Llama 3.1 8B.
"""


def test_demo_20_section_decompose_count_and_content():
    parts = decompose(DEMO_20)
    assert 18 <= len(parts) <= 22
    blob = "\n".join(p.text for p in parts)
    assert "tank" in blob.lower() or "capacity" in blob.lower()
    assert "Ananya Rao" in blob
    assert "viva evaluator" in blob.lower() or "Routing Theater" in blob
    assert "dashboard chart" in blob.lower()
    assert "request_logs" in blob
    assert any(p.task_type == "reasoning" and "REASONING" in p.text for p in parts)


def test_leading_reasoning_label_beats_translation_keyword():
    task, why = classify_segment(
        "16) REASONING: Why can a weaker free model beat a larger model on translation "
        "but lose on multi-step math?"
    )
    assert task == "reasoning"
    assert why == "leading_label"


def test_clean_strips_not_part_of_assigned_task_wall():
    messy = (
        "An API gateway is an entry point for clients.\n"
        "It handles auth and routing.\n"
        "2\n"
        "This section is not part of the assigned task.\n"
        "3\n"
        "This section is not part of the assigned task.\n"
        "26\n"
        "This section is not part of the assigned task.\n"
    )
    cleaned = clean_specialist_text(messy)
    assert "API gateway" in cleaned
    assert "auth and routing" in cleaned
    assert "not part of the assigned task" not in cleaned.lower()
    assert "\n2\n" not in cleaned and not cleaned.strip().endswith("2")
