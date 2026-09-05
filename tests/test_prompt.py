import pytest

from da_vllm.prompt import (
    ANSWER_SPEC,
    BOOTSTRAP_USER_TEXT,
    QUESTION_HEADER,
    SYSTEM_TEXT,
    TOOL_DECLARATION,
)


def test_system_turn_is_present_on_every_arm(family_case, filler):
    _, _, renderer = family_case
    for arm in ("da", "da_no_mask", "vanilla"):
        prompt = renderer.render(arm, filler, "Who?")
        assert SYSTEM_TEXT in prompt.text, arm
        assert prompt.messages[0]["role"] == "system"


def test_da_and_no_mask_render_byte_identical_prompts(family_case, filler):
    _, _, renderer = family_case
    a = renderer.render("da", filler, "Who?")
    b = renderer.render("da_no_mask", filler, "Who?")
    assert a.fingerprint == b.fingerprint


def test_tool_declaration_goes_through_the_chat_template(family_case, filler):
    _, _, renderer = family_case
    prompt = renderer.render_da(filler, "Who?")
    assert prompt.tools == (TOOL_DECLARATION,)
    assert "get_magic_chunk" in prompt.text
    # And the vanilla arm carries no tool declaration at all.
    assert "get_magic_chunk" not in renderer.render_vanilla(filler, "Who?").text


def test_tool_call_arguments_are_a_dict_not_a_json_string(family_case, filler):
    _, _, renderer = family_case
    messages, _ = renderer.da_messages(filler, "Who?")
    calls = [m for m in messages if m.get("tool_calls")]
    assert calls
    for m in calls:
        assert isinstance(m["tool_calls"][0]["function"]["arguments"], dict)


def test_transcript_shape(family_case, filler):
    _, _, renderer = family_case
    messages, segments = renderer.da_messages(filler, "Who?")
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == BOOTSTRAP_USER_TEXT
    body = messages[2:-1]
    assert len(body) == 2 * len(segments)
    for i, seg in enumerate(segments):
        call, response = body[2 * i], body[2 * i + 1]
        assert call["role"] == "assistant"
        assert call["tool_calls"][0]["id"] == f"call_{seg.index}"
        assert response["role"] == "tool"
        assert response["content"].startswith(f"Magic Chunk {seg.index}\n")
    assert messages[-1]["role"] == "user"


def test_question_header_is_the_last_marker_before_the_question(family_case, filler):
    _, _, renderer = family_case
    prompt = renderer.render_da(filler, "Which year?")
    at = prompt.text.rfind(QUESTION_HEADER)
    assert at >= 0
    after = prompt.text[at + len(QUESTION_HEADER) :].lstrip("\n")
    assert after.startswith("Which year?")


def test_magic_chunk_naming_avoids_collision_with_document_sections(family_case, filler):
    _, _, renderer = family_case
    prompt = renderer.render_da(filler, "Who?")
    assert "Magic Chunk 1" in prompt.text
    assert "magic chunk" in prompt.messages[-1]["content"].lower()


def test_instruction_prompt_keeps_the_load_bearing_scaffolding(family_case, filler):
    _, _, renderer = family_case
    instruction = renderer.render_da(filler, "Who?").messages[-1]["content"]
    # Removing the "1 to 12 words" bound brought back unclosed-focus loops;
    # removing the strategy scaffolding collapsed structured tasks to zero.
    assert "1 to 12 words" in instruction
    assert "Strategy" in instruction
    assert "at least one <focus> block" in instruction
    assert ANSWER_SPEC in instruction


def test_all_arms_share_the_answer_specification(family_case, filler):
    _, _, renderer = family_case
    for arm in ("da", "vanilla"):
        assert ANSWER_SPEC in renderer.render(arm, filler, "Who?").messages[-1]["content"]


def test_unknown_arm_raises(family_case, filler):
    _, _, renderer = family_case
    with pytest.raises(ValueError):
        renderer.render("da-nomask", filler, "Who?")


def test_fingerprint_changes_with_the_prompt(family_case, filler):
    _, _, renderer = family_case
    a = renderer.render_da(filler, "Who?")
    b = renderer.render_da(filler, "Who won?")
    assert a.fingerprint != b.fingerprint
