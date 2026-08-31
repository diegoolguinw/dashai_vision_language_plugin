import pytest

from dashai_clip_model_package.prompts import build_prompts, validate_prompt_template


def test_build_prompts_normalizes_underscores():
    assert build_prompts(
        ["cat", "golden_retriever"], "a photo of a {}"
    ) == ["a photo of a cat", "a photo of a golden retriever"]


@pytest.mark.parametrize(
    "template",
    ["a photo", "{label}", "{} {}", "{!r}", "{:>10}"],
)
def test_validate_prompt_template_rejects_unsupported_fields(template):
    with pytest.raises(ValueError, match="exactly one plain"):
        validate_prompt_template(template)
