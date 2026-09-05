from collections.abc import Sequence
from string import Formatter


def validate_prompt_template(template: str) -> None:
    if not isinstance(template, str):
        raise TypeError("prompt_template must be a string")
    fields = [part for part in Formatter().parse(template) if part[1] is not None]
    valid = (
        len(fields) == 1
        and fields[0][1] == ""
        and fields[0][2] == ""
        and fields[0][3] is None
    )
    if not valid:
        raise ValueError(
            "prompt_template must contain exactly one plain positional '{}' field"
        )


def build_prompts(labels: Sequence[object], template: str) -> list[str]:
    validate_prompt_template(template)
    return [template.format(str(label).replace("_", " ")) for label in labels]
