from dataclasses import dataclass, field
import html
import re

from .query_references import _extract_tag

from lm_prompts import StringPrompt, PromptInputT, PromptOutputT, cast_turns, Turn


@dataclass
class RefGenArgs:
    query: str
    attachments_replaced_tags: str

@dataclass
class GeneratedRef:
    references: list[str]


GENERATE_REFERENCES_FROM_SPECS_PROMPT = """
# Background
You will be given:
1) A user query sent to an LM
2) A list of reference specifications used by that query

Your task is to generate the full text for each reference so that the query can be executed without any external files. You MUST generate the entire text of each reference based on the provided specifications. Do NOT generate a "best-effort" version that omits crucial details or require the user to refer to some other document. You MUST generate the full text of each reference.

# Query
```text
{query}
```

# Reference specifications
```xml
{attachments_replaced_tags}
```

# Instructions
- Generate one full reference for each <reference> in the input specs. Your output MUST be plain-text.

You MUST make sure that the following requirements of your generation are met:
- If you need to generate multiple references, they MUST be consistent with each other (e.g., they MUST not contain contradictory information).
- The references MUST be consistent with the query (the query is a task based on the references).
- The references MUST be consistent with their corresponding specifications.

# Output formatting
Your output MUST be formatted as follows, where each reference is wrapped in <referenceOutput> tags (including the triple backticks).
```
<referenceOutput>
[the full text of the first reference]
</referenceOutput>
<referenceOutput>
[the full text of the second reference]
</referenceOutput>
[...]
```
""".strip()


ATTACHMENT_BLOCK_RE = re.compile(r"<attachment>(.*?)</attachment>", re.DOTALL | re.IGNORECASE)


def attachments_to_reference_tags(attachments_xml: str | None) -> str | None:
    if attachments_xml is None:
        return None

    attachment_blocks = ATTACHMENT_BLOCK_RE.findall(attachments_xml)
    if not attachment_blocks:
        return None

    refs: list[str] = []
    for idx, block in enumerate(attachment_blocks, start=1):
        filename = _extract_tag(block, "filename")
        filesize_kb = _extract_tag(block, "filesizeKb")
        length_description = _extract_tag(block, "lengthDescription")
        description = _extract_tag(block, "description")

        name = filename if filename else f"attachment_{idx}"

        ref_lines = [
            f'  <reference id="{idx}">',
            f"    <name>{html.escape(name, quote=False)}</name>",
        ]

        if filesize_kb:
            ref_lines.append(
                f"    <filesizeKb>{html.escape(filesize_kb, quote=False)}</filesizeKb>"
            )
        if length_description:
            ref_lines.append(
                "    <lengthDescription>"
                f"{html.escape(length_description, quote=False)}"
                "</lengthDescription>"
            )
        if description:
            ref_lines.append(
                f"    <description>{html.escape(description, quote=False)}</description>"
            )

        ref_lines.append("  </reference>")
        refs.append("\n".join(ref_lines))

    return "<reference_specs>\n" + "\n".join(refs) + "\n</reference_specs>"


REFERENCE_OUTPUT_RE = re.compile(
    r"<referenceOutput>(.*?)</referenceOutput>",
    re.DOTALL | re.IGNORECASE,
)


def parse_reference_generation_output(
    output_str: str,
) -> list[str]:
    if output_str is None:
        raise ValueError("Output string is None")

    matches = REFERENCE_OUTPUT_RE.findall(output_str)
    if not matches:
        raise ValueError(f"Failed to parse reference generation output: no <referenceOutput> blocks found in output: {output_str}")

    parsed = [str(x).strip() for x in matches if str(x).strip()]
    if not parsed:
        raise ValueError(f"Failed to parse reference generation output: all <referenceOutput> blocks are empty after stripping whitespace in output: {output_str}")

    return parsed

@dataclass
class ReferencesPrompt(StringPrompt[RefGenArgs, GeneratedRef]):
    input_type: type[RefGenArgs] = RefGenArgs
    output_type: type[GeneratedRef] = GeneratedRef
    template: str = GENERATE_REFERENCES_FROM_SPECS_PROMPT

    def parse_output_str(self, output_str: str) -> GeneratedRef:
        refs = parse_reference_generation_output(output_str)

        return GeneratedRef(references=refs)
