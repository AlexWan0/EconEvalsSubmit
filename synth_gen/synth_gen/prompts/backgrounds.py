from dataclasses import dataclass, field
import re
import xml.etree.ElementTree as ET
from typing import Optional, Callable

from lm_prompts import StringPrompt, PromptInputT, PromptOutputT, cast_turns, Turn


@dataclass
class BackgroundGenArgs:
    category_occ: str
    task_detailed: str
    num_respondents: int

@dataclass
class GeneratedBackground:
    respondent_id: int
    job_title: str
    years_experience: int
    company_description: str
    summary: str

@dataclass
class GeneratedBackgroundBatch:
    backgrounds: list[GeneratedBackground]

BACKGROUNDS_PROMPT = """
# Background
An occupation researcher is interested in what workers in the occupation of `{category_occ}` do day-to-day. Specifically, they want to know how workers perform this task: `{task_detailed}`.

To answer this, the researcher surveys {num_respondents} random workers in the occupation of `{category_occ}` and asks each of them: "Think back to the last time you performed the task of `{task_detailed}`. First provide your professional background. Then provide a high-level summary of what specifically you were doing, including any relevant context."

The workers come from varied companies, seniority levels, and contexts. Responses must reflect that real-world variety.

# Instructions
First, output the backgrounds for all {num_respondents} workers. Then, output the summaries for all {num_respondents} workers.

# Output format
Output a single XML document in a codeblock labeled xml.

Schema:
<responses>
  <backgrounds>
    <respondent id="1">
      <job_title>...</job_title>
      <years_experience>...</years_experience>
      <company_description>...</company_description>
    </respondent>
    ...
  </backgrounds>
  <summaries>
    <respondent id="1">
      <summary>...</summary>
    </respondent>
    ...
  </summaries>
</responses>

Rules:
- Include exactly {num_respondents} respondent elements in <backgrounds> and exactly {num_respondents} in <summaries>.
- Respondent id must be an integer from 1 to {num_respondents} (one for each participant).
- Job title must be less than five words.
- Years of experience must be a single integer.
- Company description must be a single short sentence (20 words at most).
- Summary must be a single sentence.
- Use standard XML escaping for &, <, and >.

Example:
```xml
<responses>
  <backgrounds>
    <respondent id="1">
      <job_title>Senior welder</job_title>
      <years_experience>12</years_experience>
      <company_description>Mid-sized fabrication shop serving regional construction firms.</company_description>
    </respondent>
  </backgrounds>
  <summaries>
    <respondent id="1">
      <summary>I repaired structural joints on a steel frame for a warehouse expansion.</summary>
    </respondent>
  </summaries>
</responses>
```
""".strip()


def _extract_xml_block(output_str: str) -> str:
    xml_blocks = re.findall(r"```xml(.*?)```", output_str, re.DOTALL | re.IGNORECASE)
    if xml_blocks:
        return xml_blocks[0].strip()

    match = re.search(r"(<responses[\s\S]*?</responses>)", output_str, re.IGNORECASE)
    if match:
        return match.group(1)

    raise ValueError("No XML block found")


def _sanitize_xml(xml_str: str) -> str:
    xml_str = xml_str.replace("&aposs;", "&apos;").replace("&aposs", "&apos;")
    xml_str = re.sub(
        r"&(?!(?:amp|lt|gt|apos|quot|#\d+|#x[0-9a-fA-F]+);)",
        "&amp;",
        xml_str,
    )
    return xml_str


def _parse_single_output(output_str: str) -> list[GeneratedBackground]:
    """
    Parses a single output string into a list of GeneratedBackground objects. loop_idx is used to offset respondent_id for uniqueness across multiple outputs.
    """
    xml_str = _sanitize_xml(_extract_xml_block(output_str))
    root = ET.fromstring(xml_str)

    backgrounds: dict[int, dict[str, object]] = {}
    summaries: dict[int, str] = {}

    for resp in root.findall(".//backgrounds/respondent"):
        resp_id = resp.get("id") or resp.findtext("id")
        job_title = resp.findtext("job_title")
        years_experience = resp.findtext("years_experience")
        company_description = resp.findtext("company_description")

        if (
            resp_id is None
            or job_title is None
            or years_experience is None
            or company_description is None
        ):
            raise ValueError("Missing required background fields")

        rid = int(str(resp_id).strip())
        backgrounds[rid] = {
            "job_title": str(job_title).strip(),
            "years_experience": int(str(years_experience).strip()),
            "company_description": str(company_description).strip(),
        }

    for resp in root.findall(".//summaries/respondent"):
        resp_id = resp.get("id") or resp.findtext("id")
        summary = resp.findtext("summary")

        if resp_id is None or summary is None:
            raise ValueError("Missing required summary fields")

        rid = int(str(resp_id).strip())
        summaries[rid] = str(summary).strip()

    results: list[GeneratedBackground] = []
    for rid, bg in backgrounds.items():
        if rid not in summaries:
            raise ValueError(f"Missing summary for respondent {rid}")
        results.append(
            GeneratedBackground(
                respondent_id=rid,
                job_title=str(bg["job_title"]),
                years_experience=int(str(bg["years_experience"])),
                company_description=str(bg["company_description"]),
                summary=summaries[rid],
            )
        )

    return results

@dataclass
class BackgroundsPrompt(StringPrompt[BackgroundGenArgs, GeneratedBackgroundBatch]):
    input_type: type[BackgroundGenArgs] = BackgroundGenArgs
    output_type: type[GeneratedBackgroundBatch] = GeneratedBackgroundBatch
    template: str = BACKGROUNDS_PROMPT

    def parse_output_str(self, output_str: str) -> GeneratedBackgroundBatch:
        return GeneratedBackgroundBatch(backgrounds=_parse_single_output(output_str))
    