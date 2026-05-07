"""Parsing helpers for interview prompt outputs."""

import re


def parse_interview_label(output, allowable_entries=("yes", "no")):
    """
    Parse an interview answer label from an <answer> XML tag.

    Assumes output is model-generated text and allowable_entries contains the
    complete set of accepted labels. Returns the normalized label, or None when
    the output does not contain an accepted answer tag.
    """
    if output is None:
        return None

    if not allowable_entries:
        return None

    pattern = "|".join([re.escape(x) for x in allowable_entries])
    match = re.search(rf"<answer>\s*({pattern})\s*</answer>", output, re.IGNORECASE)
    if match is None:
        return None

    parsed = match.group(1).lower()
    normalized_allowables = tuple(x.lower() for x in allowable_entries)
    if parsed not in normalized_allowables:
        return None

    return parsed


def parse_minutes_from_activity_time(raw_value, xml_context=None):
    """
    Convert an activity time field into minutes.

    Assumes the activity-matching prompt returns one numeric duration with a
    unit of minutes, minute, mins, min, hours, hour, hrs, or hr. Bare numeric
    values are interpreted as minutes. Numeric ranges are converted to their
    midpoint. Negative values are invalid.
    """
    text = str(raw_value).strip().lower().replace("~", "")
    text = re.sub(r"^[<>]\s*", "", text)
    if text in {"", "not_specified", "not specified"}:
        print(
            "\nWARNING: Treating missing duration as 0 in activity-task XML:\n"
            f"{xml_context if xml_context is not None else raw_value}\n",
            flush=True,
        )
        return 0.0

    if re.search(r"(^|\s)-\s*[0-9]", text):
        raise ValueError(f"Negative duration values are invalid: {raw_value}")

    number_pattern = r"[0-9]+(?:\.[0-9]+)?"
    if re.fullmatch(number_pattern, text):
        return float(text)

    range_match = re.search(
        rf"({number_pattern})\s*[-–—]\s*({number_pattern})(?:\s*([a-z]+))?",
        text,
    )
    if range_match is not None:
        midpoint = (float(range_match.group(1)) + float(range_match.group(2))) / 2
        unit = range_match.group(3)
        if unit is None or unit in {"minute", "minutes", "min", "mins"}:
            return midpoint
        if unit in {"hour", "hours", "hr", "hrs"}:
            return midpoint * 60
        raise ValueError(f"Unsupported duration unit in activity time value: {raw_value}")

    match = re.search(rf"({number_pattern})\s*([a-z]+)", text)
    if match is None:
        raise ValueError(f"Could not parse duration from activity time value: {raw_value}")

    value = float(match.group(1))
    unit = match.group(2)
    if unit in {"minute", "minutes", "min", "mins"}:
        return value
    if unit in {"hour", "hours", "hr", "hrs"}:
        return value * 60

    raise ValueError(f"Unsupported duration unit in activity time value: {raw_value}")


def parse_activity_task_match_blocks(output):
    """
    Parse activity blocks from PROMPT_INTERVIEW_MATCH_ACTIVITIES_TO_TASKS output.

    Assumes the output contains one <activity> element per activity and the
    exact structured child tags specified in
    PROMPT_INTERVIEW_MATCH_ACTIVITIES_TO_TASKS. Returns one dictionary per
    activity with the activity name, whether the activity belongs to the task
    being scored, and minute-normalized durations.
    """
    match = re.search(
        r"<activity_task_matches>\s*(.*?)\s*</activity_task_matches>",
        str(output),
        re.DOTALL,
    )
    if match is None:
        raise ValueError("Missing <activity_task_matches> block")

    activity_blocks = re.findall(
        r"<activity>\s*(.*?)\s*</activity>",
        match.group(1),
        re.DOTALL,
    )
    if not activity_blocks:
        raise ValueError("No activity entries found in <activity_task_matches>")

    # print("--------------------------------")
    # print(output)
    # print("--------------------------------")

    parsed_blocks = []
    for block in activity_blocks:
        activity_description = get_required_activity_field(
            block,
            "activity_description",
        )
        belongs_raw = get_required_activity_field(
            block,
            "belongs_to_task_being_scored",
        ).lower()
        if belongs_raw not in {"yes", "no"}:
            raise ValueError(
                "belongs_to_task_being_scored must be yes or no: "
                f"{belongs_raw}"
            )
        belongs_to_task_being_scored = belongs_raw == "yes"

        reported_savings_text = get_required_activity_field(
            block,
            "reported_time_savings_minutes",
        )
        original_time_text = get_required_activity_field(
            block,
            "original_time_minutes",
        )
        if (
            str(reported_savings_text).strip().lower().replace("_", " ") == "not specified"
            or str(original_time_text).strip().lower().replace("_", " ") == "not specified"
        ):
            print(
                "\nWARNING: Treating activity with NOT_SPECIFIED duration as 0 in activity-task XML:\n"
                f"{output}\n",
                flush=True,
            )
            parsed_blocks.append(
                {
                    "activity_description": activity_description,
                    "belongs_to_task_being_scored": belongs_to_task_being_scored,
                    "reported_time_savings_minutes": 0.0,
                    "original_time_minutes": 0.0,
                }
            )
            continue

        parsed_blocks.append(
            {
                "activity_description": activity_description,
                "belongs_to_task_being_scored": belongs_to_task_being_scored,
                "reported_time_savings_minutes": parse_minutes_from_activity_time(
                    reported_savings_text,
                    xml_context=output,
                ),
                "original_time_minutes": parse_minutes_from_activity_time(
                    original_time_text,
                    xml_context=output,
                ),
            }
        )

    return parsed_blocks


def get_required_activity_field(activity_block, tag_name):
    """
    Read a required structured field from an activity block.

    Assumes activity_block may contain malformed free-text tags, but the
    structured scoring fields use exact opening and closing tags. Returns
    stripped text and raises when the field is missing or empty.
    """
    field = re.search(
        rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>",
        activity_block,
        re.DOTALL,
    )
    if field is None or field.group(1).strip() == "":
        raise ValueError(f"Activity block is missing required XML tag: <{tag_name}>")

    return field.group(1).strip()


def bucket_activity_task_match_score(output, category_task):
    """
    Compute the v4 span bucket from activity-level task matching output.

    Assumes each activity row includes belongs_to_task_being_scored. Sums only
    activities marked yes for inclusion in the task currently being scored,
    divides their total time savings by their total original time, and returns
    the span label.
    """
    if not category_task:
        raise ValueError("category_task must identify the task being scored")

    activity_blocks = parse_activity_task_match_blocks(output)
    included_blocks = [
        block
        for block in activity_blocks
        if block["belongs_to_task_being_scored"]
    ]

    numerator = sum(
        block["reported_time_savings_minutes"]
        for block in included_blocks
    )
    denominator = sum(
        block["original_time_minutes"]
        for block in included_blocks
    )
    if denominator == 0:
        print(
            "WARNING: Total original time spent on task activities is 0",
            flush=True,
        )
        return "0-24%"

    if denominator <= 0:
        raise ValueError("Total original time spent on task activities must be positive")

    ratio = numerator / denominator
    if ratio < 0:
        raise ValueError(f"Computed negative time-savings ratio: {ratio}")
    if ratio <= 0.24:
        return "0-24%"
    if ratio <= 0.49:
        return "25-49%"
    if ratio <= 0.74:
        return "50-74%"
    return "75-100%"
