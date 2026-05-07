import argparse

import pandas as pd

from exposure_lib.utils import LLMArgs, run_prompt_column, parse_xml_tags, save_pickle_gzip


FEWSHOT_BACKGROUNDS = [
    {
        "category_occ": "General and Operations Managers",
        "task_detailed": "Direct and coordinate activities of businesses or departments concerned with the production, pricing, sales, or distribution of products. However, this does not include overseeing environmental programs, performing personnel functions, managing financial or budget activities, or directing non-merchandising departments. The focus is on overall business or departmental coordination rather than specific operational or administrative tasks.",
        "query": "You are an operations manager assistant for a national logistics company handling freight and delivery. Develop a complete, executable operations coordination plan to optimize distribution across the regional network for the upcoming quarter. The plan must be self-contained and actionable (no external data required). Produce the following deliverables in clear sections:\n\n1) Executive summary (3-5 bullets) with the objective, expected benefits, and estimated percentage reductions in lead time and cost.\n\n2) Step-by-step operational playbook (ordered tasks with responsible role, required inputs, outputs, estimated time per step, and dependencies) covering:\n   - Demand consolidation and routing batch windows\n   - Dynamic load balancing across hubs\n   - Fleet utilization optimization (including mix of truck sizes)\n   - Last-mile scheduling and slotting adjustments\n   - Contingency rerouting and exception handling\n\n3) Resource allocation table (as bullets) showing required headcount, vehicle types, and peak vs off-peak allocation for each regional hub.\n\n4) KPIs and target thresholds (primary and secondary) with definitions and suggested measurement frequency (e.g., on-time %, dwell time, load factor, cost per mile).\n\n5) 12-week implementation timeline with milestones, checkpoints, and a 2-week pilot at one hub (include success criteria for the pilot and go/no-go decision rules).\n\n6) Estimated costs and projected savings (ranges: low/likely/high) with assumptions called out.\n\n7) Risk register with at least 6 risks, likelihood and impact ratings (Low/Med/High), and mitigation actions.\n\n8) Communication and change management plan for operations staff and drivers (roles, cadence, template messages).\n\n9) A short, ready-to-send email (3-5 sentences) to senior leadership summarizing the plan and recommending approval for the pilot, including requested resources and timeline.\n\nConstraints and requirements:\n- Keep everything concise and action-oriented (use bullets and numbered lists).\n- All recommendations should be implementable by standard operations teams (no specialized software or external data).\n- Make reasonable, explicit assumptions where needed (e.g., average daily volume per hub = X shipments, average truck capacity = Y pallets). Use placeholder numeric assumptions only where unavoidable, and label them as assumptions.\n- Do not reference external files, attachments, or links. The output must be fully self-contained.",
        "job_title": "Operations Manager",
        "years_experience": 8,
        "company_description": "National logistics company specializing in freight and delivery services.",
    },
    {
        "category_occ": "Arbitrators, Mediators, and Conciliators",
        "task_detailed": "Specialize in the negotiation and resolution of environmental conflicts involving issues such as natural resource allocation or regional development planning. However, this does not include conducting formal hearings, issuing rulings, or preparing legal decisions related to these conflicts. The focus is on facilitating dialogue and agreement rather than adjudicating disputes.",
        "query": "You are a neutral conflict-resolution facilitator for a regional natural resource planning agency. Prepare a complete facilitation package for a 3-hour stakeholder workshop to negotiate a multi-party agreement on seasonal water allocations for irrigation, instream flows for fish habitat, and recreational water use on River X. The package should be ready to use by a facilitator with general mediation experience and no prior knowledge of this river conflict. Provide the following items, each clearly labeled and organized:\n\n1) A one-paragraph neutral summary of the conflict (causes, key issues, and stakes) written so stakeholders such as farmers, fisheries managers, tribal representatives, recreation groups, and local government understand its perspectives without taking offense.\n\n2) A roster of likely stakeholder roles (title and 1-2 sentence description of interests and constraints) for the workshop, including at least: irrigated agriculture representative(s), fisheries/ecosystem manager, tribal water rights/sovereignty representative, municipal water manager, recreation/tourism representative, and a regional policy official.\n\n3) A pre-workshop one-page brief to send to participants (neutral tone) explaining purpose, agenda, confidentiality rules, and what to prepare (no data files requested—only simple numbers or priorities they can provide verbally).\n\n4) A detailed 3-hour agenda with time allocations, goals for each segment, seating and breakout arrangements, and scripts for the facilitator (intro, ground rules, framing, transitions, and closing).\n\n5) A structured in-session negotiation design: include 3 concrete multiple-option packages (Option A, B, C) for seasonal water allocations that balance irrigation, instream flow, and recreation. For each option provide expected seasonal flow numbers in relative terms (High/Medium/Low) for each use, plain-language tradeoffs, and likely winners/losers.\n\n6) A simple decision rule template (voting and consensus thresholds) and a default tie-breaking mechanism acceptable to all parties that maintains perceived fairness.\n\n7) A short interest-based bargaining script with suggested wording and prompts to move from positions to interests, plus a menu of at least 8 creative concessions/tradeoffs (e.g., temporal shifts, storage timing, joint monitoring, compensation mechanisms, adaptive triggers).\n\n8) A draft memorandum of understanding (MOU) one-page template that captures agreement points, monitoring and compliance language, dispute-resolution steps, and a review/adaptive management timeline. Use plain language and include placeholders only for dates and signatory names (no other placeholders).\n\n9) A six-step rapid risk checklist to identify common implementation risks (e.g., drought, enforcement, data gaps) and recommended mitigation actions for each.\n\n10) A ten-item post-workshop follow-up email template to participants, including timeline for next steps, assigned responsibilities and deadlines, and how the MOU will be finalized.\n\nConstraints and tone:\n- Maintain a neutral, facilitative, non-judgmental tone.\n- Do not include legal adjudication language; this is mediation/facilitation.\n- Use plain English suitable for a mixed-education audience.\n- Materials should be printable on standard paper and usable without digital tools.\n- Do not request external data, attachments, or confidential documents.\n\nDeliver the full package in a single response, with clear section headings matching the numbered items above so the facilitator can copy-paste sections into separate handouts.",
        "job_title": "Conflict Resolution Specialist",
        "years_experience": 15,
        "company_description": "Government agency managing regional natural resource planning and policy.",
    },
    {
        "category_occ": "Environmental Science and Protection Technicians, Including Health",
        "task_detailed": "Investigate hazardous conditions or spills or outbreaks of disease or food poisoning, collecting samples for analysis. However, this does not include analyzing test data, preparing reports, or making regulatory recommendations based on findings. The task is limited to the initial investigation and sample collection only.",
        "query": "You are an expert Health & Safety Specialist and environmental field sampler with 10+ years’ experience supporting industrial clients responding to hazardous material incidents (chemical spills, suspect contamination, acute illness/outbreaks tied to facilities). Provide a concise, practical, self-contained field investigation and sample collection plan suitable for immediate initial site response. Do not include analysis, interpretation of lab results, regulatory recommendations, or report writing.\n\nConstraints and context:\n- Site: manufacturing facility with potential indoor liquid chemical spill and outdoor runoff to an adjacent storm drain. Unknown chemical identity.\n- Personnel: 2-person field response team, trained in HAZWOPER awareness and PPE selection but not full hazardous materials technicians.\n- Time: initial response within 4 hours of notification.\n- Available field supplies: air-purifying respirators (half-mask with P100), nitrile gloves (various sizes), chemical-resistant outer gloves, splash suits, disposable coveralls, safety glasses, hard hats, first-aid kit, multi-gas meter (LEL/O2/CO/H2S), photo documentation (smartphone), field pH strips, portable chlorine test strips, stainless-steel and glass sample bottles (pre-cleaned), zip seal bags, coolers with ice packs, chain-of-custody forms, permanent marker, waterproof labels, plastic scoops, syringes (needleless), disposable pipettes, sample coolers, decontamination wipes, hand sanitizer.\n- Lab: accredited environmental lab available same-day for priority samples; chain-of-custody required; expedited turnaround possible if prioritized.\n- Safety priority: protect responders and public; avoid creating secondary contamination.\n\nDeliverables (in order of priority):\n1) Brief pre-entry checklist (5–8 items) to verify before any personnel enter the site.\n2) Stepwise initial site assessment checklist (10–12 items) for documenting conditions and deciding safe sampling locations and what to sample. Items should be actionable and concise (one sentence each).\n3) Specific sampling plan: list which samples to collect (minimum set) for an unknown liquid spill affecting indoor floor and outdoor runoff/drain, including sample type (liquid, sediment, wipe), container type, approximate volume or area, number of replicates, and preservation/cooling instructions suitable for typical organic and inorganic contaminants. Prioritize samples that the field team can collect with the listed supplies.\n4) Field chain-of-custody checklist with required fields and custody transfer steps (concise).\n5) Simple on-site decontamination checklist (5–7 items) for responders and equipment after sampling.\n6) Short \"when to escalate\" decision criteria (3–5 triggers) that require contacting hazardous materials technicians, emergency responders, or stopping work.\n\nFormatting:\n- Use numbered lists and short bullet points.\n- Each item must be self-contained and operationally specific but no long paragraphs.\n- Do not include laboratory analytical methods, interpretation guidance, or regulatory citations—only what to collect and how to handle samples and safety actions during initial response.\n\nNow generate the requested deliverables, concise and ready to print as a one-page field guide for the two-person team.",
        "job_title": "Health Safety Specialist",
        "years_experience": 5,
        "company_description": "Private environmental consulting firm serving industrial clients nationwide.",
    },
]


PROMPT = """
# Background
You will be given a ChatGPT query that a worker has asked. You will also be given the worker's occupation and task the worker was trying to use ChatGPT to assist with. Your task is to determine the exact job title, years of experience, and employer description of the worker who asked the question.

Your output must be in XML format, surrounded in triple backticks, and must only contain the following fields: job_title, years_experience, and company_description.

# Examples
{fewshot_str}

# Instructions
Use the worker's occupation/task along with the ChatGPT query used to assist with the task to determine the worker's job title, years of experience, and employer description. The job title must fall under the occupation provided. Your output must be in XML format, surrounded in triple backticks, and must only contain the following fields: job_title, years_experience, and company_description.

There's no one right answer for this! Use your best judgement to determine the most likely job title, years of experience, and employer description for the worker based on the occupation, task, and ChatGPT query provided.

## Input
Occupation: {category_occ}
Task to be assisted with: {task_detailed}
ChatGPT query:
<conversation>
<previous_turns>
{previous_turns}
</previous_turns>
<query>
{query}
</query>
</conversation>

## Output format
```xml
<years_experience>Years of Experience</years_experience>
<job_title>Job Title</job_title>
<company_description>Company Description</company_description>
```
""".strip()

EXAMPLE_STR_FMT = """
## Example
### Input
Occupation: {category_occ}
Task to be assisted with: {task_detailed}
ChatGPT query:
<conversation>
<previous_turns>
</previous_turns>
<query>
{query}
</query>
</conversation>

### Output
<years_experience>{years_experience}</years_experience>
<job_title>{job_title}</job_title>
<company_description>{company_description}</company_description>
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieved-samples-path",
        default="data/retrieved_samples.pkl.zst",
        help="Path to input retrieved samples pickle.",
    )
    parser.add_argument(
        "--tasks-path",
        default="../synth_gen/data/detailed_tasks_30_2.csv",
        help="Path to task lookup CSV containing DWA titles.",
    )
    parser.add_argument(
        "--output-path",
        default="data/retrieved_samples_with_backgrounds.pkl.gz",
        help="Path for the output pickle with inferred background fields.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ret_df = pd.read_pickle(args.retrieved_samples_path)
    tasks_dwas_df = pd.read_csv(args.tasks_path)

    ret_df = ret_df.merge(
        tasks_dwas_df[["category_occ", "category_task", "DWA Title", "task_detailed"]],
        left_on=["category_occ", "category_task", "dwa"],
        right_on=["category_occ", "category_task", "DWA Title"],
        how="left",
        validate="m:1",
    ).drop(columns=["DWA Title"])

    fewshot_str = "\n\n".join(
        EXAMPLE_STR_FMT.format(**background) for background in FEWSHOT_BACKGROUNDS
    )

    ret_df["query"] = ret_df["convo_subset"].apply(lambda convo: convo[-1]["content"])
    ret_df["previous_turns"] = ret_df["convo_subset"].apply(
        lambda convo: "\n".join(
            f"<turn role='{turn['role']}'>{turn['content']}</turn>" for turn in convo[:-1]
        )
    )
    ret_df["fewshot_str"] = fewshot_str
    ret_df["_model_input"] = ret_df.apply(lambda row: PROMPT.format(**row), axis=1)

    ret_df = run_prompt_column(
        ret_df,
        prompt_col="_model_input",
        llm_args=LLMArgs(
            model_name="openai/gpt-5-mini@reasoning_effort=low",
            temperature=1.0,
            max_tokens=4096,
            num_workers=256,
        ),
        output_col="inferred_background",
        parser=lambda x: parse_xml_tags(
            x,
            tags=["job_title", "years_experience", "company_description"],
            require_code_fence=True,
        ) if x is not None else None,
    )

    ret_df = pd.concat(
        [
            ret_df.drop(columns=["inferred_background"]).reset_index(drop=True),
            ret_df["inferred_background"].apply(pd.Series),
        ],
        axis=1,
    )

    save_pickle_gzip({"verified_df": ret_df}, args.output_path)


if __name__ == "__main__":
    main()
