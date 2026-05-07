HW_PROMPT = """
You are provided a series of requests that a user made to an AI assistant (the assistant's responses are omitted).

Determine whether the user requests contains an obvious reference to a homework assignment.

First, read the user requests carefully. Then, explaining your reasoning in <thinking></thinking> tags, see if there are any *obvious* references to a homework assignment that the user wants completed. If there are such obvious references, then extract one of them in a <reference></reference> tag (just write <reference></reference> with nothing in the tags if there are no such references). Finally, re-evaluate the content in the <reference></reference> tags. If the extracted reference *really* contains an obvious reference to a homework assignment that the user wants the assistant to complete, then answer Yes (again, use <thinking></thinking> tags to explain you reasoning), otherwise answer No (use <answer></answer> tags).

Keep your reasoning (the content in the <thinking> tags) to at most three sentences each.

# Example
## Input
User requests:
```
Human: The data found in the data file employment .xls represent levels of employment
by industry in the UK between 1999 Quarter 1 to 2019 Quarter 4.
Your task is to write a report based on:
a) (i) Graphical (tables, charts, figures etc.) and descriptive statistical (mean,
median, mode, standard deviation etc.) presentation (using Excel).
(ii) Evaluation (write in your own words), of employment number of each
industry.
[35 marks]
...
```

## Output
<thinking>The user's request clearly outlines a graded task with marks and specific report components (graphical presentation, descriptive statistics, evaluation). This format (“[35 marks]”) indicates a homework or coursework assignment.</thinking>
<reference>Your task is to write a report based on: a) (i) Graphical (tables, charts, figures etc.) and descriptive statistical (mean, median, mode, standard deviation etc.) presentation (using Excel). (ii) Evaluation (write in your own words), of employment number of each industry. [35 marks]</reference>
<thinking>The extracted reference obviously describes a marked assignment the user wants completed.</thinking>
<answer>Yes</answer>

# Example
## Input
User requests:
```
Human: Grade this assignment submitted by a student with the title ‘How can learning theories be applied in the VET sector?’. The assignment is required to address these questions:
What are learning theories and are they useful in VET?
Are there any learning theories that are particularly suited to VET in general and
In your own teaching area specifically (you need to briefly explain your teaching area and context)?
How can the practical application of learning theories make teaching more effective?
Your essay should include an introduction and conclusion and at least six references, of which at least four should be peer reviewed.
```

## Output
<thinking>The user is asking the assistant to grade an assignment submitted by a student, not to write or complete the assignment itself. There is no obvious reference to a homework assignment the user wants the assistant to complete.</thinking>
<reference></reference>
<thinking>No obvious reference to a homework assignment the user wants the assistant to complete was found.</thinking>
<answer>No</answer>

# Example
## Input
User requests:
```
Human:  23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98%% on room air. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus. Which of the following is the best treatment for this patient?

A. Ampicillin 
B. Ceftriaxone
C. Doxycycline
D. Nitrofurantoin

Human: What is the correct choice ?
```

## Output
<thinking>The user has provided a clinical scenario followed by multiple-choice answers and explicitly asks for the correct choice. This format clearly resembles a homework or exam question, suggesting it is an assignment the user wants completed.</thinking>
<reference>23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination... Which of the following is the best treatment for this patient?</reference>
<thinking>The extracted reference explicitly presents a structured multiple-choice clinical question that strongly suggests it is a homework or exam-style question the user wants completed.</thinking>
<answer>Yes</answer>

# Target
## Input
User requests:
```
{instance_text_human_only}
```

## Output
""".strip()
