"""Prompt and retrieval text construction from explicitly allowed fields."""

import re

DESCRIPTION_PROMPT = (
    "Describe the visible image in one or two short factual sentences. "
    "Mention objects, actions, spatial relationships, and readable text when useful. "
    "Do not speculate or answer an unseen question."
)


def question_block(query):
    lines = ["Question: " + query.question]
    if query.hint:
        lines.append("Hint: " + query.hint)
    lines.append(
        "Choices:\n"
        + "\n".join("{}. {}".format(k, query.choices[k]) for k in sorted(query.choices))
    )
    return "\n".join(lines)


def draft_prompt(query, description):
    return (
        "Answer the question using the supplied image and hint.\n"
        "Model-generated description of this image (may contain errors): "
        + description
        + "\n\n"
        + question_block(query)
        + "\nAnswer with only one of the supplied option letters."
    )


def final_prompt(query, description, examples):
    lines = [
        "Answer the question using the supplied original image and hint.",
        "The following retrieved examples may be irrelevant. Their descriptions are model-generated "
        "and may contain errors. Their known answers concern only those examples. "
        "Use them only when helpful; do not copy an example's option letter.",
    ]
    for i, example in enumerate(examples, 1):
        lines.append("\nRetrieved example {}:".format(i))
        lines.append("Model-generated image description: " + example["description"])
        lines.append("Question: " + example["question"])
        if example.get("hint"):
            lines.append("Hint: " + example["hint"])
        # Answer text stays attached to its own reference question, not query options.
        lines.append("Known answer: " + example["answer_text"])
    lines.extend(
        [
            "\nNow answer the current query using its original image.",
            "Model-generated description of the current image (may contain errors): " + description,
            question_block(query),
            "Answer with only one of the supplied option letters.",
        ]
    )
    return "\n".join(lines)


def fit_examples(query, description, examples, fits):
    retained = list(examples)
    while True:
        prompt = final_prompt(query, description, retained)
        if fits(prompt, 1):
            return prompt, retained
        if not retained:
            raise ValueError(
                "query image, question, hint, and choices exceed LLaVA's context window"
            )
        retained.pop()  # Remove the lowest score first; preserve all query fields.


def parse_draft(raw, choices):
    match = re.fullmatch(r"\s*([A-Za-z])\s*\.?\s*", raw)
    letter = match.group(1).upper() if match else None
    return letter if letter in choices else None


def retrieval_text(question, hint, description, answer_text):
    lines = ["Question: " + question]
    if hint:
        lines.append("Hint: " + hint)
    lines.append("Image description: " + description)
    if answer_text:
        lines.append("Answer: " + answer_text)
    return "\n".join(lines)


def candidate_retrieval_text(question, hint, description, answer_texts):
    """Change only the answer portion; singleton queries match the baseline."""
    if not answer_texts or any(not text for text in answer_texts):
        raise ValueError("candidate retrieval requires nonempty answer texts")
    if len(answer_texts) == 1:
        return retrieval_text(question, hint, description, answer_texts[0])
    return (
        retrieval_text(question, hint, description, None)
        + "\nCandidate answers:\n"
        + "\n".join("- " + text for text in answer_texts)
    )
