import json

# Default prompt to break into subclaims.
BREAKDOWN_PROMPT = "Please breakdown the following input into a set of small, independent claims, and return the output as a jsonl, where each line is {subclaim:[CLAIM], gpt-score:[CONF]}.\n The confidence score [CONF] should represent your confidence in the claim, where a 1 is obvious facts and results like 'The earth is round' and '1+1=2'. A 0 is for claims that are very obscure or difficult for anyone to know, like the birthdays of non-notable people. The input is: "


def query_model(client, prompt, model, max_tokens=1000, temperature=0, n_samples=1):
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n_samples,
    )
    return completion.choices[0].message.content


def say_less(client, prompt, model, output, threshold):
    """
    say_less takes in the model output y, breaks it down into subclaims, and removes sub-claims up to the threshold value.
    The subclaims are scored by counting (using an LM) how many times they appear from 5 other sampled outputs. This is done
    in get_frequency_scores.
    """
    subclaims = get_subclaims(client, output, model)
    if subclaims is None:
        raise ValueError("Failed to extract subclaims")
    print(f"Extracted {len(subclaims)} subclaims.")
    print(subclaims[0])
    frequency_scores = get_frequency_scores(client, subclaims, prompt, 5, model)
    for i, subclaim in enumerate(subclaims):
        subclaim["frequency-score"] = frequency_scores[i]

    accepted_subclaims = [
        subclaim for subclaim in subclaims if subclaim["frequency-score"] > threshold
    ]
    merged_output = merge_subclaims(client, accepted_subclaims, model, prompt)

    return merged_output, (accepted_subclaims, subclaims)


def get_subclaims(
    client,
    output,
    model,
    breakdown_prompt=BREAKDOWN_PROMPT,
    max_tokens=1000,
    temperature=0,
):
    """
    Takes in an output text and breaks it down into a list of sub-claims.
    """
    # Break into sub-claims.
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant whose job is to break down your inputs into a set of small claims so that a human can easily check each one. Make sure that each claim is small and non-overlapping.",
        },
        {
            "role": "user",
            "content": breakdown_prompt + output,
        },
    ]
    completion = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    output = completion.choices[0].message.content
    print("Subclaims output:", output)
    print("extracting subclaims...")
    subclaims = extract_text_between_backticks(output)
    print(subclaims)

    # output = output.replace("```jsonl\n", "")
    # output = output.replace("\\", "\\\\")
    # subclaims = output.replace("```", "")

    # Parse as jsonl and normalize subclaims.
    try:
        subclaims = convert_text_to_json(subclaims)
        # subclaims = [json.loads(line) for line in subclaims.splitlines() if line]
        # subclaims = normalize_subclaims(subclaims)  # Normalize subclaims to handle lists
        return subclaims
    except Exception as ex:
        print(ex)
        print("Failed to parse as jsonl")
        # print(subclaims)
        return []


## String manipulation for extracting text between backticks.
def extract_text_between_backticks(text):
    """
    Extracts the text between triple backticks.
    """
    start = text.find("```") + 3  # Find the start of the text after the first ```
    end = text.rfind("```")  # Find the last occurrence of ```
    return text[start:end].strip() if start >= 3 and end > start else None


def get_frequency_scores(client, subclaims, prompt, n_samples, model):
    """
    Returns a vector of (frequency) scores corresponding to each entry of the subclaims list.
    """
    # Generate n_samples alternate outputs with temperature 1.0.
    messages = [{"role": "user", "content": prompt}]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.0,
        n=n_samples,
    )
    alternate_outputs = [choice.message.content for choice in completion.choices]
    claim_string = "\n".join(
        [str(i) + ": " + fact["subclaim"] for i, fact in enumerate(subclaims)]
    )

    # Count the number of times the alternate outputs support the sub-claims (using LM).
    # TODO: should this really be -1, 0, 1? Before it was 0, 1.
    final_scores = [0.0] * len(subclaims)
    for output in alternate_outputs:
        counting_prompt = (
            'You will get a list of claims and piece of text. For each claim, score whether the text supports, contradicts, or is unrelated to the claim. Directly return a jsonl, where each line is {"id":[CLAIM_ID], "score":[SCORE]}. Directly return the jsonl with no explanation or other formatting. For the [SCORE], return 1 for supports, -1 for contradicts, and 0 for unrelated. The claims are:\n'
            + claim_string
            + "\n\nThe text is:\n"
            + output
        )
        output = query_model(
            client, counting_prompt, model, max_tokens=1000, temperature=0
        )
        output = output.replace("```jsonl\n", "")
        output = output.replace("```", "")
        try:
            for i, line in enumerate(output.splitlines()):
                scores = json.loads(line)
                idx = int(scores["id"])
                final_scores[idx] += float(scores["score"])
        except Exception as ex:
            print(ex)
            print("Failed to parse as jsonl")
            print(output)

    return final_scores


def default_merge_prompt(subclaims, prompt):
    claim_string = "\n".join(
        [str(i) + ": " + subclaim["subclaim"] for i, subclaim in enumerate(subclaims)]
    )
    return f"You will get an instruction and a set of facts that are true. Construct an answer using ONLY the facts provided, and try to use all facts as long as its possible. If no facts are given, reply to the instruction incorporating the fact that you dont know enough to fully respond. \n\nThe facts:\n{claim_string}\n\nThe instruction:\n{prompt}"


def merge_subclaims(
    client, subclaims, model, prompt, create_merge_prompt=default_merge_prompt
):
    """
    Takes in a list of sub-claims like [{'subclaim': 'Percy Liang is a computer scientist.', 'score': 5.0}, ...] and produces a merged output.
    """
    prompt = create_merge_prompt(subclaims, prompt)
    output = (
        query_model(client, prompt, model, max_tokens=1000, temperature=0)
        if subclaims
        else "Abstain."
    )
    return output


def convert_text_to_json(text):
    """
    Converts text containing subclaims and gpt-scores into a JSON structure.
    """
    try:
        # Split the text into lines and parse each line as JSON
        subclaims = [json.loads(line) for line in text.splitlines() if line]
        
        # Normalize subclaims to ensure single values for subclaim and gpt-score
        normalized_subclaims = []
        for sc in subclaims:
            new_sc = {}
            # Extract string from list if needed
            if isinstance(sc.get("subclaim"), list):
                new_sc["subclaim"] = sc["subclaim"][0]
            else:
                new_sc["subclaim"] = sc.get("subclaim")
            # Extract float from list if needed
            if isinstance(sc.get("gpt-score"), list):
                new_sc["gpt-score"] = sc["gpt-score"][0]
            else:
                new_sc["gpt-score"] = sc.get("gpt-score")
            normalized_subclaims.append(new_sc)
        
        return normalized_subclaims
    except Exception as ex:
        print(f"Failed to convert text to JSON: {ex}")
        return []


# def normalize_subclaims(subclaims):
#     """
#     Converts subclaims with list values to single values.
#     """
#     normalized = []
#     for sc in subclaims:
#         new_sc = {}
#         # Extract string from list if needed
#         if isinstance(sc.get("subclaim"), list):
#             new_sc["subclaim"] = sc["subclaim"][0]
#         else:
#             new_sc["subclaim"] = sc.get("subclaim")
#         # Extract float from list if needed
#         if isinstance(sc.get("gpt-score"), list):
#             new_sc["gpt-score"] = sc["gpt-score"][0]
#         else:
#             new_sc["gpt-score"] = sc.get("gpt-score")
#         normalized.append(new_sc)
#     return normalized
