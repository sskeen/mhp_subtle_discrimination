"""
qualitative.py
Qualitative deductive coding functions for CHALET-style LLM-human annotation.

Simone J. Skeen x Claude Code (02-04-2026)
"""

import json
import os
import time

import openai
import pandas as pd
import requests
import yaml


# ---------------------------------------------------------------------------
# OpenAI client helper
# ---------------------------------------------------------------------------

def _get_openai_client(client = None):
    """Return provided client or create one from OPENAI_API_KEY env var."""
    if client is not None:
        return client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. "
            "Call load_dotenv() before importing this module, "
            "or pass an openai.OpenAI client explicitly."
        )
    return openai.OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# YAML config / prompt builders
# ---------------------------------------------------------------------------

def load_annotation_config(yaml_path = None):
    """
    Load annotation schema configuration from YAML file.

    Parameters
    ----------
    yaml_path : str, optional
        Path to annotation_schema.yaml. Defaults to annotation_schema.yaml in the same
        directory as this module.

    Returns
    -------
    dict
        Parsed YAML configuration with 'role' and 'codes' keys.
    """
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "annotation_schema.yaml")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def build_prompt_llama(config, alias):
    """
    Build a Llama-format prompt for a given code alias.

    The Llama instruction template requests JSON output with keys
    '{alias}_llm' and '{alias}_expl'.

    Parameters
    ----------
    config : dict
        The loaded annotation schema config (from load_annotation_config).
    alias : str
        The code alias (e.g., 'refl').

    Returns
    -------
    str
        The assembled prompt string with {text} placeholder.
    """
    role = config["role"]
    code = config["codes"][alias]
    name = code["name"]
    definition = code.get("definition_llama", code["definition"])
    clarification = code.get("clarification", "")
    examples = code.get("examples", "")

    definition_block = f'\nDefinition of "{name}": {definition}\n'

    instruction = f"""
You will be provided with a piece of text. For each piece of text:
- If it meets the definition of "{name}," output {alias}_llm as "1".
- Otherwise, output {alias}_llm as "0".
- Also provide a short explanation in exactly two sentences, stored in {alias}_expl.

Please respond in valid JSON with keys "{alias}_llm" and "{alias}_expl" only.

Text:
{{text}}
"""

    return f"{role}{definition_block}{instruction}{clarification}{examples}"


def build_prompt_gpt(config, alias):
    """
    Build a GPT-format prompt for a given code alias.

    The GPT instruction template requests text output with
    '{alias}_1', '{alias}_rationale:', and '{alias}_explanation:'
    structure.

    Parameters
    ----------
    config : dict
        The loaded codes config (from load_annotation_config).
    alias : str
        The code alias (e.g., 'afrm').

    Returns
    -------
    str
        The assembled prompt string.
    """
    role = config["role"]
    code = config["codes"][alias]
    name = code["name"]
    definition = code["definition"]
    clarification = code.get("clarification", "")
    examples = code.get("examples", "")

    definition_block = f'\nDefinition of "{name}": {definition}\n'

    instruction = f"""
You will be provided with a piece of text. For each piece of text:
- If it meets the definition of "{name}," respond with "{alias}_1"
- Otherwise, respond with "0".
- You must choose a "{alias}_1" or a "0" response.
- If your response is "{alias}_1," then begin a new paragraph with "{alias}"_rationale:" and excerpt the sentences or
phrases that determined your decision. You are allowed to choose multiple sentences or phrases, divided by an
"<|SPL|>" token.
- Then, whether you have selected a "{alias}_1" or a "0" begin a new paragraph with "{alias}_explanation:" and provide
a two sentence explanation for your response.
"""

    return f"{role}{definition_block}{instruction}{clarification}{examples}"


def build_prompts_per_code(config, aliases, backend="gpt"):
    """
    Build a prompts_per_code dict for use with code_texts_deductively_gpt.

    Parameters
    ----------
    config : dict
        The loaded codes config.
    aliases : list[str]
        List of code aliases to build prompts for.
    backend : str
        'gpt' or 'llama'.

    Returns
    -------
    dict
        {alias: [prompt_string]} for each alias.
    """
    builder = build_prompt_gpt if backend == "gpt" else build_prompt_llama
    return {alias: [builder(config, alias)] for alias in aliases}


# ---------------------------------------------------------------------------
# Coding functions
# ---------------------------------------------------------------------------

def code_texts_deductively_llama(
    df, alias, text_column, endpoint_url, prompt_template, model_name,
):
    """
    Classifies each row of 'text' column in provided df in accord with
    human-specified prompt, includes chain-of-thought reasoning, returning
    explanations for classification decision.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the text to classify.
    alias : str
        The alias (for brevity) of the qualitative code to be applied.
    text_column : str
        The column name in df containing the text to be analyzed.
    endpoint_url : str
        The URL where locally hosted Llama model runs.
    prompt_template : str
        The prompt text with a placeholder (e.g., '{text}') where the
        row's text will be inserted.
    model_name : str
        The model tasked with qualitative deductive coding.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with two new columns: '{alias}_llm'
        (either "0" or "1") and '{alias}_expl' (the explanation).
    """
    label_column = f"{alias}_llm"
    explanation_column = f"{alias}_expl"

    df[label_column] = None
    df[explanation_column] = None

    for idx, row in df.iterrows():
        row_text = row[text_column]
        prompt = prompt_template.format(text=row_text)

        response = requests.post(
            endpoint_url,
            headers={"Content-Type": "application/json"},
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
            },
        )

        print(response.status_code)
        print(response.text)

        if response.status_code == 200:
            try:
                result_json = response.json()
                raw_response_str = result_json.get("response", " ")

                start_idx = raw_response_str.find("{")
                end_idx = raw_response_str.rfind("}") + 1

                if start_idx != -1 and end_idx != -1:
                    valid_json_str = raw_response_str[start_idx:end_idx]
                    parsed_output = json.loads(valid_json_str)
                    label = parsed_output.get(label_column)
                    explanation = parsed_output.get(explanation_column)
                else:
                    print("No valid JSON found in response.")
                    label = None
                    explanation = None

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print("Parsing error:", e)
                label = None
                explanation = None
        else:
            label = None
            explanation = None

        df.at[idx, label_column] = label
        df.at[idx, explanation_column] = explanation

    return df


def code_instance_deductively_gpt(text, prompts, client=None):
    """
    Applies annotation decisions, based on multiple prompts, to a given
    text; provides rationale and explanation.

    Parameters
    ----------
    text : str
        The text to annotate.
    prompts : list[str]
        A list of prompts to apply to the text.
    client : openai.OpenAI, optional
        An OpenAI client. If None, one is created from OPENAI_API_KEY.

    Returns
    -------
    str
        The combined result from all prompts.
    """
    client = _get_openai_client(client)

    try:
        prompt_content = " ".join(prompts)

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            messages=[
                {"role": "system", "content": prompt_content},
                {"role": "user", "content": text},
            ],
        )

        result = " "
        for choice in response.choices:
            result += choice.message.content

        print(f"{text}: {result}")
        return result

    except Exception as e:
        print(f"Exception: {e}")
        return "error"


def code_texts_deductively_gpt(df, prompts_per_code, client=None):
    """
    Applies code_instance_deductively_gpt for multiple codes to each
    row in dataframe 'df'.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing texts to annotate.
    prompts_per_code : dict
        A dictionary with tag names as keys and a list of prompts
        as values.
    client : openai.OpenAI, optional
        An OpenAI client. If None, one is created from OPENAI_API_KEY.

    Returns
    -------
    pandas.DataFrame
        The updated dataframe with annotation results.
    """
    client = _get_openai_client(client)

    for index, row in df.iterrows():
        for tag, prompts in prompts_per_code.items():
            result = code_instance_deductively_gpt(
                row["text"], prompts, client=client,
            )
            if result == "error":
                continue

            rationale, explanation = None, None

            if f"{tag}_1" in result:
                tag_value = 1
                rationale = (
                    result.split(f"{tag}_rationale:")[1]
                    .split(f"{tag}_explanation:")[0]
                    .strip()
                    if f"{tag}_rationale:" in result
                    else None
                )
                explanation = (
                    result.split(f"{tag}_explanation:")[1].strip()
                    if f"{tag}_explanation:" in result
                    else None
                )
            else:
                tag_value = 0

            df.at[index, f"{tag}_gpt"] = tag_value
            df.at[index, f"{tag}_rtnl_gpt"] = rationale
            df.at[index, f"{tag}_expl_gpt"] = explanation

            time.sleep(1)

    return df
