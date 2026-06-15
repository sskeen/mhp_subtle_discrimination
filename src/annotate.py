"""
annotate.py
Qualitative deductive coding functions for CHALET-style LLM-human annotation.

Simone J. Skeen x Claude Code (05-28-2026)
"""

import os
import time

import ollama
import openai
import pandas as pd
import spacy
import yaml
from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------------------
# SpaCy model
# ---------------------------------------------------------------------------

nlp = spacy.load('en_core_web_lg')


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
        Path to annotation_schema.yaml. Defaults to annotation/schema/annotation_schema.yaml
        relative to the project root.

    Returns
    -------
    dict
        Parsed YAML configuration with 'role' and 'codes' keys.
    """
    if yaml_path is None:
        # Navigate from src/ to project root, then to annotation/schema/
        project_root = os.path.dirname(os.path.dirname(__file__))
        yaml_path = os.path.join(project_root, "annotation", "schema", "annotation_schema.yaml")
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


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
- If your response is "{alias}_1," then begin a new paragraph with "{alias}_rationale:" and excerpt the sentences or
phrases that determined your decision. You are allowed to choose multiple sentences or phrases, divided by an
"<|SPL|>" token.
- Then, whether you have selected a "{alias}_1" or a "0," begin a new paragraph with "{alias}_explanation:" and provide
a two sentence explanation for your response.
"""

    return f"{role}{definition_block}{instruction}{clarification}{examples}"


def build_prompts_per_code(config, aliases):
    """
    Build a prompts_per_code dict for use with code_texts_deductively_gpt or _qwen.

    Parameters
    ----------
    config : dict
        The loaded codes config.
    aliases : list[str]
        List of code aliases to build prompts for.

    Returns
    -------
    dict
        {alias: [prompt_string]} for each alias.
    """
    return {alias: [build_prompt_gpt(config, alias)] for alias in aliases}


# ---------------------------------------------------------------------------
# LLM coding functions
# ---------------------------------------------------------------------------

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

            rationale = None

            if f"{tag}_1" in result:
                tag_value = 1
                rationale = (
                    result.split(f"{tag}_rationale:")[1]
                    .split(f"{tag}_explanation:")[0]
                    .strip()
                    if f"{tag}_rationale:" in result
                    else None
                )
            else:
                tag_value = 0

            # Extract explanation for all decisions
            explanation = (
                result.split(f"{tag}_explanation:")[1].strip()
                if f"{tag}_explanation:" in result
                else None
            )

            df.at[index, f"{tag}_gpt"] = tag_value
            df.at[index, f"{tag}_rtnl_gpt"] = rationale
            df.at[index, f"{tag}_expl_gpt"] = explanation

            time.sleep(1)

    return df

def majority_vote_gpt(df, codes):
    """
    Create triangulated columns based on agreement between human and GPT annotations.

    For each code, creates a new column '{code}_triangulate' with value 1 if both
    the human annotation ({code}) and GPT annotation ({code}_gpt) equal 1.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing human and GPT annotations.
    codes : list[str]
        List of code aliases to triangulate (e.g., ['afrm', 'agnt', 'fitt']).

    Returns
    -------
    pandas.DataFrame
        The dataframe with new '{code}_triangulate' columns added.
    """
    for code in codes:
        human_col = code
        gpt_col = f"{code}_gpt"
        triangulate_col = f"{code}_triangulate"

        df[triangulate_col] = ((df[human_col] == 1) & (df[gpt_col] == 1)).astype(int)

    return df


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

def remove_index_artifacts(df):
    """
    Remove '/;' index artifact from DataFrame if present.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to clean.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with artifact removed.
    """
    if df.index.name == '/;':
        df = df.reset_index()
    if '/;' in df.columns:
        df = df.drop(columns=['/;'])
    return df


def condense_response_frame(d, pilot_value = 0):
    """
    Reduce and de-identify response dataframe.

    Parameters
    ----------
    d : pandas.DataFrame
        The DataFrame to process.
    pilot_value : int, optional
        Value to assign to the 'pilot' column. Default is 0.

    Returns
    -------
    pandas.DataFrame
        Processed DataFrame with selected columns.
    """
    # reduce and de-identify

    d.drop([
        'Email Address', 'Client Name',
        'INBOX Line', 'RAW Message',
        ],
        axis = 1,
        inplace = True,)

    # 'pilot' var

    d['pilot'] = pilot_value

    # float to int

    cols_to_int = ['EmailPairID', 'WithinPatientID', 'MHP ID']
    d[cols_to_int] = d[cols_to_int].fillna(0).round().astype(int)

    # drop junk rows

    d = d[d['Unique ID'] != "___"]

    # rename, reformat

    d = d.rename(columns = {'Cleaned Message': 'text'})

    d = d[[
        'EmailPairID', 'WithinPatientID', 'FirstInPair',
        'MHP ID', 'Unique ID', 'pilot', 'text',
        ]]

    return d


def ner_redact_response_texts(mhp_text):
    """
    Redacts all named entities recognized by spaCy EntityRecognizer, replaces with <|PII|> pseudo-word token.

    Parameters
    ----------
    mhp_text : str
        The text to redact.

    Returns
    -------
    str
        Text with named entities replaced by <|PII|>.
    """
    ne = list([
        'PERSON', 'NORP', 'FAC', 'ORG',
        'GPE', 'LOC', 'PRODUCT', 'EVENT',
        ])

    doc = nlp(mhp_text)
    ne_to_remove = []
    final_string = str(mhp_text)
    for sent in doc.ents:
        if sent.label_ in ne:
            ne_to_remove.append(str(sent.text))
    for i in range(len(ne_to_remove)):
        final_string = final_string.replace(
            ne_to_remove[i],
            '<|PII|>',
            )
    return final_string


# ---------------------------------------------------------------------------
# Sampling & inter-rater reliability
# ---------------------------------------------------------------------------

def sample_by_cycle(d_pilot, sample_size, cycle_number):
    """
    Creates random subsample of d_pilot, excises prior tags, and unneeded columns,
    exports to .xlsx for human annotation.

    Parameters:
    -----------

    d_pilot : pd.DataFrame
        The d_pilot df in memory.

    sample_size : int
        The number of rows to sample from d_pilot.

    cycle_number : int
        The cycle number used to name the output dataframe and the Excel file.

    Returns:
    --------
    pd.DataFrame
        A new dataframe called d_cycle_{cycle_number} after applying the operations.
    """

    d_cycle = d_pilot.sample(
        n = sample_size,
        #random_state = 56,
        )

    # reset index

    d_cycle.reset_index(
        drop = True,
        inplace = True,
        )

    # DAL themes

    d_cycle['brdn'] = ' '
    d_cycle['dmnd'] = ' '
    d_cycle['rbnd'] = ' '

    # excise prior tags

    tag_cols = [
        'prbl', 'refl', 'just', 'afrm', 'fitt', 'agnt',
        'brdn', 'dmnd', 'rbnd', 'rtnl', 'note',
        ]

    d_cycle[tag_cols] = ' '

    # excise unneeded columns

    drop_cols = [
        'EmailPairID', 'WithinPatientID', 'FirstInPair',
        #'pilot', 'MHP ID#',
        ]

    d_cycle.drop(
        columns = drop_cols,
        axis = 1,
        inplace = True,
        )

    # export

    filename = f'd_cycle_{cycle_number}.xlsx'

    d_cycle.to_excel(
        filename,
        index = True,
        )

    return d_cycle


def calculate_kappa_by_cycle(cycle_num):
    """
    Calculate Cohen's Kappa and encode disagreements between independent annotators across multiple cycles.

    Parameters:
    --------
    cycle_num : int
        Annotation cycle number, used to load the corresponding Excel files (e.g., cycle 0, cycle 1).

    Returns:
    --------
    d : pd.DataFrame
        Processed df after merging, includes encoded disagreements in *_dis columns.

    kappa_results : dict
        A dictionary containing the Cohen's Kappa scores for each indepednently co-annotated target.
    """
    # read independently annotated files

    d_dal = pd.read_excel(f'd_cycle_{cycle_num}_dal.xlsx', index_col = [0])
    d_dal.columns = [f'{col}_dal' for col in d_dal.columns]

    d_sjs = pd.read_excel(f'd_cycle_{cycle_num}_sjs.xlsx', index_col = [0])
    d_sjs.columns = [f'{col}_sjs' for col in d_sjs.columns]

    # merge

    d = pd.merge(
        d_dal,
        d_sjs,
        left_index = True,
        right_index = True,
        )

    # housekeeping

    targets = [
        'afrm_dal', 'afrm_sjs', 'agnt_dal', 'agnt_sjs', 'brdn_dal', 'brdn_sjs',
        'dmnd_dal', 'dmnd_sjs', 'fitt_dal', 'fitt_sjs', 'just_dal', 'just_sjs',
        'prbl_dal', 'prbl_sjs', 'rbnd_dal', 'rbnd_sjs', 'refl_dal', 'refl_sjs',
        ]

    texts = [
        'text_dal', 'text_sjs',
        'rtnl_dal', 'rtnl_sjs',
        'note_dal', 'note_sjs',
        ]

    d[targets] = d[targets].apply(
        pd.to_numeric,
        errors = 'coerce',
        )
    d[targets] = d[targets].fillna(0)
    d[texts] = d[texts].replace(' ', '.')

    d = d[[
        'text_dal',
        'afrm_dal', 'afrm_sjs', 'agnt_dal', 'agnt_sjs', 'brdn_dal', 'brdn_sjs',
        'dmnd_dal', 'dmnd_sjs', 'fitt_dal', 'fitt_sjs', 'just_dal', 'just_sjs',
        'prbl_dal', 'prbl_sjs', 'rbnd_dal', 'rbnd_sjs', 'refl_dal', 'refl_sjs',
        'rtnl_dal', 'rtnl_sjs', 'note_dal', 'note_sjs',
        ]].copy()

    # kappa fx

    def calculate_kappa(d, col_dal, col_sjs):
        return cohen_kappa_score(d[col_dal], d[col_sjs])

    col_pairs = [
        ('afrm_dal', 'afrm_sjs'), ('agnt_dal', 'agnt_sjs'), ('brdn_dal', 'brdn_sjs'),
        ('dmnd_dal', 'dmnd_sjs'), ('fitt_dal', 'fitt_sjs'), ('just_dal', 'just_sjs'),
        ('prbl_dal', 'prbl_sjs'), ('rbnd_dal', 'rbnd_sjs'), ('refl_dal', 'refl_sjs'),
        ]

    # initialize dict

    kappa_results = {}

    # kappa loop
    print("\n--------------------------------------------------------------------------------------")
    print(f"Cycle {cycle_num}: Cohen's Kappa by target")
    print("--------------------------------------------------------------------------------------")

    for col_dal, col_sjs in col_pairs:
        kappa = calculate_kappa(d, col_dal, col_sjs)
        kappa_results[f'{col_dal} and {col_sjs}'] = kappa

    for pair, kappa in kappa_results.items():
        print(f"{pair} Kappa = {kappa:.2f}")

    # dummy code disagreements fx

    def encode_disagreements(row):
        return 1 if row[0] != row[1] else 0

    col_dis = [
        ('afrm_dal', 'afrm_sjs', 'afrm_dis'), ('agnt_dal', 'agnt_sjs', 'agnt_dis'), ('brdn_dal', 'brdn_sjs', 'brdn_dis'),
        ('dmnd_dal', 'dmnd_sjs', 'dmnd_dis'), ('fitt_dal', 'fitt_sjs', 'fitt_dis'), ('just_dal', 'just_sjs', 'just_dis'),
        ('prbl_dal', 'prbl_sjs', 'prbl_dis'), ('rbnd_dal', 'rbnd_sjs', 'rbnd_dis'), ('refl_dal', 'refl_sjs', 'refl_dis'),
        ]

    for col1, col2, dis_col in col_dis:
        d[dis_col] = d[[col1, col2]].apply(encode_disagreements, axis = 1)

    # display counts for targets

    print("\n--------------------------------------------------------------------------------------")
    print(f"Cycle {cycle_num}: Counts by target")
    print("--------------------------------------------------------------------------------------")
    print(d[targets].apply(pd.Series.value_counts))

    # drop target cols for readability + fillna

    d = d.drop(
        targets,
        axis = 1,
        )
    d = d.fillna('.')

    # export: cycle-specific

    d.to_excel(f'd_cycle_{cycle_num}_dis.xlsx')

    return d, kappa_results