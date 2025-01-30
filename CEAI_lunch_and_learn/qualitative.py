
import requests
import json
import pandas as pd

def code_texts_deductively_llama(df, alias, text_column, endpoint_url, prompt_template, model_name):
    """
    Classifies each row of 'text' column in provided df in accord with human-specified prompt,
    includes chain-of-thought reasoning, returning explanations for classification decision.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the text to classify.
    alias : str
        The alias (for brevity) of the qualitative code to be applied.
    text_column : str
        The column name in df containing the text to be analyzed.
    endpoint_url : str
        The URL where locally hosted Llama model runs.
    prompt_template : str
        The prompt text with a placeholder (e.g., '{text}') where the row's text will be inserted.
    model_name : str
        The model tasked with qualitative deductive coding.

    Returns:
    --------
    pandas.DataFrame
        The original DataFrame with two new columns: '{alias}_llm' (either "0" or "1")
        and '{alias}_expl' (the explanation).
    """

    # dynamically create {alias} column names

    label_column = f'{alias}_llm'
    explanation_column = f'{alias}_expl'

    # create empty tag ['*_llm'] and reasoning ['*_expl'] column

    df[label_column] = None
    df[explanation_column] = None

    for idx, row in df.iterrows():
        row_text = row[text_column]

        # replace '{text}' in prompt_template with df 'text' data

        prompt = prompt_template.format(text = row_text)

        # send request to local Llama endpoint.

        response = requests.post(
            endpoint_url,
            headers = {'Content-Type': 'application/json'},
            json = {
                'model': model_name,
                'prompt': prompt,
                'stream': False
                },
        )

        # print statements for debugging

        print(response.status_code)
        print(response.text)

        if response.status_code == 200:
            try:
                # parse top-level JSON

                result_json = response.json()

                # 'response' field contains JSON string

                raw_response_str = result_json.get('response', ' ')

                # extract only the JSON portion: identify first `{` and last `}` braces

                start_idx = raw_response_str.find("{")
                end_idx = raw_response_str.rfind("}") + 1

                if start_idx != -1 and end_idx != -1:

                # extract and parse JSON portion

                    valid_json_str = raw_response_str[start_idx:end_idx]
                    parsed_output = json.loads(valid_json_str)

                # extract tag and reasoning fields

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

        # insert classification results into df

        df.at[idx, label_column] = label
        df.at[idx, explanation_column] = explanation

    return df

import time
import openai

api_key = ' '
client = openai.OpenAI(api_key = api_key)

def code_instance_deductively_gpt(text, prompts):
    """
    Applies annotation decisions, based on multiple prompts, to a given text; provides rationale and explanation.
    Parameters:
    - text: The text to annotate.
    - prompts: A list of prompts to apply to the text.

    Returns:
    - result: The combined result from all prompts.
    """
    try:

        # concatenate prompts

        prompt_content = ' '.join(prompts)

        response = client.chat.completions.create(
            model = 'gpt-4o',
            temperature = 0.2,
            messages = [
                {
                    'role': 'system',
                    'content': prompt_content
                },
                {
                    'role': 'user',
                    'content': text
                }
            ]
        )

        # collect results

        result = ' '
        for choice in response.choices:
            result += choice.message.content

        print(f'{text}: {result}')
        return result
    except Exception as e:
        print(f'Exception: {e}')
        return 'error'

def code_texts_deductively_gpt(df, prompts_per_code):
    """
    Applies code_instance_deductively_gpt for multiple codes to each row in dataframe 'df'.

    Parameters:
    - df: The dataframe containing texts to annotate.
    - prompts_per_code: A dictionary with tag names as keys and a list of prompts as values.

    Returns:
    - df: The updated dataframe with annotation results.
    """
    for index, row in df.iterrows():
        for tag, prompts in prompts_per_code.items():
            result = code_instance_deductively_gpt(row['text'], prompts)
            if result == 'error':
                continue

            # initialize variables for annotation outputs

            rationale, explanation = None, None

            if f'{tag}_1' in result:
                tag_value = 1

                # extract rationale

                rationale = result.split(f'{tag}_rationale:')[1].split(f'{tag}_explanation:')[0].strip() if f'{tag}_rationale:' in result else None

                # extract explanation

                explanation = result.split(f'{tag}_explanation:')[1].strip() if f'{tag}_explanation:' in result else None

            else:
                tag_value = 0

            # results to df

            df.at[index, f'{tag}_gpt'] = tag_value
            df.at[index, f'{tag}_rtnl_gpt'] = rationale
            df.at[index, f'{tag}_expl_gpt'] = explanation

            # impose delay between API calls

            time.sleep(1)

    return df
