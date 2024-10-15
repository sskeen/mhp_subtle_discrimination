# -*- coding: utf-8 -*-
"""mhp_annotate_iaa_append.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YvPRUGVIBNLFiDOHP0VClmkTz8AmUOQ0

## Linguistic markers of subtle discrimination among mental healthcare professionals

_TKTK_

> mhp_annotate_iaa_append.ipynb<br>
> Simone J. Skeen (10-14-2024)

1. [Prepare](xx)
2. [Write](xx)
3. [Sample](xx)
4. [Calculate $\kappa$](xx)

### Prepare
Installs, imports, and downloads requisite models and packages. Organizes RAP-consistent directory structure.
***

**Install**
"""

#%%capture

#!python -m spacy download en_core_web_lg --user

"""**Import**"""

import numpy as np
import os
import pandas as pd
import re
import spacy
import warnings

from bs4 import BeautifulSoup

from google.colab import drive

#spacy.cli.download('en_core_web_lg')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

pd.options.mode.copy_on_write = True

pd.set_option(
              'display.max_columns',
              None,
              )

pd.set_option(
              'display.max_rows',
              None,
              )

warnings.simplefilter(
                      action = 'ignore',
                      category = FutureWarning,
                      )

#!python -m prodigy stats

"""**Mount gdrive**"""

drive.mount(
            '/content/drive',
            #force_remount = True,
            )

"""**Structure directories**"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My Drive/Colab/mhp_subtle_discrimination
#%cd /content/drive/My Drive/#<my_project_folder>

#%mkdir inputs

#%cd inputs
#%mkdir annotation data html

mhp_subtle_discrimination/
└── inputs/
    ├── annotation/
    │   └── #d_cycle_{0...n}xlsx
    ├── data/
    │   └── d_pilot.xlsx
    └── html/

"""#### Housekeeping: $\mathcal{d}$<sub>pilot</sub>"""

# Commented out IPython magic to ensure Python compatibility.
# %cd inputs/data

d_pilot = pd.read_excel(
                        'd_pilot.xlsx',
                        index_col = 'index',
                        )
# 'pilot' var

d_pilot['pilot'] = 1

# 'MHP ID#' var

d_pilot['MHP ID#'] = '.'

d_pilot.info()
d_pilot.head(3)

d_pilot.to_excel(
                 'd_pilot.xlsx',
                 index = True,
                 )

"""### 2. Write
Writes and imports custom modules
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd code

"""#### preprocess.py

**_ner_redact_response_texts_**
"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile preprocess.py
# 
# import spacy
# nlp = spacy.load('en_core_web_lg')
# 
# def ner_redact_response_texts(mhp_text):
#     """
#     Redacts all named entities recognized by spaCy EntityRecognizer, replaces with <|PII|> pseudo-word token.
#     """
#     ne = list(
#               [
#                'PERSON',   ### people, including fictional
#                'NORP',     ### nationalities or religious or political groups
#                'FAC',      ### buildings, airports, highways, bridges, etc.
#                'ORG',      ### companies, agencies, institutions, etc.
#                'GPE',      ### countries, cities, states
#                'LOC',      ### non-GPE locations, mountain ranges, bodies of water
#                'PRODUCT',  ### objects, vehicles, foods, etc. (not services)
#                'EVENT',    ### named hurricanes, battles, wars, sports events, etc.
#                ]
#                 )
# 
#     doc = nlp(mhp_text)
#     ne_to_remove = []
#     final_string = str(mhp_text)
#     for sent in doc.ents:
#         if sent.label_ in ne:
#             ne_to_remove.append(str(sent.text))
#     for i in range(len(ne_to_remove)):
#         final_string = final_string.replace(
#                                             ne_to_remove[i],
#                                             '<|PII|>',
#                                             )
#     return final_string

"""#### annotate.py

**_sample_by_cycle_**
"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile annotate.py
# 
# import pandas as pd
# 
# def sample_by_cycle(d_pilot, sample_size, cycle_number):
#     """
#     Creates random subsample of d_pilot, excises prior tags, and unneeded columns,
#     exports to .xlsx for human annotation.
# 
#     Parameters:
#     -----------
# 
#     d_pilot : pd.DataFrame
#         The d_pilot df in memory.
# 
#     sample_size : int
#         The number of rows to sample from d_pilot.
# 
#     cycle_number : int
#         The cycle number used to name the output dataframe and the Excel file.
# 
#     Returns:
#     --------
#     pd.DataFrame
#         A new dataframe called d_cycle_{cycle_number} after applying the operations.
#     """
# 
#     d_cycle = d_pilot.sample(
#                              n = sample_size,
#                              #random_state = 56,
#                              )
# 
#     # reset index
# 
#     d_cycle.reset_index(
#                         drop = True,
#                         inplace = True,
#                         )
#     # DAL themes
# 
#     d_cycle['brdn'] = ' '
#     d_cycle['dmnd'] = ' '
#     d_cycle['rbnd'] = ' '
# 
#     # excise prior tags
# 
#     tag_cols = [
#                 'prbl',
#                 'refl',
#                 'just',
#                 'afrm',
#                 'fitt',
#                 'agnt',
#                 'brdn',
#                 'dmnd',
#                 'rbnd',
#                 'rtnl',
#                 'note',
#                 ]
# 
#     d_cycle[tag_cols] = ' '
# 
#     # excise unneeded columns
# 
#     drop_cols = [
#                  'EmailPairID',
#                  'WithinPatientID',
#                  'FirstInPair',
#                  'pilot',
#                  'MHP ID#',
#                  ]
# 
#     d_cycle.drop(
#                  columns = drop_cols,
#                  axis = 1,
#                  inplace = True,
#                  )
# 
#     # export
# 
#     filename = f'd_cycle_{cycle_number}.xlsx'
# 
#     d_cycle.to_excel(
#                      filename,
#                      index = True,
#                      )
# 
#     return d_cycle

"""#### calculate.py

**_calculate_kappa_by_cycle_**
"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile calculate.py
# 
# import pandas as pd
# from sklearn.metrics import cohen_kappa_score
# 
# def calculate_kappa_by_cycle(cycle_num):
#     """
#     Calculate Cohen's Kappa and encode disagreements between independent annotators across multiple cycles.
# 
#     Parameters:
#     --------
#     cycle_num : int
#         Annotation cycle number, used to load the corresponding Excel files (e.g., cycle 0, cycle 1).
# 
#     Returns:
#     --------
#     d : pd.DataFrame
#         Processed df after merging, includes encoded disagreements in *_dis columns.
# 
#     kappa_results : dict
#         A dictionary containing the Cohen's Kappa scores for each indepednently co-annotated target.
#     """
#     # read independently annotated files
# 
#     d_dal = pd.read_excel(f'd_cycle_{cycle_num}_dal.xlsx', index_col = [0])
#     d_dal.columns = [f'{col}_dal' for col in d_dal.columns]
# 
#     d_sjs = pd.read_excel(f'd_cycle_{cycle_num}_sjs.xlsx', index_col = [0])
#     d_sjs.columns = [f'{col}_sjs' for col in d_sjs.columns]
# 
#     # merge
# 
#     d = pd.merge(
#                  d_dal,
#                  d_sjs,
#                  left_index = True,
#                  right_index = True,
#                  )
# 
#     # housekeeping
# 
#     targets = [
#                'afrm_dal', 'afrm_sjs',
#                'agnt_dal', 'agnt_sjs',
#                'brdn_dal', 'brdn_sjs',
#                'dmnd_dal', 'dmnd_sjs',
#                'fitt_dal', 'fitt_sjs',
#                'just_dal', 'just_sjs',
#                'prbl_dal', 'prbl_sjs',
#                'rbnd_dal', 'rbnd_sjs',
#                'refl_dal', 'refl_sjs',
#                ]
# 
#     texts = [
#              'text_dal', 'text_sjs',
#              'rtnl_dal', 'rtnl_sjs',
#              'note_dal', 'note_sjs',
#              ]
# 
#     d[targets] = d[targets].apply(
#                                   pd.to_numeric,
#                                   errors = 'coerce',
#                                   )
#     d[targets] = d[targets].fillna(0)
#     d[texts] = d[texts].replace(' ', '.')
# 
#     d = d[[
#            'text_dal',
#            'afrm_dal', 'afrm_sjs',
#            'agnt_dal', 'agnt_sjs',
#            'brdn_dal', 'brdn_sjs',
#            'dmnd_dal', 'dmnd_sjs',
#            'fitt_dal', 'fitt_sjs',
#            'just_dal', 'just_sjs',
#            'prbl_dal', 'prbl_sjs',
#            'rbnd_dal', 'rbnd_sjs',
#            'refl_dal', 'refl_sjs',
#            'rtnl_dal', 'rtnl_sjs',
#            'note_dal', 'note_sjs',
#            ]].copy()
# 
#     # kappa Fx
# 
#     def calculate_kappa(d, col_dal, col_sjs):
#         return cohen_kappa_score(d[col_dal], d[col_sjs])
# 
#     col_pairs = [
#                  ('afrm_dal', 'afrm_sjs'),
#                  ('agnt_dal', 'agnt_sjs'),
#                  ('brdn_dal', 'brdn_sjs'),
#                  ('dmnd_dal', 'dmnd_sjs'),
#                  ('fitt_dal', 'fitt_sjs'),
#                  ('just_dal', 'just_sjs'),
#                  ('prbl_dal', 'prbl_sjs'),
#                  ('rbnd_dal', 'rbnd_sjs'),
#                  ('refl_dal', 'refl_sjs'),
#                  ]
# 
#     # initialize dict
# 
#     kappa_results = {}
# 
#     # kappa loop
#     print("\n--------------------------------------------------------------------------------------")
#     print(f"Cycle {cycle_num}: Cohen's Kappa by target")
#     print("--------------------------------------------------------------------------------------")
# 
#     for col_dal, col_sjs in col_pairs:
#         kappa = calculate_kappa(d, col_dal, col_sjs)
#         kappa_results[f'{col_dal} and {col_sjs}'] = kappa
# 
#     for pair, kappa in kappa_results.items():
#         print(f"{pair} Kappa = {kappa:.2f}")
# 
#     # dummy code disagreements Fx
# 
#     def encode_disagreements(row):
#         return 1 if row[0] != row[1] else 0
# 
#     col_dis = [
#                ('afrm_dal', 'afrm_sjs', 'afrm_dis'),
#                ('agnt_dal', 'agnt_sjs', 'agnt_dis'),
#                ('brdn_dal', 'brdn_sjs', 'brdn_dis'),
#                ('dmnd_dal', 'dmnd_sjs', 'dmnd_dis'),
#                ('fitt_dal', 'fitt_sjs', 'fitt_dis'),
#                ('just_dal', 'just_sjs', 'just_dis'),
#                ('prbl_dal', 'prbl_sjs', 'prbl_dis'),
#                ('rbnd_dal', 'rbnd_sjs', 'rbnd_dis'),
#                ('refl_dal', 'refl_sjs', 'refl_dis'),
#                ]
# 
#     for col1, col2, dis_col in col_dis:
#         d[dis_col] = d[[col1, col2]].apply(encode_disagreements, axis = 1)
# 
#     # display counts for targets
# 
#     print("\n--------------------------------------------------------------------------------------")
#     print(f"Cycle {cycle_num}: Counts by target")
#     print("--------------------------------------------------------------------------------------")
#     print(d[targets].apply(pd.Series.value_counts))
# 
#     # drop target cols for readability + fillna
# 
#     d = d.drop(targets, axis = 1)
#     d = d.fillna('.')
# 
#     # export: cycle-specific
# 
#     d.to_excel(f'd_cycle_{cycle_num}_dis.xlsx')
# 
#     return d, kappa_results

"""#### Import"""

from annotate import(
                     sample_by_cycle
                     )

#from preprocess import(
#                       ner_redact_response_texts
#                       )

from calculate import(
                      calculate_kappa_by_cycle
                      )

"""### 3. Sample
Randomly samples cycle-specific MHP response subsets for annotation.
***
"""

# Commented out IPython magic to ensure Python compatibility.
# %pwd

# Commented out IPython magic to ensure Python compatibility.
# %cd ../inputs/data

d_pilot = pd.read_excel(
                        'd_pilot.xlsx',
                        index_col = 'index',
                        )

#d_pilot.info()
#d_pilot.head(3)

"""#### Cycle 0"""

# Commented out IPython magic to ensure Python compatibility.
# %cd ../annotation

# sample

d_cycle_0 = d_pilot.sample(
                           n = 50,
                           random_state = 56,
                           )

# reset index

d_cycle_0.reset_index(
                      drop = True,
                      inplace = True,
                      )

# excise prior tags

tag_cols = [
            'afrm',
            'agnt',
            'fitt',
            'just',
            'prbl',
            'refl',
            'rtnl',
            'note',
            ]

d_cycle_0[tag_cols] = ' '

# excise unneeded cols

drop_cols = [
             'EmailPairID',
             'WithinPatientID',
             'FirstInPair',
             'pilot',
             'MHP ID#',
             ]

    ### SJS 9/16: add DAL targets (for now): brdn, dmnd, rbnd

d_cycle_0.drop(
               columns = drop_cols,
               axis = 1,
               inplace = True,
               )

# export

d_cycle_0.head(3)

d_cycle_0.to_excel(
                   'd_cycle_0.xlsx',
                   index = True,
                   )

"""#### Cycle 1"""

# Commented out IPython magic to ensure Python compatibility.
# %cd ../annotation

# call sample_by_cycle

d_cycle_1 = sample_by_cycle(
                            d_pilot,
                            80, # sample_size = 80
                            1, # cycle_number = 1
                            )

d_cycle_1.info()
d_cycle_1.head(3)

"""#### Cycle 2"""

# Commented out IPython magic to ensure Python compatibility.
# %cd ../annotation

# call sample_by_cycle

d_cycle_2 = sample_by_cycle(
                            d_pilot,
                            80, # sample_size = 80
                            2, # cycle_number = 2
                            )

d_cycle_2.info()
d_cycle_2.head(3)

"""### 4. Calculate $\kappa$

#### Cycle 0
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd ../inputs/annotation

d, kappa_results = calculate_kappa_by_cycle(0)

"""#### Cycle 1"""

# Commented out IPython magic to ensure Python compatibility.
# %cd ../inputs/annotation

d, kappa_results = calculate_kappa_by_cycle(1)

d.head(3)

"""### html - pilot"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My Drive/Colab/mhp_subtle_discrimination/inputs/html/TEST

html = [file for file in os.listdir() if file.endswith(('.htm', '.html'))]

# initialize list

html_data = []

# load

for h in html:
    with open(h, 'r', encoding = 'utf-8') as file:
        content = file.read()

    # parse .html

    soup = BeautifulSoup(
                         content,
                         'html.parser',
                         )

    # extract attributes

    name_title = soup.find(
                           'meta',
                           property = 'og:title',
                           )

    profile = soup.find(
                        'meta',
                        property = 'og:url',
                        )
    image = soup.find(
                      'meta',
                      property = 'og:image',
                      )

    # extract attribute contents

    practice_name_text = name_title['content'] if name_title else '.'
    #description_text = description['content'] if description else '.'
    profile_url = profile['content'] if profile else '.'
    image_url = image['content'] if image else '.'

    # extract filename as MHP ID#

    mhp_id = h.replace('.html', ' ').replace('.htm', ' ')

    # extract full text

    full_text = soup.get_text()

    # 'pronouns' str: extract text preceding "Verified"

    extracted_text = re.search(
                               r'^(.*?)Verified',
                               full_text,
                               re.DOTALL,
                               )

    if extracted_text:
        extracted_text = extracted_text.group(1).strip()
    else:
        extracted_text = ' '

    # extract pronouns from parens

    pronoun_text = re.findall(r'\(([^0-9]+?)\)', extracted_text)
    pronoun_text = ' '.join(pronoun_text).strip()

    # 'description' str: extract text between "Let's Connect" and "Call or Email"

    start_description = full_text.find("Let's Connect")
    end_description = full_text.find("Call or Email", start_description)
    description_text = full_text[start_description + len("Let's Connect"):end_description].strip() \
    if start_description != -1 \
    and end_description != -1 \
    else '.'

    # 'at_a_glance' str (incl 'finances'): extract text between "Practice at a Glance" and "Qualifications"

    start_glance = full_text.find("Practice at a Glance")
    end_glance = full_text.find("Qualifications", start_glance)
    glance_text = full_text[start_glance + len("Practice at a Glance"):end_glance].strip() \
    if start_glance != -1 \
    and end_glance != -1 \
    else '.'

    # Extract the text between "Finances" and "Insurance"
    #start_finances = full_text.find("Finances", end_glance)
    #end_finances = full_text.find("Insurance", start_finances)
    #finances_text = full_text[start_finances + len("Finances"):end_finances].strip() \
    #if start_finances != -1 \
    #and end_finances != -1 \
    #else '.'

    # Extract the text between "Insurance" and "Check fees"
    #start_insurance = full_text.find("Insurance", end_finances)
    #end_insurance = full_text.find("Check fees", start_insurance)
    #insurance_text = full_text[start_insurance + len("Insurance"):end_insurance].strip() \
    #if start_insurance != -1 \
    #and end_insurance != -1 \
    #else '.'

    # 'qualifications' str: extract text between "Qualifications" and "Feel free to ask"

    start_qualifications = full_text.find("Qualifications")
    end_qualifications = full_text.find("Feel free to ask", start_qualifications)
    qualifications_text = full_text[start_qualifications + len("Qualifications"):end_qualifications].strip() \
    if start_qualifications != -1 \
    and end_qualifications != -1 \
    else '.'

    # Extract the text between "Top Specialties" and "Do these issues"
    start_specialties = full_text.find("Top Specialties")
    end_specialties = full_text.find("Do these issues", start_specialties)
    specialties_text = full_text[start_specialties + len("Top Specialties"):end_specialties].strip() \
    if start_specialties != -1 \
    and end_specialties != -1 \
    else '.'

    # Extract the text between "Client Focus" and "Religion"
    start_client = full_text.find("Client Focus")
    end_client = full_text.find("Religion", start_client)
    client_text = full_text[start_client + len("Client Focus"):end_client].strip() \
    if start_client != -1 \
    and end_client != -1 \
    else '.'

    # Extract the text between "Religion" and "Treatment Approach"
    start_religion = full_text.find("Religion")
    end_religion = full_text.find("Treatment Approach", start_religion)
    religion_text = full_text[start_religion + len("Religion"):end_religion].strip() \
    if start_religion != -1 \
    and end_religion != -1 \
    else '.'

    # Extract the text between "Types of Therapy" and "Ask about what"
    start_therapy = full_text.find("Types of Therapy")
    end_therapy = full_text.find("Ask about what", start_therapy)
    therapy_text = full_text[start_therapy + len("Types of Therapy"):end_therapy].strip() \
    if start_therapy != -1 \
    and end_therapy != -1 \
    else '.'

    # append to list

    html_data.append({
                      'MHP ID#': mhp_id,
                      'practice_name': practice_name_text,
                      'pronouns': pronoun_text,
                      'description': description_text,
                      'profile_url': profile_url,
                      'image_url': image_url,
                      'at_a_glance': glance_text,
                      #'finance': finances_text,
                      #'insurance': insurance_text,
                      'qualifications': qualifications_text,
                      'specialties': specialties_text,
                      'client_focus': client_text,
                      'religion': religion_text,
                      'types_of_therapy': therapy_text,
                      })

# build df

d = pd.DataFrame(html_data)

# 'finances' str: extract text following "Finances" from 'at_a_glance'

d['finances'] = d['at_a_glance'].str.extract(
                                             r'Finances(.*)',
                                             expand = False,
                                             )

d['finances'] = d['finances'].fillna('.').str.strip()

# delete "Finances" from 'at_a_glance'

d['at_a_glance'] = d['at_a_glance'].str.replace(
                                                r'Finances.*',
                                                ' ',
                                                regex = True,
                                                n = 1,
                                                ).str.strip()

# 'name' str: extract from 'practice_name'

d['name'] = d['practice_name']
d['name'] = d['name'].str.split(
                                ',',
                                n = 1,
                                ).str[0].str.strip()

# 'availability' str: extract (pre-specified) from 'at_a_glance'

availabilities = [
                  'Available both in-person and online',
                  'Available in-person',
                  'Available online',
                  ]

d['availability'] = d['at_a_glance'].str.extract(
                                                 f"({'|'.join(availabilities)})",
                                                 expand = False,
                                                 )

d['availability'] = d['availability'].fillna('.')

    ### SJS 9/19: loop over these fillna('.') at end...TKTK

# 'years_in_practice' str: extract from 'qualifications

d['years_in_practice'] = d['qualifications'].str.extract(
                                                         r'In Practice for (\d+) Years',
                                                         expand = False,
                                                         )

d['years_in_practice'] = pd.to_numeric(
                                       d['years_in_practice'],
                                       errors = 'coerce',
                                       )

d['years_in_practice'] = d['years_in_practice'].fillna('.')


# 'degrees_title' str: extract from full_text

d['degrees_title'] = d.apply(lambda row: re.search(rf"{re.escape(row['name'])}\s*(.*?)\s*Specialties and Expertise",
                                                     full_text).group(1) if re.search(rf"{re.escape(row['name'])}\s*(.*?)\s*Specialties and Expertise",
                                                     full_text) else '', axis=1)

d['degrees_title'] = d['degrees_title'].str.strip()

    ### SJS 9/19: this does _not_ currently work...


# housekeeping

# delete PT footer from 'practice_name'

d['practice_name'] = d['practice_name'].str.replace(
                                              '| Psychology Today',
                                              ' ',
                                              regex = False,
                                              )

# delete contact details from 'description'

tel_re = r'\(\d{3}\) \d{3}-\d{4}'

d['description'] = d['description'].str.replace(
                                                'Take the first step to help',
                                                ' ',
                                                regex = False,
                                                )

d['description'] = d['description'].str.replace(
                                                'Email me',
                                                ' ',
                                                regex = False,
                                                )

d['description'] = d['description'].str.replace(
                                                'Email us',
                                                ' ',
                                                regex = False,
                                                )

d['description'] = d['description'].str.replace(
                                                tel_re,
                                                ' ',
                                                regex = True,
                                                )

# delete duped text (follows "Let's Connect") from 'at_a_glance'

d['at_a_glance'] = d['at_a_glance'].str.replace(
                                                r"Let's Connect.*",
                                                ' ',
                                                regex = True,
                                                ).str.strip()

# add space: 'specialties'

d['specialties'] = d['specialties'].str.replace(
                                                r'([a-z])([A-Z])',
                                                r'\1 \2',
                                                regex = True,
                                                )

d['specialties'] = d['specialties'].str.replace(
                                                r'(\(BPD\)|OCD\)|ADHD|LGBTQ\+|PTSD)',
                                                r'\1 ',
                                                regex = True,
                                                )

d['specialties'] = d['specialties'].str.strip()

# delete whitespace from 'pronouns'

d['pronouns'] = d['pronouns'].replace(
                                      ' ',
                                      np.nan,
                                      ).fillna('.').str.strip()

d['pronouns'] = d['pronouns'].replace(
                                      r'^\s*$',
                                      '.',
                                      regex = True,
                                      )

# reorder

d = d.reindex(
              columns = [
                         'MHP ID#',
                         'name',
                         'pronouns',
                         'practice_name',
                         'description',
                         'profile_url',
                         'image_url',
                         'at_a_glance',
                         'qualifications',
                         'specialties',
                         'client_focus',
                         'religion',
                         'types_of_therapy',
                         'finances',
                         'availability',
                         'years_in_practice',
                         'degrees_title',
                          ]
              )

# inspect

d.info()
d.head(3)

# Commented out IPython magic to ensure Python compatibility.
#%pwd
# %cd ../../outputs/tables

d.to_excel(
           'd_html.xlsx',
           index = True,
           )

"""> End of mhp_annotate_iaa_append.ipynb"""