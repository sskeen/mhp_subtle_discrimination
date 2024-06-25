# -*- coding: utf-8 -*-
"""mhp_train_test_pilot.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cc-frKp-mYhjXd8hyY5Qd7vykNfhYQNm

## Linguistic markers of subtle discrimination among mental healthcare professionals

_Preprocesses, trains, tests (5-fold CV), and ranks feature importance of mental health professional (MHP) response quality targets while replying to appointment queries, using binary XGBoost classifiers. Fine tunes and evaluates (5-fold CV) rationale-augmented MHP response quality targets across BERT, RoBERTa, and DistilBERT pretrained LMs._

> mhp_train_test_pilot.ipynb<br>
> Simone J. Skeen (06-25-2024)
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install contractions
# %pip install simpletransformers
# %pip install unidecode
# %pip install wandb

!python -m spacy download en_core_web_lg --user

import contractions
import logging
import nltk
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import spacy
import string
import wandb.sdk
import warnings

from bs4 import BeautifulSoup
from collections import Counter
from functools import reduce
from google.colab import drive
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score, matthews_corrcoef
from textblob import TextBlob
from unidecode import unidecode
from xgboost import XGBClassifier

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

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

# Commented out IPython magic to ensure Python compatibility.
drive.mount(
            '/content/gdrive/',
            force_remount = True,
            )

# %cd gdrive/My Drive/Colab/mhp_subtle_discrimination/data

"""###Pre-annotation
***

**_Import_**
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# 
# d = pd.read_csv(
#                 'Therapy Discrimination - Data collection file - Response Data.csv',
#                 encoding = 'utf8',
#                 header = 1,
#                 )
# 
# print(len(d.columns))
# for col in d.columns:
#     print(col)
# d.head(3)

"""**_Reformat_**"""

# Reduce

d = d[[
       'Response (Y/N)',
       'Email Pair ID#',
       'Within Patient ID',
       'First Email in Pair? (Y/N)',
       'Email, Phone, or Text (E/P/T)',
       'Offered a pre-appt consultation/to talk on the phone? (Y/N)',
       'Implied or explicit appointment offer? (Y/N)',
       'Rejection? (Y/N)',
       'Asked about insurance/payment? (Y/N)',
       'Asked about trans/nonbinary issues? (Y/N)',
       'Copy-and-Paste Email Response of Voicemail (if called but no voicemail, leave empty)',
       'Outcome Recode (A = Appointment, C = Call offer, Q = Screening questions, W = Waitlist, R = Referral, X = Rejection, NV = no voice message, N = No response, I = Inconclusive)\n',
       'Secondary Outcome Recode (A = Appointment, C = Call offer, Q = Screening questions, W = Waitlist, R = Referral, X = Rejection, NV = no voice message, N = No response, I = Inconclusive)',
       ]]

# Rename

d.rename(
         columns = {
                    'Response (Y/N)': 'response',
                    'Email Pair ID#': 'email_pair_id',
                    'Within Patient ID': 'client_id',
                    'First Email in Pair? (Y/N)': 'first_email',
                    'Email, Phone, or Text (E/P/T)': 'ept',
                    'Offered a pre-appt consultation/to talk on the phone? (Y/N)': 'consult',
                    'Implied or explicit appointment offer? (Y/N)': 'appointment',
                    'Rejection? (Y/N)': 'reject',
                    'Asked about insurance/payment? (Y/N)' : 'tnb_ask',
                    'Asked about trans/nonbinary issues? (Y/N)' : 'ins_ask',
                    'Copy-and-Paste Email Response of Voicemail (if called but no voicemail, leave empty)': 'text',
                    'Outcome Recode (A = Appointment, C = Call offer, Q = Screening questions, W = Waitlist, R = Referral, X = Rejection, NV = no voice message, N = No response, I = Inconclusive)\n': 'prmr_outcome',
                    'Secondary Outcome Recode (A = Appointment, C = Call offer, Q = Screening questions, W = Waitlist, R = Referral, X = Rejection, NV = no voice message, N = No response, I = Inconclusive)': 'scnd_outcome',
                    }, inplace = True,
        )

# Restrict: response = 1

d = d[d['response'] == 1.0]

# Encode 'Y/N' string

le = LabelEncoder()

outcomes = [
            'consult',
            'appointment',
            'reject',
            'tnb_ask',
            'ins_ask',
            ]

for outcome in outcomes:
    if d[outcome].isnull().any():
        d[outcome].fillna(
                          'unknown',
                          inplace = True,
                          )

    d[outcome] = le.fit_transform(d[outcome])
    d[outcome].replace({
                        2: 1,
                        1: 0,
                            }, inplace = True,
                       )

d.shape
d.dtypes
d.head(3)

"""**_Preprocess_**"""

# Define spaCy NE redaction Fx

nlp = spacy.load('en_core_web_lg')

def redact_ne(mhp_text):
    '''replaces named entities with <|PII|>'''
    ne = list(
              [
               'PERSON',   ### people, including fictional
               'NORP',     ### nationalities or religious or political groups
               'FAC',      ### buildings, airports, highways, bridges, etc.
               'ORG',      ### companies, agencies, institutions, etc.
               'GPE',     ### countries, cities, states
               'LOC',      ### non-GPE locations, mountain ranges, bodies of water
               'PRODUCT',  ### objects, vehicles, foods, etc. (not services)
               'EVENT',    ### named hurricanes, battles, wars, sports events, etc.
               ]
                )

    doc = nlp(mhp_text)
    ne_to_remove = []
    final_string = str(mhp_text)
    for sent in doc.ents:
        if sent.label_ in ne:
            ne_to_remove.append(str(sent.text))
    for n in range(len(ne_to_remove)):
        final_string = final_string.replace(
                                            ne_to_remove[n],
                                            '<|PII|>',
                                            )
    return final_string

# NER redaction

d['text'] = d['text'].astype(str).apply(lambda i: redact_ne(i))

# Excise numerals

d['text'] = d['text'].apply(lambda i: re.sub(
                                             r'\d+',
                                             ' ',
                                             i)
                            )

# Harmonize manual redactions

redactions = [
              '[PATIENT NAME]',
              '[MHP NAME]',
              '[CITY]',
              ]

d['text'] = d['text'].astype(str).apply(lambda i: reduce(
                                                         lambda s, r: s.replace(
                                                                                r,
                                                                                '<|PII|>',
                                                                                ), redactions, i
                                                         )
                                        )


                           )

d['text'] = d['text'].astype(str).apply(lambda i: i.replace(
                                                            '\n',
                                                            ' ',
                                                            )
                                        )

d.head(3)

d.to_excel('d_anon.xlsx')











"""### Post-annotation
***
"""

# Replace NaN

targets = [
           'prob', ### prob = _not_ coherent target; exploratory-qualitative only
           'refl',
           'just',
           'afrm',
           'fit',
           'agnt',
           ]

for target in targets:
    d[target].fillna(
                     0,
                     inplace = True,
                     )

d.head(3)

# Encode audit study outcomes

        ### _note_: exploratory

le = LabelEncoder()

outcomes = [
            'consult',
            'appointment',
            'reject',
            ]


for outcome in outcomes:
    if d[outcome].isnull().any():
        d[outcome].fillna('unknown', inplace = True) ### NaNs recoded

    d[outcome] = le.fit_transform(d[outcome])
    d[outcome].replace({
                        2: 1,
                        1: 0,
                            }, inplace = True)

d.head(3)

# Drop pilot preprocessing

        ### _note_ 3/26: drop exploratorily preprocessed data from pilot runs

d.drop(
       'text_response_pre',
       axis = 1,
       inplace = True,
       )

d.columns

# Drop annotation artifacts

artifacts = [
             '<PII>', ### <PII> _only_ in non-augmented data
             #'<PROB>',
             #'<JUST>',
             #'<AFRM>',
             #'<FIT>',
             #'<AGNT>',
             #'<REFL>',
             ]

for artifact in artifacts:
    d['text_response_anon'] = d['text_response_anon'].str.replace(artifact, ' ', regex = True)

        ### SJS 3/24: artifacts = perfect 1:1 x_pred; must excise in annotated data

d.to_csv('d_inspect.csv', index = False)

# Verify

d.head(3)

# Preprocess

# Convert to lowercase

t_col = d['text_response_anon'].astype(str).apply(lambda i: i.lower())

# Expand contractions

t_col = t_col.apply(lambda i: ' '.join([contractions.fix(expanded_word) for expanded_word in i.split()]))

# Excise numbers

t_col = t_col.apply(lambda i: re.sub(r'\d+', ' ', i))

# Excise punctuation

t_col = t_col.apply(lambda i: re.sub('[%s]' % re.escape(string.punctuation), ' ' , i))

# Convert diacriticals

t_col = t_col.apply(lambda i: unidecode(i, errors = 'preserve'))

        ### _note_ 10/5: errors = 'preserve': retain if no replacement character possible

# Standardize/correct spelling

t_col = t_col.apply(lambda i: str(TextBlob(i).correct()))

        ### SJS 3/19: ~5 min runtime...

# Update stoplist

sw_nltk = stopwords.words('english')
sw_add = [
          'um',
          ]

sw_nltk.extend(sw_add)

# Apply updated stoplist

d['text_response_pre'] = t_col.apply(lambda i: ' '.join([ word for word in i.split() if word not in sw_nltk]))

# Inspect

d.head(3)
d.to_csv('d_inspect.csv', index = False)

# Contextualize artifacts from pilot runs

d_inspect = pd.read_csv(
                        'd_inspect.csv',
                        encoding = 'unicode_escape',
                        header = 0,
                        )


# Unstandardized text: 'um'

d_inspect['text_response_pre'] = d_inspect['text_response_pre'].apply(lambda i: nltk.word_tokenize(str(i)) if i is not None else [])
nltk_t = nltk.Text(word for tokens in d['text_response_pre'] for word in tokens)

print(nltk_t.concordance('um', width = 120, lines = 100))

# Count vectorize

        ### _note_ 3/7: accomodating feature importance downstream...

#from sklearn.feature_extraction.text import CountVectorizer

#cv = CountVectorizer(
#                     ngram_range = (1, 3),
#                     min_df = 0.01, ### ignore tokens that appear in <1% of responses
                     #stop_words = 'english',
#                     )

#cv_matrix = cv.fit_transform(d['text_response_pre'].values.astype('U'))

#cv_d = pd.DataFrame(cv_matrix.toarray(), columns = cv.get_feature_names_out())
#cv_d.shape

#cv_features = pd.concat([d, cv_d], axis = 1) ### f = features

#cv_features.head(3)

        ### _note_ 3/7: export, c/b of use

#cv_features.to_csv('mh_document-feature_matrix.csv')

#tfidf vectorize

tv = TfidfVectorizer(
                     ngram_range = (1, 3),
                     min_df = 0.01, ### ignore tokens that appear in <1% of responses
                     #stop_words = 'english',
                     )

tf_matrix = tv.fit_transform(d['text_response_pre'].values.astype('U'))

tf_d = pd.DataFrame(tf_matrix.toarray(), columns = tv.get_feature_names_out())
tf_d.shape


tf_features = pd.concat([d, tf_d], axis = 1)

tf_features.head(3)

#tf_features.to_csv('mh_document-feature_matrix.csv')

# Value counts

d[[
   'refl',
   'just',
   'afrm',
   'fit',
   'agnt',
   ]].apply(pd.Series.value_counts)

        ### _note_ 3/26: computed for '_QUAL' (non-augmented) df

# Weights

'refl: p_w'
674 / 81 # p_w = 8.3210

'just: p_w'
709 / 46 # p_w = 15.4130

'afrm: p_w'
709 / 46 # p_w = 15.4130

'fit: p_w'
699 / 56 # p_w = 12.4821

'agnt: p_w'
715 / 40 # p_w = 17.8750

# Dummy code augmented rows

d['augment'] = 0

t_indices = d['rationale'].apply(lambda i: isinstance(i, str))

d.loc[t_indices.shift(1, fill_value = False), 'augment'] = 1

        ### _note_ 3/21: use for 'drop if' prior to eval to avoid inflated metrics - annotated data _only_

d.head(10)
#d.to_csv('d_inspect.csv', index = False)

"""### XGBoost

**rskf train-test-feature loop: _not_ augmented**
"""

targets = [
           'refl',
           'just',
           'afrm',
           'fit',
           'agnt',
           ]

# Scale pos weights: n neg / n pos

p_w = {
       'refl': 8.3210,
       'just': 15.4130,
       'afrm': 15.4130,
       'fit':  12.4821,
       'agnt': 17.8750,
       }


for target in targets:
    '''Preprocesses, trains, tests (5-fold CV), and ranks feature importance of mental health professional (MHP)
    response quality targets while replying to appointment queries, using binary XGBoost classifiers.'''

    X = d['text_response_pre'].astype(str)
    y = d[target]

    # Vectorize

    #cv = CountVectorizer(
                         #binary = True,
                         #ngram_range = (1, 3),
                         #min_df = 0.01,
                         #stop_words = 'english',
                         #)

    tv = TfidfVectorizer(
                         ngram_range = (1, 3),
                         min_df = 0.01,
                         #stop_words = 'english',
                         )

    #X_count = cv.fit_transform(X)
    X_tfidf = tv.fit_transform(X)

    # Convert sparse matrix to dense array

    #X_count = X_count.toarray()
    X_tfidf = X_tfidf.toarray()

    X_train, X_test, y_train, y_test = train_test_split(
                                                        #X_count,
                                                        X_tfidf,
                                                        y,
                                                        test_size = 0.2,
                                                        stratify = y,
                                                        random_state = 56,
                                                        )

    # Scale

    s = StandardScaler()

        ### _note_ 3/7: _fit_ on training set only

    X_train_scaled = s.fit_transform(X_train)
    X_test_scaled = s.transform(X_test)

    # Classifier params

    xgb = XGBClassifier(
                        scale_pos_weight = p_w[target],
                        #verbosity = 0,
                        )

    # Cross validate

    n_splits = 5
    n_repeats = 5
    rskf = RepeatedStratifiedKFold(
                                   n_splits = n_splits,
                                   n_repeats = n_repeats,
                                   random_state = 56,
                                   )

    # Lists to store eval metrics

    f1_macro = []
    auprc = []
    mcc = []

    # Iterate over k-fold splits

    for train_index, val_index in rskf.split(X_train_scaled, y_train):
        X_train_cv, X_val_cv = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train

        xgb.fit(X_train_cv, y_train_cv)

        # Predict on validation set

        y_pred_val = xgb.predict(X_val_cv)

        # F1 macro, AUPRC, MCC eval

        f1_macro.append(f1_score(y_val_cv, y_pred_val, average = 'macro'))
        auprc.append(average_precision_score(y_val_cv, xgb.predict_proba(X_val_cv)[:, 1], average = 'macro'))
        mcc.append(matthews_corrcoef(y_val_cv, y_pred_val))

    print('----------------------------------------------------------------------------------------')
    print(f'\nTarget: {target}')
    print(f'Average F1 Macro: {sum(f1_macro) / len(f1_macro)}')
    print(f'Average AUPRC: {sum(auprc) / len(auprc)}')
    print(f'Average MCC: {sum(mcc) / len(mcc)}')

    # Plot feature importances

    #print('----------------------------------------------------------------------------------------')
    #plt.figure(figsize = (20, 260))
    #sorted_idx = xgb.feature_importances_.argsort()

        ### SJS 3/7: _NOTE_'tf_d' assigned in cell above

    #plt.barh(tf_d.columns[sorted_idx], xgb.feature_importances_[sorted_idx], color = 'skyblue')
    #plt.xlabel("XGBoost Feature Importance")

    #plt.title(f'Feature Importance - {target}')
    #plt.show()

    # List feature importances

    print('----------------------------------------------------------------------------------------')
    feature_names = tv.get_feature_names()
    importance_scores = xgb.feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})
    feature_importance_df = feature_importance_df.sort_values(by = 'Importance', ascending=False)

    top_20_features = feature_importance_df.head(20)
    print(f'\nTop 20 features for {target}:')
    print(top_20_features)

"""### BERT"""

os.chdir('<my_dir>')
#%pwd

# Import

        ### _note_ 3/7: mh_audit_wave1_QUAL = single-annotated, n = 755

        ### _note_ 3/21: mh_audit_wave1_QUAL_aug = single-annotated, _manually_ augmented w/ rationales, n = 990

d = pd.read_csv(
                'mh_audit_wave1_QUAL_aug.csv',
                encoding = 'unicode_escape',
                header = 0,
                )

# Replace NaN

targets = [
           'prob', ### prob = _not_ coherent target; expl only
           'refl',
           'just',
           'afrm',
           'fit',
           'agnt',
           ]

for target in targets:
    d[target].fillna(0, inplace = True)

# Dummy code augmented rows

d['augment'] = 0

t_indices = d['rationale'].apply(lambda i: isinstance(i, str))

d.loc[t_indices.shift(1, fill_value = False), 'augment'] = 1

# Drop annotation artifacts

artifacts = [
             '<PII>',
             '<PROB>',
             '<JUST>',
             '<AFRM>',
             '<FIT>',
             '<AGNT>',
             '<REFL>',
             ]

for artifact in artifacts:
    d['text_response_aug'] = d['text_response_aug'].str.replace(artifact, ' ', regex = True)

d.head(10)

# SimpleTransformer prep

#d.head(3)
#d.columns
d['text'] = d['text_response_aug'].str.replace(r'<PII>', ' ', regex = True)
d.drop(
       columns = [
                  'id',
                  'response',
                  'emailpair',
                  'pid',
                  'firstemail',
                  'ept',
                  'consult',
                  'appointment',
                  'reject',
                  'text_response_anon',
                  'prob',
                  'rationale',
                  'notes',
                  'prmr_outcome',
                  'scnd_outcome',
                  #'text_response_pre',
                  ], inplace = True)

st_format = [
             'text',
             'augment',
             'refl',
             'just',
             'afrm',
             'fit',
             'agnt',
             ]

d = d[st_format]
d.head(3)

d.shape

# Class weights: N / n classes * n

c_w = {
       'refl': {
                '0': 0.5978,
                '1': 3.0556,
                },
       'just': {
                '0': 0.5525,
                '1': 5.2660,
                },
       'afrm': {
                '0': 0.5518,
                '1': 5.3226,
                },
       'fit': {
                '0': 0.5644,
                '1': 4.3805,
                },
       'agnt': {
                '0': 0.5464,
                '1': 5.8929,
                }
       }

# View defaults

'BERT defaults'
bert_uncased = ClassificationModel(
                                   'bert',
                                   'bert-large-uncased',
                                   use_cuda = False,
                                   )
bert_uncased.args

'RoBERTa defaults'
roberta_base = ClassificationModel(
                                    'roberta',
                                    'roberta-base',
                                    use_cuda = False,
                                    )
roberta_base.args

'DistilBERT defaults'
distilbert_base = ClassificationModel(
                                      'distilbert',
                                      'distilbert-base-cased',
                                      use_cuda = False,
                                      )
distilbert_base.args

"""**kf train-test loop: _augmented_**"""

# Fx iterates over models, args, subconstructs, class weights

def train_and_evaluate_model(model_type, model_name, target, class_weights, train_d, eval_d):
    '''Fine tunes and evaluates (5-fold CV) rationale-augmented MHP response quality targets across BERT, RoBERTa, and
    DistilBERT pretrained LMs.'''

    print('======================================================================================')
    print(f"\nTraining {model_type}:{model_name}, target **{target}**")
    print('======================================================================================')

    # Args

    model_args = ClassificationArgs(
                                    num_train_epochs = 2,
                                    sliding_window = False, ### ~2x runtime w/ True; 'jagged arrays' incompatible w/ loop
                                    #do_lower_case = True,
                                    #reprocess_input_data = False,
                                    overwrite_output_dir = True,
                                    manual_seed = 56,
                                    )

    # Initialize model(s)

    model = ClassificationModel(
                                model_type = model_type,
                                model_name = model_name,
                                use_cuda = False,
                                num_labels = 2,
                                weight = class_weights[target],
                                args = model_args,
                                )

    # Train

    model.train_model(train_d)

    print(f"Evaluating {model_type} model for {target}")

    # Eval

    result, model_outputs, _ = model.eval_model(eval_d)

    # Metric

    predictions = np.argmax(model_outputs, axis = 1)
    f1_macro = f1_score(eval_d[target], predictions, average = 'macro')

    print('----------------------------------------------------------------------------------------')
    print(f"F1 macro: {model_type} for {target} = {f1_macro}")
    print("Evaluation:")
    print(result)

# Model types, model names

model_types = [
               'bert',
               'roberta',
               'distilbert',
                ]

model_names = [
               'bert-base-cased',
               'roberta-base',
               'distilbert-base-cased',
                ]

# Class weights

class_weights = {
                 'refl': [
                          0.5978,
                          3.0556,
                          ],
                 'just': [
                          0.5525,
                          5.2660,
                          ],
                 'afrm': [
                          0.5518,
                          5.3226,
                          ],
                 'fit': [
                         0.5644,
                         4.3805,
                         ],
                 'agnt': [
                          0.5464,
                          5.8929,
                          ]
                 }

# Iterate over models

for model_type, model_name in zip(model_types, model_names):

    # Iterate over subconstructs

    for target in [
                   'refl',
                   'just',
                   'afrm',
                   'fit',
                   'agnt',
                   ]:

        # Stratified train-test split

#        train_df, eval_df = train_test_split(
#                                             trans_df[['text', target]],
#                                             test_size = 0.2,
#                                             stratify = trans_df[target],
#                                             random_state = 56,
#                                             )

        # Subconstruct-stratified 5-fold cv

        d_subset = d[['text', target, 'augment']]

        kf = KFold(
                   n_splits = 5,
                   shuffle = True,
                   random_state = 56,
                   )

        for train_index, eval_index in kf.split(d_subset):
            train_d, eval_d = d_subset.iloc[train_index], d_subset.iloc[eval_index]

            eval_d = eval_d[eval_d['augment'] != 1]

            train_and_evaluate_model(
                                     model_type,
                                     model_name,
                                     target,
                                     class_weights,
                                     train_d,
                                     eval_d,
                                     )

"""> End of mh_audit_v1.ipynb (03-28-2024)"""
