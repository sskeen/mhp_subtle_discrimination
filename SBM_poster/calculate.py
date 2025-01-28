
import pandas as pd
from sklearn.metrics import cohen_kappa_score

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
               'afrm_dal', 'afrm_sjs',
               'agnt_dal', 'agnt_sjs',
               'brdn_dal', 'brdn_sjs',
               'dmnd_dal', 'dmnd_sjs',
               'fitt_dal', 'fitt_sjs',
               'just_dal', 'just_sjs',
               'prbl_dal', 'prbl_sjs',
               'rbnd_dal', 'rbnd_sjs',
               'refl_dal', 'refl_sjs',
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
           'afrm_dal', 'afrm_sjs',
           'agnt_dal', 'agnt_sjs',
           'brdn_dal', 'brdn_sjs',
           'dmnd_dal', 'dmnd_sjs',
           'fitt_dal', 'fitt_sjs',
           'just_dal', 'just_sjs',
           'prbl_dal', 'prbl_sjs',
           'rbnd_dal', 'rbnd_sjs',
           'refl_dal', 'refl_sjs',
           'rtnl_dal', 'rtnl_sjs',
           'note_dal', 'note_sjs',
           ]].copy()

    # kappa Fx

    def calculate_kappa(d, col_dal, col_sjs):
        return cohen_kappa_score(d[col_dal], d[col_sjs])

    col_pairs = [
                 ('afrm_dal', 'afrm_sjs'),
                 ('agnt_dal', 'agnt_sjs'),
                 ('brdn_dal', 'brdn_sjs'),
                 ('dmnd_dal', 'dmnd_sjs'),
                 ('fitt_dal', 'fitt_sjs'),
                 ('just_dal', 'just_sjs'),
                 ('prbl_dal', 'prbl_sjs'),
                 ('rbnd_dal', 'rbnd_sjs'),
                 ('refl_dal', 'refl_sjs'),
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

    # dummy code disagreements Fx

    def encode_disagreements(row):
        return 1 if row[0] != row[1] else 0

    col_dis = [
               ('afrm_dal', 'afrm_sjs', 'afrm_dis'),
               ('agnt_dal', 'agnt_sjs', 'agnt_dis'),
               ('brdn_dal', 'brdn_sjs', 'brdn_dis'),
               ('dmnd_dal', 'dmnd_sjs', 'dmnd_dis'),
               ('fitt_dal', 'fitt_sjs', 'fitt_dis'),
               ('just_dal', 'just_sjs', 'just_dis'),
               ('prbl_dal', 'prbl_sjs', 'prbl_dis'),
               ('rbnd_dal', 'rbnd_sjs', 'rbnd_dis'),
               ('refl_dal', 'refl_sjs', 'refl_dis'),
               ]

    for col1, col2, dis_col in col_dis:
        d[dis_col] = d[[col1, col2]].apply(encode_disagreements, axis = 1)

    # display counts for targets

    print("\n--------------------------------------------------------------------------------------")
    print(f"Cycle {cycle_num}: Counts by target")
    print("--------------------------------------------------------------------------------------")
    print(d[targets].apply(pd.Series.value_counts))

    # drop target cols for readability + fillna

    d = d.drop(targets, axis = 1)
    d = d.fillna('.')

    # export: cycle-specific

    d.to_excel(f'd_cycle_{cycle_num}_dis.xlsx')

    return d, kappa_results
