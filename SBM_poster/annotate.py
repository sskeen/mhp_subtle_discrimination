
import pandas as pd

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
                'prbl',
                'refl',
                'just',
                'afrm',
                'fitt',
                'agnt',
                'brdn',
                'dmnd',
                'rbnd',
                'rtnl',
                'note',
                ]

    d_cycle[tag_cols] = ' '

    # excise unneeded columns

    drop_cols = [
                 'EmailPairID',
                 'WithinPatientID',
                 'FirstInPair',
                 'pilot',
                 'MHP ID#',
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
