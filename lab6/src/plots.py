from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.stats as st

# Change to your src directory
src_dir = '/Users/mbair13/Downloads/ParallelProgramming/lab6/src'

""" Read DataFrame from .csv file    """
def get_df(file_name='seq.csv', file_dir='measurements', 
            sep=',', names:[]=None) -> pd.DataFrame:
    script_dir = os.path.dirname(src_dir)
    results_dir = os.path.join(script_dir, file_dir)

    df = pd.read_csv(results_dir + '/' + file_name, sep=sep, names=names)
    return df

def save_figure(results_dir: str, file_name: str):
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)

    plt.savefig(results_dir + '/' + file_name, bbox_inches='tight')
    plt.close()

""" Calculate parallel accelerations    """
def get_accelerations_table(table: pd.DataFrame) -> pd.DataFrame:
  table['div_col'] = table['lab1-gcc-seq-delta'] # temporary column
  for col in table.columns:
    table[col] = table['div_col'].div(table[col])
  
  table.drop('div_col', axis=1, inplace=True)

  return table

""" Returns the dataframe grouped by 10 iterations,
    aggregated by function specified in agg_param   """
def __get_agg_time_per_iter(df: pd.DataFrame, agg_param: str = 'min') -> pd.DataFrame:
    return df.groupby((df.index+10)//10)['T'].transform(agg_param)

""" Skip every 10 iter rows, which have the same values
    by taking the 'last' row out of ten """
def __skip_iter_rows(df: pd.DataFrame, agg_dict: dict) -> pd.DataFrame:
    for value in agg_dict.values():
	    assert(value == 'last')
    return df.groupby((df.index+10)//11).agg(agg_dict)

# This function is used only in plot_exec_time()
def __get_temp_df(file_name: str) -> pd.DataFrame:
    df = get_df(file_name=file_name).loc[:, 'N':'iter']

    df['min'] = __get_agg_time_per_iter(df, 'min')
    df = df.drop(['iter', 'T'], axis=1)

    df = __skip_iter_rows(df, {'N':'last','min':'last'})
    df.set_index('N', inplace=True)

    return df

""" Plot exec time for all threads to choose the best option """
def plot_exec_time(figsize:tuple=(18, 8)):
    n_threads = [10, 20, 50, 100, 200]
    files = [f'lab6-{n}.csv' for n in n_threads]
    mylabels = [f'min-{n}' for n in n_threads]

    _, ax = plt.subplots(1, 1) # initialize axes

    for idx, file in enumerate(files):
        df = __get_temp_df(file)
        if (idx == 0):
            ax = df['min'].plot(figsize=figsize)
        else:
            df['min'].plot(ax=ax)

    ax.legend(labels=mylabels)
    plt.title('Comparison of minimal execution times per thread')
    plt.xlabel('N')
    plt.ylabel('Min exec time (ms)')

    results_dir = src_dir + f'/../tasks/'
    save_figure(results_dir, 'compare_exec_times')

    # min_100 = get_temp_df('lab6-100.csv').mean()['min']
    # min_200 = get_temp_df('lab6-200.csv').mean()['min']
    # print(min_100, min_200)


def plot_interval_accelerations(figsize:tuple=(20,8)):
    """ Process lab6 results for n_thread=200 """

    df_lab6 = get_df('lab6-200.csv').loc[:, 'N':'iter']
    df_lab6['mean'] = __get_agg_time_per_iter(df_lab6, 'mean')
    df_lab6['sem'] = __get_agg_time_per_iter(df_lab6, 'sem')
    df_lab6['min'] = __get_agg_time_per_iter(df_lab6, 'min')
    df_lab6 = df_lab6.drop(['iter', 'T'], axis=1)
    df_lab6 = __skip_iter_rows(df_lab6, {'N':'last','mean':'last','sem':'last','min':'last'})
    df_lab6['low'], df_lab6['high'] = st.t.interval(alpha=0.95,
                                                    df=len(df_lab6)-1,
                                                    loc=df_lab6['mean'],
                                                    scale=df_lab6['sem'])
    df_lab6 = df_lab6.drop(['mean','sem'], axis=1)
    df_lab6['lab1-gcc-seq-delta'] = \
                    get_df(file_name='lab1-gcc-seq.csv', sep=';', names=['N', 'min'])['min']
    
    df_lab6.set_index('N', inplace=True)
    df_lab6 = get_accelerations_table(df_lab6)

    """ Process lab4 results for n_thread=200   """

    df_lab4 = get_df('lab4-4.csv')
    df_lab4['low'], df_lab4['high'] = st.t.interval(alpha=0.95,
                                                    df=len(df_lab4)-1,
                                                    loc=df_lab4.loc[:, '0':'9'].mean(axis=1),
                                                    scale=st.sem(df_lab4.loc[:, '0':'9'], axis=1))
    droprange = list(range(1, 11))  # drop columns named from '0' to '9'
    df_lab4.drop(df_lab4.columns[droprange], axis=1, inplace=True)
    df_lab4['lab1-gcc-seq-delta'] = \
                    get_df(file_name='lab1-gcc-seq.csv', sep=';', names=['N', 'min'])['min']
    df_lab4.set_index('N', inplace=True)
    df_lab4 = get_accelerations_table(df_lab4)

    """ Plot the intervals in one graph """

    ax = df_lab6.loc[:, ['min','low','high']].plot(figsize=figsize)
    df_lab4.loc[:, ['min','low','high']].plot(ax=ax)
    plt.title('Comparison of lab4 and lab6 accelerations (min & low & high)')
    plt.xlabel('N')
    plt.ylabel('Acceleration')
    ax.legend(labels=['lab6-min-200', 'lab6-low-200', 
                    'lab6-high-200', 'lab4-min-4', 'lab4-low-4', 'lab4-high-4'])
    
    results_dir = src_dir + f'/../tasks/task2.2/'
    save_figure(results_dir, 'compare_stud_intervals')


def __normalize_cols(df:pd.DataFrame, col_names:[]) -> pd.DataFrame:
    return df.loc[:, col_names].div(df.loc[:, col_names].sum(axis=1), axis=0)


def plot_exec_parts(figsize:tuple=(12, 6)):
    part_names = ['Generation', 'Map', 'Merge', 'Sort', 'Reduce']

    """ Process lab 6 data  """

    df_lab6 = get_df('lab6-200.csv')
    df_lab6['min'] = __get_agg_time_per_iter(df_lab6, 'min')
    df_lab6 = df_lab6.drop(['iter', 'T'], axis=1)
    # Arrange aggregation parameters
    agg_keys = df_lab6.columns.to_list()
    agg_vals = ['last' for i in range(len(agg_keys))]
    agg_dict = dict(zip(agg_keys, agg_vals))
    # Take such rows of part_names where indices align with df['min']
    df_lab6 = __skip_iter_rows(df_lab6, agg_dict)
    df_lab6.loc[:, part_names] = __normalize_cols(df_lab6, part_names)

    """ Process lab 4 data  """

    df_lab4 = get_df('lab5_4-4.csv')
    df_lab4.loc[:, part_names] = __normalize_cols(df_lab4, part_names)

    """ Plot the merged data    """

    lab6_col_names = [f'lab6_{col}' for col in part_names]
    lab4_col_names = [f'lab4_{col}' for col in part_names]
    lab6_col_names.extend(lab4_col_names)

    part_names.insert(0, 'N')

    df = pd.merge(df_lab6.loc[:, part_names], df_lab4.loc[:, part_names], on='N')

    title = 'Comparison of lab 4 and lab 6 execution steps by N'
    df.plot.bar(x='N', stacked=True, title=title, figsize=figsize)
    plt.ylabel("Normalized Time (ms)")
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='best', labels=lab6_col_names)

    results_dir = src_dir + f'/../tasks/task2.3/'
    save_figure(results_dir, title)

def main():
    n_threads = [10, 20, 50, 100, 200]
    
    # According to this plot, the best option is n_thread=200
    plot_exec_time(figsize=(16, 6))

    plot_interval_accelerations()

    plot_exec_parts()

    

if __name__ == "__main__":
    main()