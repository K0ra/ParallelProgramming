import os
import pandas as pd
from plot_data import PlotData

# TODO: consider typing

# Change to your src folder
src_folder = '/Users/mbair13/Downloads/ParallelProcessing-lab5/src'

# Remove .csv
def remove_extension(file_name):
  return file_name[:-4]

# Get dataframe from file
def get_df(file_name='seq.csv', file_dir='measurements', sep=',', names=[]):
  script_dir = os.path.dirname(src_folder)
  results_dir = os.path.join(script_dir, file_dir)
  skiprows = 1

  if not names:
    col_id = remove_extension(file_name)
    names = ['N', f'{col_id}-delta']
    skiprows = 0

  try:
    df = pd.read_csv(results_dir + '/' + file_name, skiprows=skiprows, sep=sep, names=names)
    return df
  except Exception as e:
    # If results_dir doesn't exist print error
    print(e)

# Calculate parallel accelerations
def get_accelerations_table(table: pd.DataFrame) -> pd.DataFrame:
  table['div_col'] = table['lab1-gcc-seq-delta'] # temporary column
  for col in table.columns:
    table[col] = table['div_col'].div(table[col])
  table.drop('div_col', axis=1, inplace=True)

  return table

def get_time_plot(df_lab5: pd.DataFrame, df_lab4: pd.DataFrame, num_thread: int):
    df_time_5 = df_lab5.loc[:, ['N', 'min']]
    df_time_4 = df_lab4.loc[:, ['N', 'min']]
    df_time_5['min_4'] = df_time_4['min']
    df_time_5.rename(columns={'min': "delta_ms_lab5", "min_4": "delta_ms_lab4"}, inplace=True)
    df_time_5.set_index('N', inplace=True)

    title = f'Time complexity lab5 and lab4 (num threads={num_thread})'
    xlabel = 'N'
    ylabel = 'Time (ms)'
    bar_plot = PlotData(df_time_5, title, xlabel, ylabel)
    bar_plot.plot('bar', 20, 10)
    bar_plot.save_figure(file_dir=src_folder + '/../tasks/task3.1',
                        filename=f'Time_complexity_threads_{num_thread}')


def get_accelerations_plot(df_lab5: pd.DataFrame, df_lab4: pd.DataFrame, num_thread: int):
    df_acc_5 = df_lab5.loc[:, ['N', 'min', 'lab1-gcc-seq-delta']]
    df_acc_4 = df_lab4.loc[:, ['N', 'min', 'lab1-gcc-seq-delta']]
    df_acc_5['min_4'] = df_acc_4['min']
    df_acc_5.set_index('N', inplace=True)
    get_accelerations_table(df_acc_5)
    df_acc_5.rename(columns={'min': "acceleration_lab5", "min_4": "acceleration_lab4"}, inplace=True)

    title = f'Accelelrations lab5 and lab4 (num threads={num_thread})'
    xlabel = 'N'
    ylabel = 'Acceleration'

    bar_plot = PlotData(df_acc_5, title, xlabel, ylabel)
    bar_plot.plot('bar', 20, 10)
    bar_plot.save_figure(file_dir=src_folder + '/../tasks/task3.2',
                        filename=f'Compare_accelerations_threads_{num_thread}')


def get_compare_exec_time_plot(df_exec_5: pd.DataFrame, df_exec_4: pd.DataFrame, num_thread: int):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path
    sns.set(rc={'figure.figsize':(15, 8)})

    def plot_compare(df_exec_total: pd.DataFrame, colname: str, num_thread: int):
        ax = sns.histplot(df_exec_total, x='N', hue='lab#',
                    weights=colname, multiple='stack') #, ax=ax
        
        ax.set_title(f'{colname} comparison by N between lab4 and lab5 (threads_{num_thread})')
        ax.set_ylabel(colname)
        # Fix the legend so it's not on top of the bars.
        legend = ax.get_legend()
        legend.set_bbox_to_anchor((1, 1))
        results_dir = src_folder + f'/../tasks/task3.3/compare/thread_{num_thread}'
        path = Path(results_dir)
        path.mkdir(parents=True, exist_ok=True)

        plt.savefig(results_dir + '/' + f'Compare_exec_time_{colname}_threads_{num_thread}')
        plt.close()

    # Combine dataframes from lab4 and lab5 adding an identifier column "lab#"
    df_exec_total = pd.concat([df_exec_5, df_exec_4], axis=0, ignore_index=False)
    df_exec_total['lab#'] = (len(df_exec_5)*(5,) + len(df_exec_4)*(4,))
    df_exec_total.reset_index(inplace=True)

    colnames = ['Generation', 'Map', 'Merge', 'Sort', 'Reduce']
    for colname in colnames:
        plot_compare(df_exec_total, colname, num_thread)

def get_exec_time_plot(df_lab5: pd.DataFrame, df_lab4: pd.DataFrame, num_thread: int):
    import matplotlib.pyplot as plt
    from pathlib import Path

    def plot_exec_time(df_exec: pd.DataFrame, labname: str, num_thread: int):
        df_exec.plot(x='N', kind='bar', stacked=True,
                title=f'{labname}: Execution steps by N (threads={num_thread})', figsize=(20, 15))
        plt.ylabel("Time (ms)")
        results_dir = src_folder + f'/../tasks/task3.3/separate/thread_{num_thread}'
        path = Path(results_dir)
        path.mkdir(parents=True, exist_ok=True)

        plt.savefig(results_dir + '/' + f'{labname}_exec_time_threads_{num_thread}')
        plt.close()
    
    names = ['N', 'Generation', 'Map', 'Merge', 'Sort', 'Reduce']
    df_exec_5 = df_lab5.loc[:, names]
    df_exec_4 = df_lab4.loc[:, names]

    plot_exec_time(df_exec_5, 'lab5', num_thread)
    plot_exec_time(df_exec_4, 'lab4', num_thread)
    get_compare_exec_time_plot(df_exec_5, df_exec_4, num_thread)

def main():
    n_threads = [1, 2, 3, 4, 6, 8]
    names = ['N', 'min', 'Generation', 'Map', 'Merge', 'Sort', 'Reduce']

    for n in n_threads:
        df_lab5 = get_df(file_name=f'lab5-{n}.csv', names=names)
        df_lab4 = get_df(file_name=f'lab5_4-{n}.csv', names=names)
        df1 = get_df('lab1-gcc-seq.csv', sep=';')
        df2 = get_df('lab1-gcc-seq.csv', sep=';')
        df_lab5['lab1-gcc-seq-delta'] = df1['lab1-gcc-seq-delta']
        df_lab4['lab1-gcc-seq-delta'] = df2['lab1-gcc-seq-delta']

        get_time_plot(df_lab5, df_lab4, n)
        get_accelerations_plot(df_lab5, df_lab4, n)
        get_exec_time_plot(df_lab5, df_lab4, n)


if __name__ == "__main__":
    main()


