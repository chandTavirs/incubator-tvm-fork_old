import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


max_pool_configs = [
    # "(3,3,0,0,2,2,1)",
    # "(3,3,1,1,2,2,0)",
    # "(3,3,1,1,1,1,1)",
    "(2,2,0,0,2,2,1)"
]

def plot_df(df, x, y, title="", xlabel='Sample', ylabel='Read B/W', dpi=100):
    plt.figure(figsize=(16, 5), dpi=dpi)
    plt.plot(x, y, color='tab:purple')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

def generate_boxplot(file_data):

    df = pd.DataFrame()
    for item in file_data:
        df0 = pd.DataFrame.from_dict(item["results"]['2']["samples"])
        df0 = df0.drop(columns=["read_bytes","write_bytes","write_bw"])
        df0['workload'] = item["workload_str"]
        df = df.append(df0)

    # boxplot = df.boxplot(column='read_bw', by='workload_str')
    # plt.show()
    df_long = pd.melt(df, "workload", var_name="Read Bandwidth", value_name="Value (MB/s)")
    sns.factorplot("Read Bandwidth", hue="workload", y="Value (MB/s)", data=df_long, kind="box")
    plt.show()


def plot_bytes(file_data, slot, rOrW, plot_diff=False):

    data_list = []
    if plot_diff:
        diff_or_overall = "diff"
    else:
        diff_or_overall = "overall"

    for item in file_data:
        if rOrW == 'read':
            data_list.append({"workload": item["workload_str"], "read_bytes": item["results"][slot][diff_or_overall]["read_bytes"]})
        else:
            data_list.append(
                {"workload": item["workload_str"], "write_bytes": item["results"][slot][diff_or_overall]["write_bytes"]})

    df = pd.DataFrame.from_dict(data_list)

    fig, ax = plt.subplots()

    if rOrW == 'read':
        df.plot('workload', "read_bytes", kind='line', ax=ax)
    else:
        df.plot('workload', "write_bytes", kind='line', ax=ax)

    for k, v in df.iterrows():
        if rOrW == 'read':
            ax.annotate(v.read_bytes, [k, v.read_bytes])
        else:
            ax.annotate(v.write_bytes, [k, v.write_bytes])

    plt.show()

def time_series(file_data, slot, rOrW, bOrBw):
    if rOrW == 'read':
        if bOrBw == 'bytes':
            flag_data = "read_bytes"
        else:
            flag_data = "read_bw"
    else:
        if bOrBw == 'bytes':
            flag_data = "write_bytes"
        else:
            flag_data = "write_bw"

    for config in max_pool_configs:
        overall_bytes = []
        trials = file_data["results"][config]
        for trial in trials:
            overall_bytes.append(trial[slot]["overall"]["write_bytes"])

        index_max = max(range(len(overall_bytes)), key=overall_bytes.__getitem__)

        data_list = []
        data = file_data["results"][config][index_max][slot]["samples"]

        for i, sample in enumerate(data):
            data_list.append({"sample_number": i, flag_data: sample[flag_data]})


        df = pd.DataFrame.from_dict(data_list)

        fig, ax = plt.subplots()

        df.plot('sample_number', flag_data, kind='line', ax=ax)
        #plt.fill_between(df['sample_number'], y1=df[flag_data], y2=-df[flag_data], alpha=0.5, linewidth=2, color='seagreen')
        plt.title("Conv{} + MaxPool{}".format(file_data["workload_str"],config))

        plt.show()

def time_series_simul(file_data, bOrBw):
    if bOrBw == 'bytes':
        flag_read = "read_bytes"
        flag_write = "write_bytes"
    else:
        flag_read = "read_bw"
        flag_write = "write_bw"

    overall_bytes = []
    for trial in file_data:
        overall_bytes.append(trial['5']["overall"]["write_bytes"])

    index_max = max(range(len(overall_bytes)), key=overall_bytes.__getitem__)

    read_list = []
    write_list = []
    data_read = file_data[index_max]['2']["samples"]
    data_write = file_data[index_max]['5']["samples"]

    for i, sample in enumerate(data_read):
        read_list.append({"sample_number": i, flag_read: sample[flag_read]})
        write_list.append({"sample_number": i, flag_write: sample[flag_write]})

    for i, sample in enumerate(data_write):
        write_list.append({"sample_number": i, flag_write: sample[flag_write]})


    df_read = pd.DataFrame.from_dict(read_list)
    df_write = pd.DataFrame.from_dict(write_list)


    fig, ax = plt.subplots()


    df_read.plot('sample_number', flag_read, kind='line', ax=ax)
    df_write.plot('sample_number', flag_write, kind='line', ax=ax)


    # for k, v in df_bytes.iterrows():
    #     if rOrW == 'read':
    #         ax.annotate(v.read_bytes, [k, v.read_bytes])
    #     else:
    #         ax.annotate(v.write_bytes, [k, v.write_bytes])
    # for k, v in df_bw.iterrows():
    #     if rOrW == 'read':
    #         ax.annotate(v.read_bw, [k, v.read_bw])
    #     else:
    #         ax.annotate(v.write_bw, [k, v.write_bw])

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AXI Performance Monitor Convolutions Data visualization')
    parser.add_argument('--log_file', type=str, default="logs/log.json",
                        help='input log file path')
    parser.add_argument('--workloads', type=str, default="0,1",
                        help='workloads to compare and plot')
    parser.add_argument('--boxplot', action='store_true',
                        help='generate box plots of read b/w of workloads')
    parser.add_argument('--plot_write_bytes', action='store_true',
                        help='generate plot of write bytes of workloads')
    parser.add_argument('--plot_read_bytes', action='store_true',
                        help='generate plot of read bytes of workloads')
    parser.add_argument('--plot_read_bytes_uop', action='store_true',
                        help='generate plot of read bytes from compute uop of workloads')
    parser.add_argument('--plot_read_bytes_data', action='store_true',
                        help='generate plot of read bytes from compute data of workloads')
    parser.add_argument('--plot_read_bytes_fetch', action='store_true',
                        help='generate plot of read bytes from fetch data of workloads')
    parser.add_argument('--plot_read_bytes_overall', action='store_true',
                        help='generate plot of read bytes from overall data of workloads')
    parser.add_argument('--plot_write_bytes_overall', action='store_true',
                        help='generate plot of write bytes from overall data of workloads')
    parser.add_argument('--plot_diff', action='store_true',
                        help='plot diff instead of overall' )
    parser.add_argument('--plot_model_time_series', action='store_true',
                        help='plot model profiling data as time series')
    parser.add_argument('--plot_model_time_series_simul', action='store_true',
                        help='plot model profiling read and write data as time series')
    args = parser.parse_args()

    log_file = args.log_file
    workloads = [int(workload) for workload in args.workloads.split(',')]

    data = []
    with open(log_file, 'r') as myfile:
        file_data = json.load(myfile)
        if args.plot_model_time_series or args.plot_model_time_series_simul:
            data = file_data["workloads"][workloads[0]]
        else:
            for wkl_idx in workloads:
                data.append(file_data["workloads"][wkl_idx])

    if args.boxplot:
        generate_boxplot(data)

    if args.plot_write_bytes:
        plot_bytes(data, '5', 'write', args.plot_diff)

    if args.plot_write_bytes_overall:
        plot_bytes(data, '0', 'write', args.plot_diff)

    if args.plot_read_bytes:
        plot_bytes(data, '2', 'read', args.plot_diff)

    if args.plot_read_bytes_overall:
        plot_bytes(data, '0', 'read', args.plot_diff)

    if args.plot_read_bytes_uop:
        plot_bytes(data,'3','read', args.plot_diff)

    if args.plot_read_bytes_data:
        plot_bytes(data,'4','read', args.plot_diff)

    if args.plot_read_bytes_fetch:
        plot_bytes(data, '1', 'read', args.plot_diff)

    if args.plot_model_time_series:
        time_series(data, '1', 'read','bytes')

    if args.plot_model_time_series_simul:
        time_series_simul(data, 'bytes')
