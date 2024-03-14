#!/usr/bin/env python3
"""
Created on Fri Oct  6 02:10:33 2023

@author: mohammad
"""

import warnings
warnings.filterwarnings("ignore")

from multiblocks_to_json import parse_from_arg, parse_log_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json



parser = argparse.ArgumentParser(description='Convert log data to JSON format.')
parser.add_argument('-i', '--input', help='Input log file')
parser.add_argument('--json', action='store_true')
args = parser.parse_args()

if args.json:
    with open(args.input, "r") as f:
        parsed_data = json.load(f)
else:
    parsed_data = parse_from_arg(args.input, parser)

    
kernel_time_data = []
total_time_data = []
cpu_time_data = []
for entry in parsed_data:
    m = entry["m"]
    gpu_methods = entry["gpu_methods"]
    # cpu_time_data.append({"m": m, "cpu_time_ms": entry["cpu_time_ms"]})
    for method_data in gpu_methods:
        total_time_data.append({"m": m, "methods": method_data["method"], "total_time_ms": method_data["total_time_ms"]})
        kernel_time_data.append({"m": m, "methods": method_data["method"], "kernel_time_ms": method_data["kernel_time_ms"]})


# Create pandas DataFrames
col_orders = ["BLELOCHE", "PADDED", "SACT1", "SACT2", "ESACT"]
df_total_time_ms = pd.DataFrame(total_time_data).pivot(index='m', columns='methods', values='total_time_ms')[col_orders]
df_kernel_time_ms = pd.DataFrame(kernel_time_data).pivot(index='m', columns='methods', values='kernel_time_ms')[col_orders]
# df_cpu_time_ms = pd.DataFrame(cpu_time_data).set_index('m')


df_kernel_time_ms = df_kernel_time_ms.applymap(np.median)
print("\n\nmedian of methods:\n==================")
print(df_kernel_time_ms)


df = df_kernel_time_ms.apply(lambda x: (x['BLELOCHE'] - x)/x['BLELOCHE'], axis=1).drop('BLELOCHE', axis=1)
print("\n\nmean of improving:\n==================")
print(df.mean())

df = df_kernel_time_ms.divide(df_kernel_time_ms["BLELOCHE"], axis=0)

df_kernel_time_ms.plot(logy=True)
plt.savefig((Path(__file__).parent.parent / "results" / "multiblocks_analysis_results_abs.pdf").as_posix(), 
            format="pdf", bbox_inches="tight")
# df_cpu_time_ms.plot(logy=True)
# plt.show()
plt.close()


df.plot(ylim=(.1, 1.3))
# plt.show()
plt.savefig((Path(__file__).parent.parent / "results" / "multiblocks_analysis_results_rel.pdf").as_posix(), 
            format="pdf", bbox_inches="tight")
# plt.show()
plt.close()

