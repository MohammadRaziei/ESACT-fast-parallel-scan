#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 02:10:33 2023

@author: mohammad
"""

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

    
#
# with open("m.log", "r") as f:
#     parsed_data = parse_log_data(f.read())

kernel_time_data = []
total_time_data = []
cpu_time_data = []
for entry in parsed_data:
    m = entry["m"]
    gpu_methods = entry["gpu_methods"]
    cpu_time_data.append({"m": m, "cpu_time_ms": entry["cpu_time_ms"]})
    for method_data in gpu_methods:
        total_time_data.append({"m": m, "methods": method_data["method"], "total_time_ms": method_data["total_time_ms"]})
        kernel_time_data.append({"m": m, "methods": method_data["method"], "kernel_time_ms": method_data["kernel_time_ms"]})


# Create pandas DataFrames
col_orders = ["BLELOCHE", "PADDED", "SACT1", "SACT2", "ESACT"]
df_total_time_ms = pd.DataFrame(total_time_data).pivot(index='m', columns='methods', values='total_time_ms')[col_orders]
df_kernel_time_ms = pd.DataFrame(kernel_time_data).pivot(index='m', columns='methods', values='kernel_time_ms')[col_orders]
df_cpu_time_ms = pd.DataFrame(cpu_time_data).set_index('m')


df_kernel_time_ms = df_kernel_time_ms.applymap(np.median)

print(df_kernel_time_ms)


df = df_kernel_time_ms.apply(lambda x: (x['BLELOCHE'] - x)/x['BLELOCHE'], axis=1).drop('BLELOCHE', axis=1)
print(df.mean())

df = df_kernel_time_ms.divide(df_kernel_time_ms["BLELOCHE"], axis=0)
# df["PADDED"] = (.25*df["SACT1"].cummax() + .45*df["ESACT"] + .3 * df["PADDED"])
# df_kernel_time_ms["PADDED"] = df["PADDED"] * df_kernel_time_ms["BLELOCHE"]

df_kernel_time_ms.plot(logy=True)
plt.savefig((Path(__file__).parent.parent / "results" / "multiblocks_analysis_results1.pdf").as_posix(), 
            format="pdf", bbox_inches="tight")
# df_cpu_time_ms.plot(logy=True)
plt.show()


df.plot()
# plt.show()
plt.savefig((Path(__file__).parent.parent / "results" / "multiblocks_analysis_results2.pdf").as_posix(), 
            format="pdf", bbox_inches="tight")
plt.show()


# for i, method in enumerate(col_orders):
#     color = colors[i]
#     df = pd.DataFrame(df_kernel_time_ms[method]).T.reset_index(drop=True)
#     df = pd.DataFrame(np.array(df.values.tolist()).squeeze().T, columns=df.columns.values)
#     df.boxplot(grid=False, 
#                 color=dict(boxes=color, whiskers=color, medians=color, caps=color),
#                 )
