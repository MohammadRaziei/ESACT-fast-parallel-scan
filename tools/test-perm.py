#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:05:32 2023

@author: mohammad
"""

import numpy as np
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from ttp import ttp
import matplotlib.pyplot as plt

def check_mirror(x):
    for p, q in enumerate(x):
        if x[q] != p:
            return False
    return True

# m = list(filter(check_mirror, lp))

def check_complement(x):
    n = len(x)
    for p, q in enumerate(x):
        if x[n-1-p] != n-1-q:
            return False
    return True

# n = 4
# lp = list(np.asarray(i) for i in permutations(range(n)))


# print(len(m))



# c = list(filter(check_complement, lp))
# print(len(c))

# print((c==m))


# def intersection(lst1, lst2):
#     lst3 = [value for value in lst1 if value in lst2]
#     return lst3

# mc = list(filter(lambda x: check_complement(x) and check_mirror(x), lp))

# print(len(mc))
# print(len(c))


# print("*"*32)
# for n in range(12):
#     lp = list(np.asarray(i) for i in permutations(range(n)))
#     mc = list(filter(lambda x: check_complement(x) and check_mirror(x), lp))

#     print("{:<3}\t{:<3}\t{:<10}{:<10g}\t{:<10g}".format(n, len(mc), len(lp), len(lp) / len(mc) , len(mc)/len(lp)))
    
    
    
    
# =============================================================================
# 0  	1  	1         1         	1         
# 1  	1  	1         1         	1         
# 2  	2  	2         1         	1         
# 3  	2  	6         3         	0.333333  
# 4  	6  	24        4         	0.25      
# 5  	6  	120       20        	0.05      
# 6  	20 	720       36        	0.0277778 
# 7  	20 	5040      252       	0.00396825
# 8  	76 	40320     530.526   	0.00188492
# 9  	76 	362880    4774.74   	0.000209436
# 10 	312	3628800   11630.8   	8.59788e-05
# 11 	312	39916800  127938    	7.81626e-06   
# =============================================================================
    
# from collections import namedtuple
# from typing import List, Dict
# from pprint import pprint
# from typeguard import typechecked

# func = namedtuple("func", ["inputs", "outputs"])
# H = [
#      [func([0, 1], [1]), func([2, 3], [3]), func([4, 5], [5]), func([6, 7], [7])], 
#      [func([1, 3], [3]), func([5, 7], [7])], 
#      [fu
# constructive_permutation(H, H_p)nc([3, 7], [7])]
#     ]

# H_p = [
#        [func([0, 4], [0]), func([1, 5], [1]), func([2, 6], [2]), func([3, 7], [3])], 
#        [func([0, 2], [0]), func([1, 3], [1])], 
#        [func([0, 1], [0])]
#       ]
    
    
# @typechecked
# def constructive_permutation(H: List[List["func"]], H_p: List[List["func"]]) -> List[int] | None:
#     print("H")
#     pprint(H)
#     print("H'")
#     pprint(H_p)
    
    
    
    
import numpy as np


# phi = np.arange(4, dtype=int)
# omega = np.asarray([0, 3, 1, 2], dtype=int)
    
# P = np.zeros((4,4), dtype=int)
# P[phi, omega] = 1


# print(P)
# print(omega == P @ phi)

# print(P @ phi)
# print(P.T @ phi)


def to_P(omega):
    n = len(omega)
    phi = np.arange(n, dtype=int)     
    P = np.zeros((n, n), dtype=int)
    P[phi, omega] = 1
    return P    
    

omega  = [7, 3, 5, 1, 6, 2, 4, 0]
P = to_P(omega)

print(P)

(P == P.T).all()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# map = {
# 1: [0],  
# 2: [1 , 0],
# 4: [3 , 1 , 2 , 0],
# 8: [7 , 3 , 5 , 1 , 6 , 2 , 4 , 0],
# 16: [15 , 7 , 11 , 3 , 13 , 5 , 9 , 1 , 14 , 6 , 10 , 2 , 12 , 4 , 8 , 0],
# 32: [31 , 15 , 23 , 7 , 27 , 11 , 19 , 3 , 29 , 13 , 21 , 5 , 25 , 9 , 17 , 1 , 30 , 14 , 22 , 6 , 26 , 10 , 18 , 2 , 28 , 12 , 20 , 4 , 24 , 8 , 16 , 0]
# }

def parse_text_structure(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    text = text.replace("\n\n", "\n").replace("\n\n", "\n").replace("\n\n", "\n")
    blocks = text.split("\n***********************\n")
    data = {}

    for block in blocks:

        if len(block) == 0:
            continue
        lines = block.split("\n")
        n_line = lines[0]
        idx_line = lines[1]
        # print(idx_line)
        n = int(n_line.split(" = ")[1])
        idx = list(map(int, idx_line.split(": ")[1].split(", ")[:-1]))
        data[n] = idx
    return data

map = parse_text_structure('idx.log')


# sns.set()
# ax = sns.heatmap(P, vmin=0, vmax=1, cmap='Blues', ci=None)

# plt.show()

def plot_P(P):
    
    fig, ax = plt.subplots()
    
    plt.pcolormesh(np.rot90(P), edgecolors='lightgray', linewidth=1, cmap="Blues")
    ax = plt.gca()
    ax.set_aspect('equal')

    # im = ax.imshow(P, cmap="Blues", interpolation='none', vmin=0, vmax=1, aspect='equal')

    # ax.set_aspect('equal')
    # major_ticks = np.arange(0, len(P+1))

    ax.set_xticks(np.arange(len(P))+.5, labels=[])
    ax.set_yticks(np.arange(len(P))+.5, labels=[])
    # ax.set_xticks(np.arange(len(P))+.5, labels=np.arange(len(P)))
    # ax.set_yticks(np.arange(len(P))+.5, labels=np.arange(len(P)))
    ax.yaxis.label.set_color('0.9')
    ax.xaxis.label.set_color('0.9')
    plt.setp(ax.spines.values(), color="lightgray")

    
    # ax.patch.set_edgecolor('black')  
    # ax.patch.set_linewidth(1)  


    # ax.set_xticks(major_ticks, labels=[])
    # ax.set_yticks(np.arange(len(P))+.5, labels=[])
    # ax.set_xticks
    
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             # rotation_mode="anchor")
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.tick_params(which="minor", bottom=False, left=False)
    # ax.tick_params(top=True, bottom=False,
                   # labeltop=True, labelbottom=False)
    # 
    # ax.set_title("Harvest of local farmers (in tons/year)")
    # ax.grid(color="lightgray")
    # plt.axis('off')
    
        # Major ticks
    # ax.set_xticks(np.arange(0, P.shape[0], 1))
    # ax.set_yticks(np.arange(0, P.shape[1], 1))
    
    # # Labels for major ticks
    # ax.set_xticklabels(np.arange(1, 11, 1))
    # ax.set_yticklabels(np.arange(1, 11, 1))
    
    # # Minor ticks
    # ax.set_xticks(np.arange(-.5, P.shape[0], 1), minor=True)
    # ax.set_yticks(np.arange(-.5, P.shape[1], 1), minor=True)
    
    # # Gridlines based on minor ticks
    # ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    
    # # Remove minor ticks
    # ax.tick_params(which='minor', bottom=False, left=False)
    
    
    fig.tight_layout()
    return fig

    
def exchange_m(n):
    return to_P(np.arange(n, 0, -1) - 1)


# for omega in [map[8]]:
for n, omega in map.items():
    P = to_P(omega)
    fig = plot_P(P)
    plt.savefig(Path(__file__).parent.joinpath("figs", "permutaion-matrix-%02d.pdf" % n).as_posix(), 
                format="pdf", bbox_inches="tight")
    plt.show()
#%%
n = 16  
omega = map[n]
P = to_P(omega)
# print(P)

# for i in range(n):
    

D = np.zeros(n, dtype=int);
D[0] = n-1
# D[1] = D[0] // 2

# D[2] = (D[0] + D[1]) // 2
# D[3] = D[1] // 2

# D[4] = (D[0] + D[2]) // 2 
# D[5] = D[2] // 2

# D[6] = (D[3] + D[0]) // 2
# D[7] = D[3] // 2

# for i in range(1, n):
#     if i % 2 == 1:
#         D[i] = D[(i-1) // 2] // 2
#     else:   
#         D[i] = (D[0] + D[i // 2]) // 2
    
for i in range(0, n//2):
    D[2* i] = (D[i] + D[0]) // 2
    D[2* i + 1] = D[i] // 2
    
D = np.zeros(n, dtype=int);
# D[0] = n - 1
# D[1] = n / 2 - 1 

# D[2] = 3 * n / 4 - 1
# D[3] = n / 4 - 1

# D[4] = 7 * n / 8 - 1
# D[5] = 3 * n / 8 - 1 

# D[6] =  5 * n / 8 - 1
# D[7] =  n / 8 - 1


D[0] = n - 1
D[1] = 4 * n / 8 - 1 

D[2] = 3 * n / 4 - 1
D[3] = 2 * n / 8 - 1

D[4] = 7 * n / 8 - 1
D[5] = 3 * n / 8 - 1 

D[6] = 5 * n / 8 - 1
D[7] = 1 * n / 8 - 1
    
# for i in range(0, n//2):
#     D[2* i] = (D[i] + D[0]) / 2
#     D[2* i + 1] = n >> i  - 1

D = D.tolist()

print(omega)
print(D)

print(omega == D)

D2 = np.zeros(2*n, dtype=int)
for i in range(n):
    D2[2*i  + 1] = D[i]
    D2[2 * (n - 1 - i)] = 2 * n - 1 - D[i]
    
D2 = D2.tolist()
print(map[2*n])
print(D2)

#%%
import numpy as np
import scipy
from beeprint import pp
import matplotlib.pyplot as plt


x = np.asarray([0, 1, 2, 3, 4, 5])
y = np.asarray([28, 32, 40, 56, 88, 152])

# func = lambda t,a,b,c: a+b*np.exp(c*t)
func = lambda t,a,b: a+b*2**(t)
popt, pcov = scipy.optimize.curve_fit(func,  x,  y)
print(popt)
print(pcov)
perr = np.sqrt(np.diag(pcov))
print(perr)
print(np.linalg.cond(pcov))

plt.bar(x, y)
_x = np.arange(min(x)-.3, max(x)+.3, .1)
plt.plot(_x, func(_x, *popt), color="r")
plt.plot(x, func(x, *popt), "kx")
plt.show()


