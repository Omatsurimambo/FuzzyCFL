import itertools
import time
import numpy as np
import logging

# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)

a = np.array([2, 3, 5, 7, 11])

q = 10000000000000000000001

def fastExpMod(b, e, m):
    r = 1
    while e != 0:
        if (e & 1) == 1:
            # ei = 1, then mul
            r = (r * b) % m
        e >>= 1
        # b, b^2, b^4, b^8, ... , b^(2^n)
        b = (b * b) % m
    return r


def total(k, fuzzyIdentity, cluster_assign):
    x = np.zeros((k, k), dtype=int)
    for k_i in range(k):
        for index, f_list in enumerate(fuzzyIdentity):
            if k_i in f_list:
                true_id = cluster_assign[index]
                x[k_i, true_id] += 1

    results = []
    for i in range(k):
        result = 1
        for j in range(k):
            result = (result * fastExpMod(a[j], x[i, j], q)) % q
        results.append(result)
    return results


def find_x(row_sum, row_idx, k, p, cl, result):
    target_sum = row_sum

    fuzzy = round(cl * p)
    search_ranges = [
        [cl] if i == row_idx 
        else range(0,30)
        for i in range(k)
    ]

    for x_values in itertools.product(*search_ranges):
        if sum(x_values) != target_sum:
            continue
        
        product = 1
        for i in range(k):
            product = (product * fastExpMod(a[i], x_values[i], q)) % q
        
        if product == result:
            return np.array(x_values)
    
    return None

def construct_matrix(k, p, row_sums, cl_ct, results):

    res=[]
    for i in range(k):
        x_rows = find_x(row_sums[i], i, k, p, cl_ct[i], results[i])
        if x_rows is not None:
            res.append(x_rows)
        else:
            print(f"Did not find a matching x value for row {i + 1}")
    final_res = np.vstack(res)
    return final_res
