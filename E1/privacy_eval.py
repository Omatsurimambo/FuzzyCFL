import random
import time

import FMD2 as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import json
import argparse


import json
import matplotlib.pyplot as plt
import seaborn as sns

def draw_pics(data_array):
    # data_array = data_array[:, ::-1]

    with open('config.json', 'r') as fi:
        conf = json.load(fi)
        size = conf["num"]
        n = conf["n"]
        count = conf["count"]
    fi.close()

    if count == 0:
        x_bar = "Number of Clusters"
        y_bar = "Privacy Threshold"
    elif count == 1:
        x_bar = "False Positive Rate"
        y_bar = "Privacy Threshold"
    elif count == 2:
        x_bar = "False Positive Rate"
        y_bar = "Number of Clusters"

    plt.figure(figsize=(8, 6)) 
    cbar_kws = {'format': '%.3f'}
    ax = sns.heatmap(data_array, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws=cbar_kws)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.invert_yaxis()

    ax.tick_params(axis='both', which='both', length=0)

    ax.set_xticklabels(['4', '5', '6', '7', '8'], rotation=0, ha='right')
    # ax.set_xticklabels(['1/2', '1/4', '1/8', '1/16', '1/32'], rotation=0, ha='right')
    ax.set_yticklabels(['2', '3', '4', '5', '6'], rotation=0, va='center')
    # ax.set_yticklabels(['4', '5', '6', '7', '8'], rotation=0, va='center')

    plt.title('Adversary Advantage', fontsize=14, y=1.02)
    # plt.xlabel('number of clusters k', fontsize=12)
    plt.xlabel(x_bar, fontsize=12)
    # plt.ylabel('Number of Clusters k', fontsize=12)
    plt.ylabel(y_bar, fontsize=12)


    plt.savefig(f'Adversary preference profiling accuracy_{count}.png')
    plt.close()  


def clear_pickle_file(file_path):
    # 以二进制写入模式打开文件
    with open(file_path, 'wb') as f:
        # 写入空内容
        pickle.dump(None, f)


class PrivacyTest:
    """
    initial the key pairs for tag generation
    """

    def __init__(self, t=2, p=1, n=6):
        # Threshold T for the identity number
        self.t = t
        # cluster number
        self.n = n
        # false positive rate p
        self.p = p
        # client size
        self.m = 120
        self.index = 0

        # key pairs for each cluster
        self.key_pairs = []
        self.curve = F.Curve.get_curve('secp256r1')
        self.keyGen()

    """
    Client generate identity tags until the tag size reach threshold t
    t : threshold
    n : client size
    p : false positive rate
    """

    def keyGen(self):
        for _ in range(8):
            key_pair = F.keyGen(self.curve, self.p)
            key_dict = {'sk': key_pair[0], 'pk': key_pair[1]}
            self.key_pairs.append(key_dict)

    def tag_generate(self):
        tag_size = []
        if self.t > self.n:
            #print("zero gen")
            return 1000000

        for i in range(4):
            j = 0
            count = 0

            while count < self.t:
                count = 0

                f = F.flag(self.key_pairs[j]["pk"], self.curve)
                for k in range(0, self.n):
                    if F.test(self.curve, self.key_pairs[k]["sk"], f):
                        count = count + 1
            tag_size.append(count)
        return sum(tag_size) / len(tag_size)

    """
    possibility function: two variable change, one fixed
    compute the matrix of the possibility of guessing clients' identity for Adversary A
    """

    def possibility(self, t_bool, n_bool, p_bool):
        t = [2, 3, 4, 5, 6]
        n = [4, 5, 6, 7, 8]
        p = [1, 2, 3, 4, 5]
        matrix = [[] for k in range(5)]
        for b in range(5):
            for j in range(5):
                if n_bool and t_bool:
                    self.n = n[b]
                    self.t = t[j]
                    num = self.tag_generate()
                    # print(self.n, num, 1 / num)
                    matrix[j].append(1 / num)
                if p_bool and t_bool:
                    self.p = p[b]
                    self.keyGen()
                    self.t = t[j]
                    num = self.tag_generate()
                    # print(self.n, num, 1 / num)
                    matrix[j].append(1 / num)
                if p_bool and n_bool:
                    self.p = p[b]
                    self.keyGen()
                    self.n = n[j]
                    num = self.tag_generate()
                    # print(self.n ,num,1 / num)
                    matrix[j].append(1 / num)
        print(matrix)
        return matrix

    def run(self, round, t_bool, n_bool, p_bool):
        with open('config.json', 'r') as f1:
            config = json.load(f1)
            config["num"] = 0


        data_to_save = {}
        for m in range(round):
            total_sum = np.array((self.possibility(t_bool, n_bool, p_bool)))
            list_name = f'list_{self.index + m + 1}'
            print(list_name)
            data_to_save[list_name] = total_sum
            print(m, " round\n")
        config["num"] = self.index + round
        config["n"] = self.n
        with open('config.json', 'w+') as f2:
            json.dump(config, f2, indent=4)
        f2.close()

        with open('lists.pkl', 'wb') as f3:
            pickle.dump(data_to_save, f3)

    def test(self):

        for _ in range(100):
            self.tag_generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('idGen', type=int, help='1 generates identities, 0 draws heatmaps', default=1)
    parser.add_argument('round', type=int, help='identity generation round', default=10)
    parser.add_argument('t', type=int, help='evaluate impact of threshold t', default=1)
    parser.add_argument('k', type=int, help='evaluate impact of cluster size k', default=1)
    parser.add_argument('p', type=int, help='evaluate impact of mingled rate p', default=0)

    args = parser.parse_args()
    if args.idGen == 1:
        clear_pickle_file('lists.pkl')
        privacy_test = PrivacyTest()
        privacy_test.run(args.round, args.t, args.k, args.p)
    elif args.idGen == 0:
        with open('config.json', 'r') as f:
            config = json.load(f)
            index = config["num"]
            n = config["n"]

        with open(f'lists.pkl', 'rb') as file:

            loaded_data = pickle.load(file)

        sum_list = np.zeros((5, 5))
        for i in range(0, index):
            sum_list += loaded_data[f'list_{i + 1}']
        sum_list = sum_list / index
        print(sum_list)
        draw_pics(sum_list)
        
        config["count"]+=1
        if config["count"] == 3:
            config["count"]=0
        with open('config.json', 'w+') as f2:
            json.dump(config, f2, indent=4)
        f.close()
        file.close()
