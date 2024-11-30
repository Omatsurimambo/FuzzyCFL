import json
import argparse
import time

from aggregator import TrainCluster

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training') 
    parser.add_argument('--dataset', type=str, default='texas100', help='dataset used for training')  
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.01)')   
    parser.add_argument('--tau', type=int, default=5, help='number of local epochs')   
    parser.add_argument('--k', type=int, default=5, help='number of test clusters')
    parser.add_argument('--m', type=int, default=120,  help='number of clients')
    parser.add_argument('--t', type=int, default=2,  help='threshold') 
    parser.add_argument('--alg', type=str, default='ifca',
                             help='fl algorithms: mingledpie, ifca')
    parser.add_argument('--m_test', type=int, default=60,  help='number of test clients')
    parser.add_argument('--p', type=int, default=1,  help='false positive rate')
    parser.add_argument('--num_epochs', type=int, default=300, help='number of maximum communication roun') 
    parser.add_argument('--label_groups', type=str, default='default', help='Division method')
    parser.add_argument('--per_epochs', type=int, default=1, help='pre training rounds')  
    parser.add_argument('--samples_per_subset', type=int, default=1000, help='samples_per_subset')
    parser.add_argument('--disruption', type=bool, default=False, help='Whether to disrupt initialization clustering')
    parser.add_argument('--LR_DECAY', type=bool, default=False, help='Whether the learning rate is decreasing')
    parser.add_argument('--isomerism', type=bool, default=False, help='')
    parser.add_argument('--h1', type=int, default=200, help='hidden layer')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--train_seed', type=int, default=0, help='Random seed')
    parser.add_argument("--project-dir",type=str,default="output")
    parser.add_argument("--dataset-dir",type=str,default="output")
    args = parser.parse_args()


    if args.label_groups == 'default':
        label_groups = parse_label_groups(args.label_groups)
    else:
        label_groups = json.loads(args.label_groups.replace("'", '"'))
    args.label_groups = label_groups

    return args

def parse_label_groups(arg):
    return [[57, 76, 23, 56, 63, 87, 11, 55, 46, 5, 48, 27, 58, 98, 31, 30, 43, 54, 14, 83],
            [97, 22, 16, 38, 26, 65, 64, 91, 59, 40, 19, 15, 8, 49, 3, 34, 68, 79, 77, 70],
            [1, 10, 82, 51, 92, 12, 90, 84, 28, 45, 6, 93, 94, 39, 44, 99, 67, 96, 32, 61],
            [74, 33, 80, 47, 9, 21, 41, 88, 85, 25, 62, 60, 73, 75, 37, 36, 86, 52, 69, 17],
            [71, 35, 50, 2, 4, 42, 18, 89, 53, 72, 0, 78, 66, 81, 7, 13, 95, 20, 24, 29]]

def main():

    args = get_args()
    args.train_seed = args.data_seed
    print("args:",args)

    exp = TrainCluster(args)
    exp.setup()
    exp.run()

if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster Ended in %0.2f hour (%.3f sec) " % (duration/float(3600), duration))

