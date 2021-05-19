import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./datasets/AID', help='dataset you want to deal')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.dataset
    dataset_type = root.split('/')[-2]
    num = 0
    if not os.path.isfile(os.path.join('./datasets/dataset_dict', '{}_dict.txt'.format(dataset_type))):
        for classes in os.listdir(root):
            with open(os.path.join('./datasets/dataset_dict', '{}_dict.txt'.format(dataset_type)), "a") as f:
                f.write(classes + ':' + str(num) + '\n')
            num += 1
    print('Reading {} finished!'.format(dataset_type))