import os
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import time 

if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.

    test_dataset = create_dataset(opt, flag='test')  # create a dataset given opt.dataset_mode and other options
    print('Dataset length:{}'.format(len(test_dataset)))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)
    model.eval()
    acc_totalacc = 0
    acc_perclass = [0 for i in range(opt.classes)]
    y_classes = ['Totalacc']
    data_name = opt.dataroot.split('/')[-1]

    with open(os.path.join('./datasets/dataset_dict', '{}_dict.txt'.format(data_name)), "r") as f:
        lines = f.readlines()
        for i in lines:
            classes = i.strip().split(":")[0]
            y_classes.append(classes)
    num_perclass = []

    for i in y_classes:
        if not i == 'Totalacc':
            clas_dir = os.path.join(opt.dataroot, 'test_{}'.format(opt.train_ratio), i)
            num_perclass.append(len(os.listdir(clas_dir)))
    data_start_time = time.time()
    for i, data in enumerate(test_dataset):
        data_end_time = time.time()
        datatime = data_end_time - data_start_time
        model.set_input(data)
        cal_start_time = time.time()
        model.test()
        cal_end_time = time.time()
        caltime = cal_end_time - cal_start_time
        acc = model.get_current_acc()
        print('iter:{}, datatime:{}, caltime:{}'.format(i, datatime, caltime))
        assert acc['totalacc'] == np.sum(acc['perclass'])
        acc_totalacc += acc['totalacc']
        acc_perclass = np.sum([acc_perclass, acc['perclass']], axis=0)
        data_start_time = time.time()

        
    message = '----------------------Result----------------------\n'
    print(message)
    for k, v in enumerate(y_classes):
        if k == 0:
            message += v + '\t  %.2f%%\n' % ((acc_totalacc / len(test_dataset)) * 100)
            message += '------------------------AP------------------------\n'
            print(v, '\t  %.2f%%' % ((acc_totalacc / len(test_dataset)) * 100))
            print('----------------------AP----------------------\n')
        else:
            message += v + '\t  %.2f%%\n' % ((acc_perclass[k - 1] / (num_perclass[k - 1])) * 100)
            print(v, '\t  %.2f%%' % ((acc_perclass[k - 1] / (num_perclass[k - 1])) * 100))
    message += '----------------------Result----------------------\n'
    print('----------------------Result----------------------\n')
    test_name = os.path.join(opt.result_dir, opt.name, 'test_log.txt')
    with open(test_name, "a") as test_file:
        test_file.write('{}\n'.format(opt.testname))
        test_file.write('%s\n' % message)