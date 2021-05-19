from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import time
import os


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt, flag='test')  # create a dataset given opt.dataset_mode and other options
    print('Dataset Length:{}'.format(len(dataset)))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()
    acc_adv = 0
    total_time_start = time.time()
    iter_time_start = time.time()
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        iter_time_end = time.time()
        iter_time = iter_time_end - iter_time_start
        print('processing (%06d)-th image, time : %.4f ' % (i, iter_time))
        acc_adv += model.acc_adv.detach().cpu().numpy()
        iter_time_start = time.time()
    total_time_end = time.time()
    total_time = total_time_end - total_time_start
    print('adv: %.4f%%' % ((acc_adv / len(dataset)) * 100))
    print('total time: %.4f' % (total_time))
    message = 'adv: %.4f%%\t' % ((acc_adv / len(dataset)) * 100)

    test_name = os.path.join(opt.result_dir, opt.name, 'test_log.txt')
    with open(test_name, "a") as test_file:
        test_file.write('{}\n'.format(opt.testname))
        test_file.write('%s\n' % message)