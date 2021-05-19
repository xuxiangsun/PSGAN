import time
from data import create_dataset
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from models import create_model



if __name__=="__main__":
    opt = TrainOptions().parse()
    opt.save_latest_freq = 500 * opt.batch_size
    train_dataset = create_dataset(opt, flag='train')  # create a dataset given opt.dataset_mode and other options
    train_dataset_size = len(train_dataset)

    print('training dataset size:', train_dataset_size)

    pix2pix = create_model(opt)
    pix2pix.setup(opt)
    total_iters = 0
    visualizer = Visualizer(opt)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        data_start_time = time.time()
        for i, data in enumerate(train_dataset):
            iter_start_time = time.time()  # timer for computation per iteration
            data_end_time = iter_start_time
            datatime = data_end_time - data_start_time
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            pix2pix.set_input(data)
            cal_start_time = time.time()
            pix2pix.optimize_parameters(epoch)
            cal_end_time = time.time()
            caltime = cal_end_time - cal_start_time
            losses = pix2pix.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, total_iters, losses, datatime, caltime)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                pix2pix.save_networks(save_suffix)
            iter_data_time = time.time()
            data_start_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            pix2pix.save_networks('latest')
            pix2pix.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        pix2pix.update_learning_rate(opt)   # update learning rates at the end of every epoch
        
