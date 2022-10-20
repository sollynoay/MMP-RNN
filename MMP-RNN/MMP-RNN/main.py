from para import Parameter
from train import Trainer

if __name__ == '__main__':
    para = Parameter().args
    para.model = 'MMPRNN'
    para.do_skip = True
    para.threads = 8
    para.n_blocks_a = 3
    para.n_blocks_b = 4
    para.n_features= 16
    para.num_gpus = 4
    para.end_epoch = 1
    para.frames = 8
    para.past_frames = 2
    para.future_frames = 2
    para.centralize = True
    # # choose one dataset
    # para.dataset = 'BSD'
    para.dataset = 'BSD'
    # para.dataset = 'reds_lmdb'
    # para.video = True

    # # resume training from existing checkpoint
    # para.resume = True
    # para.resume_file = './experiment/2020_12_29_01_56_43_ESTRNN_gopro_ds_lmdb/model_best.pth.tar'

    # # test using existing checkpoint
    # para.test_only = True
    # para.test_save_dir = './results/'
    # para.test_checkpoint = './experiment/2020_12_29_01_56_43_ESTRNN_gopro_ds_lmdb/model_best.pth.tar'

    trainer = Trainer(para)
    trainer.run()
