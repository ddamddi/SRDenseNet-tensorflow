from utils import *
from ops import *
import time

class SRDenseNet:
    def __init__(self, sess, args):
        self.model_name = 'SRDenseNet'
        self.dataset = '291'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.val_interval = args.val_interval

        self.pixel_max = 1.
        self.pixel_min = 0.
        self.patch_size = 25
        self.img_c = 1
        self.scale = args.scale
        # self.img_c = args.num_channel
        self.channel = args.channel

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.init_lr = args.lr
        
        if args.phase == 'train':
            ''' Load 291 Dataset for Training ''' 
            self.train_HR, self.train_LR = load_291(scale=self.scale)

            ''' Load Datasets for Validation (Set5)'''
            self.test_HR, self.test_LR = load_set5(scale=self.scale)
            self.test_HR, self.test_LR = create_sub_patches((self.test_HR, self.test_LR))

            print("---------\nDatasets\n---------")
            print("TRAIN LABEL : ", str(np.array(self.train_HR).shape))
            print("TRAIN INPUT : ", str(np.array(self.train_LR).shape))
            print("TEST LABEL  : ", str(np.array(self.test_HR).shape))
            print("TEST INPUT  : ", str(np.array(self.test_LR).shape))

    def network(self, x, reuse=False):
        with tf.variable_scope("SRDenseNet", reuse=reuse):
            for idx in range(8):
                x = dense_block(x, growth_rate=16, kernel_size=3, stride=1, padding='SAME', scope='dense_block_' + str(idx))
            
            for idx in range(2):
                x = deconv(x, 256, kernel_size=3, stride=2, padding='VALID', use_bias=True, scope='deconv_' + str(idx))
                x = relu(x)

            x = conv(x, self.img_c, kernel_size=3, stride=1, padding='SAME', use_bias=True, scope='reconstrucion_0')
            return x

    def build_model(self):
        """ Graph Input
            Ground-Truth(HR) : Y 
            Low-Resolution   : X """
        self.train_X = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, self.img_c], name='train_X') 
        self.train_Y = tf.placeholder(tf.float32, [self.batch_size, self.patch_size * self.scale, self.patch_size * self.scale, self.img_c], name='train_Y')

        self.test_X  = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, self.img_c], name='test_X')
        self.test_Y  = tf.placeholder(tf.float32, [None, self.patch_size * self.scale, self.patch_size * self.scale, self.img_c], name='test_Y')

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        """ Model """
        self.train_logits = self.network(self.train_X)
        self.test_logits  = self.network(self.test_X, reuse=True)

        self.train_output = tf.clip_by_value(self.train_logits, self.pixel_min, self.pixel_max)
        self.test_output = tf.clip_by_value(self.test_logits, self.pixel_min, self.pixel_max)

        self.train_loss = tf.reduce_mean(tf.square((self.train_logits - self.train_Y)))
        self.test_loss  = tf.reduce_mean(tf.square((self.test_logits - self.test_Y)))

        # reg_loss = tf.losses.get_regularization_loss()
        # self.train_loss += reg_loss
        # self.test_loss += reg_loss

        self.train_psnr = tf.reduce_mean(tf.image.psnr(self.train_output, self.train_Y, max_val=self.pixel_max-self.pixel_min))
        self.test_psnr = tf.reduce_mean(tf.image.psnr(self.test_output, self.test_Y, max_val=self.pixel_max-self.pixel_min))

        self.train_ssim = tf.reduce_mean(tf.image.ssim(self.train_output, self.train_Y, max_val=self.pixel_max-self.pixel_min))
        self.test_ssim = tf.reduce_mean(tf.image.ssim(self.test_output, self.test_Y, max_val=self.pixel_max-self.pixel_min))

        """ Training """
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.train_loss)

        """ Summary """
        self.summary_train_loss = tf.summary.scalar("train_loss", self.train_loss)
        self.summary_train_psnr = tf.summary.scalar("train_psnr", self.train_psnr)
        self.summary_train_ssim = tf.summary.scalar("train_ssim", self.train_ssim)

        self.summary_test_loss = tf.summary.scalar("test_loss", self.test_loss)
        self.summary_test_psnr = tf.summary.scalar("test_psnr", self.test_psnr)
        self.summary_test_ssim = tf.summary.scalar("test_ssim", self.test_ssim)

        self.train_summary = tf.summary.merge([self.summary_train_loss, self.summary_train_psnr, self.summary_train_ssim])
        self.test_summary = tf.summary.merge([self.summary_test_loss, self.summary_test_psnr, self.summary_test_ssim])
    
    def train(self):
        # Initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # Model Saver
        self.saver = tf.train.Saver()
        
        # Summary Writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        total_batch_iter = len(self.train_HR) // self.batch_size
        print(" [*] Total batch iterations :", total_batch_iter)
        
        # Restore checkpoints if exists
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)        
        if could_load:
            start_epoch = checkpoint_counter
            train_counter = checkpoint_counter * total_batch_iter + 1
            val_counter = checkpoint_counter * (total_batch_iter // self.val_interval + 1) + 1
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            train_counter = 1
            val_counter = 1
            print(" [!] Load failed...")
        
        epoch_lr = self.init_lr
        if(start_epoch == self.epoch // 2):
            epoch_lr /= 10

        epoch_mean_loss = []
        mean_loss = 0
        start_time = time.time()
        print(" [*] Start Training...")
        for epoch in range(start_epoch, self.epoch):

            if epoch == self.epoch // 2:
                print(" [*] Learning rate Decreased from %.5f to %.5f" % (epoch_lr , epoch_lr/10))
                epoch_lr /= 10

            if epoch >= 60 and epoch_mean_loss[-1] < epoch_mean_loss[-2]:
                print(" [*] No Improvements of the Loss... Stop Training...")
                break
            
            for batch_idx in range(total_batch_iter):
                batch_HR = self.train_HR[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                batch_LR = self.train_LR[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]

                train_feed_dict = {
                    self.train_Y : batch_HR,
                    self.train_X : batch_LR,
                    self.lr : epoch_lr
                }

                _, train_loss, train_psnr, train_summary_str = self.sess.run(
                    [self.train_op, self.train_loss, self.train_psnr, self.train_summary], feed_dict=train_feed_dict) 
                self.writer.add_summary(train_summary_str, train_counter)
                train_counter += 1
                mean_loss += train_loss

                if batch_idx != 0 and batch_idx % self.val_interval == 0:
                    test_loss, test_psnr = self.validate(val_counter)
                    val_counter += 1
                    print("Epoch: [%2d] [%5d/%5d] train_loss: %.5f, train_psnr: %.5f, test_loss: %.5f, test_psnr: %.5f" % (epoch+1, batch_idx+1, total_batch_iter, train_loss, train_psnr, test_loss, test_psnr))
                else:
                    print("Epoch: [%2d] [%5d/%5d] train_loss: %.5f, train_psnr: %.5f" % (epoch+1, batch_idx+1, total_batch_iter, train_loss, train_psnr))


            mean_loss /= total_batch_iter
            epoch_mean_loss.append(mean_loss)
            mean_loss = 0

            self.save(self.checkpoint_dir, epoch+1)

        self.save(self.checkpoint_dir, self.epoch)
        print("Elapsed Time : %dhour %dmin %dsec" % time_calculate(time.time() - start_time))

    def validate(self, val_counter):
        test_feed_dict = {
            self.test_X : self.test_LR,
            self.test_Y : self.test_HR
        }

        summary_str, test_loss, test_psnr = self.sess.run([self.test_summary, self.test_loss, self.test_psnr], feed_dict=test_feed_dict)
        self.writer.add_summary(summary_str, val_counter)

        return test_loss, test_psnr
    
    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        test_datasets = ['Set5', 'Set14', 'B100', 'Urban100']

        ''' TEST DATA LOAD '''
        for dataset in test_datasets:
            if dataset == 'Set5':
                self.test_HR, self.test_LR = load_set5(scale=self.test_scale)
            if dataset == 'Set14':
                self.test_HR, self.test_LR = load_set14(scale=self.test_scale)
            if dataset == 'B100':
                self.test_HR, self.test_LR = load_b100(scale=self.test_scale)
            if dataset == 'Urban100':
                self.test_HR, self.test_LR = load_urban100(scale=self.test_scale)

            self.test_HR, self.test_LR = create_sub_patches((self.test_HR, self.test_LR))
            # self.test_HR_cbcr, self.test_LR_cbcr = load_set14(scale=2, color_space='YCbCr')
            # print(self.test_HR_cbcr.shape)

            test_loss_mean = 0.
            test_psnr_mean = 0.
            test_ssim_mean = 0.

            start_time = time.time()
            for idx in range(len(self.test_HR)):
                h, w, c = self.test_HR[idx].shape
                _h, _w, _c = self.test_LR[idx].shape
                h = min(h, _h)
                w = min(w, _w)

                HR = self.test_HR[idx][:h,:w]
                LR = self.test_LR[idx][:h,:w]
                
                # HR_cbcr = self.test_HR_cbcr[idx][:,:,:2]
                # LR_cbcr = self.test_LR_cbcr[idx][:,:,:2]

                HR = HR.reshape([1,h,w,c])
                LR = LR.reshape([1,h,w,c])

                test_feed_dict = {
                    self.test_X : LR,
                    self.test_Y : HR
                }

                test_output, test_loss, test_psnr, test_ssim = self.sess.run([self.test_output, self.test_loss, self.test_psnr, self.test_ssim], feed_dict=test_feed_dict)

                test_loss_mean += test_loss
                test_psnr_mean += test_psnr
                test_ssim_mean += test_ssim

                # test_output = test_output.reshape([h,w,c])
                # output = np.concatenate((test_output, LR_cbcr[:,:,0:1], LR_cbcr[:,:,1:2]), axis=2)
                # output = denormalize(output)
                # output = ycrcb2bgr(output)
                
                # gt = self.test_HR_cbcr[idx] 
                # gt = np.concatenate((gt[:,:,2:3], gt[:,:,0:2]),axis=2)
                # gt = denormalize(gt)
                # gt = ycrcb2bgr(gt)

                # ilr = self.test_LR_cbcr[idx]
                # ilr = np.concatenate((ilr[:,:,2:3], ilr[:,:,0:2]),axis=2)
                # ilr = denormalize(ilr)
                # ilr = ycrcb2bgr(ilr)

                # print('Image' + str(idx) + '- psnr: {}, ssim: {}'.format(test_psnr, test_ssim))
                # cv2.imshow('Infrence' + str(idx), output)
                # cv2.imshow('GT' + str(idx), gt)
                # cv2.imshow('ILR' + str(idx), ilr)

                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            test_loss_mean /= len(self.test_HR)
            test_psnr_mean /= len(self.test_HR)
            test_ssim_mean /= len(self.test_HR)

            print("{} Average - test_loss: {}, test_psnr: {}, test_ssim: {}".format(dataset, test_loss_mean, test_psnr_mean, test_ssim_mean))
            print("Elapsed Time : %dhour %dmin %dsec" % time_calculate(time.time()- start_time))

    @property
    def model_dir(self):
        return "{}_{}x_{}_{}_{}".format(self.model_name, self.scale, self.dataset, self.channel, self.batch_size)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        print(" [*] Model Saving...")
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)
        print(" [*] Save complete")

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
