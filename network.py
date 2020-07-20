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
        self.infer_dir = args.infer_dir
        self.val_interval = args.val_interval

        self.pixel_max = 1.
        self.pixel_min = 0.
        self.patch_size = 25
        self.img_c = 1
        self.scale = args.scale
        self.channel = args.channel

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.init_lr = args.lr
        
        if args.phase == 'train':
            ''' Load 291 Dataset for Training ''' 
            self.train_HR = prepare_patches(load_train(), patch_size=100, stride=25)
            self.train_LR = bicubic_downsampling(self.train_HR, scale=self.scale)

            self.train_HR = preprocessing(self.train_HR)
            self.train_LR = preprocessing(self.train_LR)


            ''' Load Datasets for Validation (Set5)'''
            self.test_HR = prepare_patches(load_set5(), patch_size=100, stride=100)
            self.test_LR = bicubic_downsampling(self.test_HR, scale=self.scale)

            self.test_HR = preprocessing(self.test_HR)
            self.test_LR = preprocessing(self.test_LR)

            self.test_HR = self.test_HR[:32]
            self.test_LR = self.test_LR[:32]

            print("---------\nDatasets\n---------")
            print("TRAIN LABEL : ", str(np.array(self.train_HR).shape))
            print("TRAIN INPUT : ", str(np.array(self.train_LR).shape))
            print("TEST LABEL  : ", str(np.array(self.test_HR).shape))
            print("TEST INPUT  : ", str(np.array(self.test_LR).shape))
            quit()

    def network(self, x, reuse=False):
        skipConnect = []
        with tf.variable_scope("SRDenseNet", reuse=reuse):
            x = conv(x, 16, kernel_size=3, stride=1, padding='SAME', use_bias=True, scope='low_level_conv_0')
            x = relu(x)
            skipConnect.append(x)
            for idx in range(8):
                x = denseBlock(x, growth_rate=16, kernel_size=3, stride=1, padding='SAME', scope='denseBlock_' + str(idx))
                skipConnect.append(x)

            x = concatenation(skipConnect)
            x = bottleneck(x, channels=256, kernel_size=1, stride=1, padding='SAME', use_bias=True, scope='bottleneck_0')

            for idx in range(2):
                x = deconv(x, 256, kernel_size=3, stride=2, padding='SAME', use_bias=True, scope='deconv_' + str(idx))
                x = relu(x)

            x = conv(x, self.img_c, kernel_size=3, stride=1, padding='SAME', use_bias=True, scope='reconstrucion_0')
            return x

    def build_model(self):
        """ Graph Input
            Ground-Truth(HR) : Y 
            Low-Resolution   : X """
        self.train_X = tf.placeholder(tf.float32, [self.batch_size, self.patch_size, self.patch_size, self.img_c], name='train_X') 
        self.train_Y = tf.placeholder(tf.float32, [self.batch_size, self.patch_size * self.scale, self.patch_size * self.scale, self.img_c], name='train_Y')

        # self.test_X  = tf.placeholder(tf.float32, [len(self.test_HR), self.patch_size, self.patch_size, self.img_c], name='test_X')
        # self.test_Y  = tf.placeholder(tf.float32, [len(self.test_LR), self.patch_size * self.scale, self.patch_size * self.scale, self.img_c], name='test_Y')

        self.test_X  = tf.placeholder(tf.float32, [1, None, None, self.img_c], name='test_X')
        self.test_Y  = tf.placeholder(tf.float32, [1, None, None, self.img_c], name='test_Y')

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

            if epoch == 30:
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

        # self.save(self.checkpoint_dir, self.epoch)
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
        # test_datasets = ['Urban100']

        ''' TEST DATA LOAD '''
        for dataset in test_datasets:
            if dataset == 'Set5':
                self.test_HR = load_set5()
            if dataset == 'Set14':
                self.test_HR = load_set14()
            if dataset == 'B100':
                self.test_HR = load_b100()
            if dataset == 'Urban100':
                self.test_HR = load_urban100()

            self.test_HR = prepare_patches(self.test_HR, patch_size=100, stride=100)
            self.test_LR = bicubic_downsampling(self.test_HR, scale=self.scale)

            test_loss_mean = 0.
            test_psnr_mean = 0.
            test_ssim_mean = 0.

            start_time = time.time()
            for idx in range(len(self.test_HR)):
                label_img = self.test_HR[idx]
                input_img = self.test_LR[idx]

                h, w = label_img.size
                _h, _w = input_img.size

                label_img = np.array(label_img)[:,:,0:1]
                input_img = np.array(input_img)[:,:,0:1]

                label_img = normalize(label_img)
                input_img = normalize(input_img)

                label_img = label_img.reshape([1, h, w,self.img_c])
                input_img = input_img.reshape([1,_h,_w,self.img_c])

                test_feed_dict = {
                    self.test_X : input_img,
                    self.test_Y : label_img
                }

                test_output, test_loss, test_psnr, test_ssim = self.sess.run(
                                        [self.test_output, self.test_loss, self.test_psnr, self.test_ssim], 
                                        feed_dict=test_feed_dict)

                test_loss_mean += test_loss
                test_psnr_mean += test_psnr
                test_ssim_mean += test_ssim

            test_loss_mean /= len(self.test_HR)
            test_psnr_mean /= len(self.test_HR)
            test_ssim_mean /= len(self.test_HR)

            print("{} Average - test_loss: {}, test_psnr: {}, test_ssim: {}".format(dataset, test_loss_mean, test_psnr_mean, test_ssim_mean))
            print("     Elapsed Time : %dhour %dmin %dsec" % time_calculate(time.time()- start_time))

    def infer(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        # infer_datasets = ['Set5', 'Set14', 'B100', 'Urban100']
        infer_datasets = ['Set5']

        ''' Infer DATA LOAD '''
        for dataset in infer_datasets:
            if dataset == 'Set5':
                self.test_HR = load_set5()
            if dataset == 'Set14':
                self.test_HR = load_set14()
            if dataset == 'B100':
                self.test_HR = load_b100()
            if dataset == 'Urban100':
                self.test_HR = load_urban100()

            test_loss_mean = 0.
            test_psnr_mean = 0.
            test_ssim_mean = 0.

            start_time = time.time()
            for idx in range(len(self.test_HR)):
                label_img = mod_crop(self.test_HR[idx], self.scale)
                input_img = bicubic_downsampling(label_img, scale=self.scale)[0]

                h, w = label_img.size
                _h, _w = input_img.size

                # cbcr = input_img.resize((h, w), Image.BICUBIC)
                cbcr = bicubic_upsampling(input_img, scale=self.scale)[0]
                cbcr = np.array(cbcr)[:,:,1:3]

                label_y = np.array(label_img)[:,:,0:1]
                input_y = np.array(input_img)[:,:,0:1]

                label_y = normalize(label_y)
                input_y = normalize(input_y)

                label_y = label_y.reshape([1, w, h, self.img_c])
                input_y = input_y.reshape([1,_w,_h, self.img_c])

                test_feed_dict = {
                    self.test_X : input_y,
                    self.test_Y : label_y
                }

                test_output, test_loss, test_psnr, test_ssim = self.sess.run(
                                        [self.test_output, self.test_loss, self.test_psnr, self.test_ssim], 
                                        feed_dict=test_feed_dict)
                    
                test_output = denormalize(test_output)
                test_output = test_output.reshape([w, h, self.img_c])
                test_output = np.concatenate((test_output, cbcr), axis=2)
                test_output = Image.fromarray(test_output, mode='YCbCr')
                test_output = ycbcr2rgb(test_output)

                if not os.path.exists(os.path.join(self.infer_dir, dataset, str(self.scale)+'x')):
                    os.makedirs(os.path.join(self.infer_dir, dataset, str(self.scale)+'x'))
                
                infer_path = os.path.join(self.infer_dir, dataset, str(self.scale)+'x')
                imsave(test_output, os.path.join(infer_path, 'SRDenseNet_' + str(self.scale) + 'x_' + str(idx) + '_' + str(test_psnr) + '_' + str(test_ssim) + '.png'))
                print("{}[{}] - test_loss: {}, test_psnr: {}, test_ssim: {}".format(dataset, idx, test_loss, test_psnr, test_ssim))

                test_loss_mean += test_loss
                test_psnr_mean += test_psnr
                test_ssim_mean += test_ssim

            test_loss_mean /= len(self.test_HR)
            test_psnr_mean /= len(self.test_HR)
            test_ssim_mean /= len(self.test_HR)

            print("{} Average - test_loss: {}, test_psnr: {}, test_ssim: {}".format(dataset, test_loss_mean, test_psnr_mean, test_ssim_mean))
            print("     Elapsed Time : %dhour %dmin %dsec" % time_calculate(time.time()- start_time))

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
