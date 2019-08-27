import os, sys, glob, re
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.backend import manual_variable_initialization
from keras.objectives import binary_crossentropy 
from keras.metrics import categorical_accuracy 
from keras.layers import Dropout, Dense, Flatten, Input
from keras.models import Model

manual_variable_initialization(True)
import tables
from tensorflow.python.client import device_lib

from azureml.accel.models import (
    Densenet121,
    Resnet152,
    Resnet50,
    SsdVgg,
    Vgg16,
    QuantizedDensenet121,
    QuantizedResnet152,
    QuantizedResnet50,
    QuantizedSsdVgg,
    QuantizedVgg16,
    )

import tqdm
from tqdm import tqdm_notebook, tqdm

device_lib.list_local_devices()
# %load_ext autoreload
# %autoreload 2

from utils import *
import importlib
from time import strftime


def reload():
    importlib.reload(sys.modules[__name__])


from data import data


#____________________________________________________________________
# Setting paths

class Trainer(object):
    """docstring for Trainer"""

    arch_dict = {
        'Densenet121' : Densenet121,
        'Resnet152' : Resnet152,
        'Resnet50' : Resnet50,
        'SsdVgg' : SsdVgg,
        'Vgg16' : Vgg16,
        }

    arch_dict_quantized = {
        'Densenet121' : QuantizedDensenet121,
        'Resnet152' : QuantizedResnet152,
        'Resnet50' : QuantizedResnet50,
        'SsdVgg' : QuantizedSsdVgg,
        'Vgg16' : QuantizedVgg16,
        }

    arch_dict_inputsize = {
        'Densenet121' : 224, # ?
        'Resnet152'   : 224, # ?
        'Resnet50'    : 224,
        'SsdVgg'      : 300,
        'Vgg16'       : 224, # ?
        }

    arch_dict_fcsize = {
        'Densenet121' : 2048,
        'Resnet152'   : 1024, # ?
        'Resnet50'    : 1024,
        'SsdVgg'      : 1024, # ?
        'Vgg16'       : 1024, # ?
        }

    arch_dict_inputdimclassifier = {
        'Densenet121' : 1024,
        'Resnet152'   : 0, # ?
        'Resnet50'    : 20448,
        'SsdVgg'      : 0, # ?
        'Vgg16'       : 0, # ?
        }

    def __init__(self):
        super(Trainer, self).__init__()
        self.base                 = '/home/thomas/acceltraining'
        self.training_dir_base = osp.join(
            self.base,
            'MachineLearningNotebooks/project-brainwave/trainings'
            )
        self.data = data
        self.data_size = data.data_size
        self.tags = []



    def get_counter_training(self):
        print(osp.join(self.training_dir_base, 'training_*'))
        training_dirs = glob.glob(osp.join(self.training_dir_base, 'training_*'))
        print(training_dirs)
        counters = []
        for t in training_dirs:
            match = re.match(
                r'training_(\d+)',
                osp.basename(t)
                )
            if not match:
                continue
            counter = int(match.group(1))
            counters.append(counter)

        if len(counters) > 0:
            return max(counters) + 1
        else:
            return 0

    def set_new_training_dir(self, dry=False):
        training_dir = 'training_{0}_{1}'.format(
            self.get_counter_training(),
            strftime('%b%d')
            )
        for tag in self.tags:
            training_dir += '_' + tag
        training_dir = osp.join(self.training_dir_base, training_dir)

        if osp.isdir(training_dir):
            raise ValueError('Error creating unique training dir: was trying {0}'.format(training_dir))

        self.set_training_dir(training_dir)

        if not dry:
            if not osp.isdir(self.training_dir): os.makedirs(self.training_dir)
            if not osp.isdir(self.custom_weights_dir): os.makedirs(self.custom_weights_dir)
            if not osp.isdir(self.custom_weights_dir_q): os.makedirs(self.custom_weights_dir_q)
            if not osp.isdir(self.results_dir): os.makedirs(self.results_dir)
        print('Creating new training dir: {0}'.format(training_dir))


    def set_new_training_dir_old(self, dry=False):
        training_dir = osp.join(
            self.base,
            'MachineLearningNotebooks/project-brainwave/trainings/training{0}'
            )
        i = 1
        while osp.isdir(training_dir.format(i)):
            i += 1
        training_dir = training_dir.format(i)
        self.set_training_dir(training_dir)

        if not dry:
            if not osp.isdir(self.training_dir): os.makedirs(self.training_dir)
            if not osp.isdir(self.custom_weights_dir): os.makedirs(self.custom_weights_dir)
            if not osp.isdir(self.custom_weights_dir_q): os.makedirs(self.custom_weights_dir_q)
            if not osp.isdir(self.results_dir): os.makedirs(self.results_dir)
        print('Creating new training dir: {0}'.format(training_dir))


    def set_training_dir_latest(self):
        training_dir = osp.join(
            self.base,
            'MachineLearningNotebooks/project-brainwave/trainings/training{0}'
            )
        i = 100
        while not(osp.isdir(training_dir.format(i))):
            i -= 1
        training_dir = training_dir.format(i)
        self.set_training_dir(training_dir)
        print('Using latest found training dir: {0}'.format(training_dir))

    def set_training_dir(self, training_dir):
        self.training_dir = training_dir
        self.custom_weights_dir   = osp.join(self.training_dir, 'custom-weights')
        self.custom_weights_dir_q = osp.join(self.training_dir, 'custom-weights-quantized')
        self.saved_model_dir      = osp.join(self.training_dir, 'models')
        self.results_dir          = osp.join(self.training_dir, 'results')
        self.logfile              = osp.join(self.training_dir, 'train.log')

    def log(self, msg):
        if not osp.isdir(self.training_dir): return
        with open(self.logfile, 'a') as f:
            f.write(msg + '\n')

    def print_and_log(self, *args):
        print(*args)
        self.log(' '.join(map(str, args)))


    def preprocess_images(self, from_size, to_size):
        # Create a placeholder for our incoming images
        in_images = tf.placeholder(tf.float32)
        in_images.set_shape([None, from_size, from_size, 3])
        if from_size == to_size:
            image_tensors = in_images
        else:
            image_tensors = tf.image.resize_images(in_images, [to_size, to_size])
            image_tensors = tf.to_float(image_tensors)            
        return in_images, image_tensors


    def construct_classifier(self, architecture='Resnet50'):
        K.set_session(tf.get_default_session())
        
        # FC_SIZE = 1024
        FC_SIZE = self.arch_dict_fcsize[architecture]
        input_size = self.arch_dict_inputdimclassifier[architecture]
        self.print_and_log(
            'Constructing classifier for {0}; FC_SIZE = {1}, input_size = {2}'
            .format(architecture, FC_SIZE, input_size)
            )

        NUM_CLASSES = 2


        in_layer = Input(shape=(1, 1, input_size,),name='input_1')
        x = Dense(FC_SIZE, activation='relu', input_dim=(1, 1, input_size,),name='dense_1')(in_layer)
        x = Flatten(name='flatten_1')(x)
        preds = Dense(NUM_CLASSES, activation='softmax', input_dim=FC_SIZE, name='classifier_output')(x)
        
        model = Model(inputs = in_layer, outputs = preds)
        
        return model


    def construct_model(
            self,
            quantized,
            saved_model_dir = None,
            starting_weights_directory = None,
            is_frozen = False,
            is_training = True,
            size = 224,
            weights_file = None,
            architecture = 'Resnet50',
            ):
        
        Architecture = self.arch_dict[architecture] if not(quantized) else self.arch_dict_quantized[architecture]

        self.print_and_log(
            'Constructing model with architecture {0}{1}'
            .format(
                architecture,
                ', quantized' if quantized else ''
                )
            )

        # # Convert images to 3D tensors [width,height,channel]
        # in_images, image_tensors = preprocess_images(size=size)
        self.print_and_log('Scaling input tensor from size {0} to {1}'.format(size, self.arch_dict_inputsize[architecture]))
        in_images, image_tensors = self.preprocess_images(size, self.arch_dict_inputsize[architecture])

        # Construct featurizer using quantized or unquantized Resnet50 model
        if not quantized:
            featurizer = Architecture(saved_model_dir)
        else:
            featurizer = Architecture(
                saved_model_dir,
                is_frozen = is_frozen,
                custom_weights_directory = starting_weights_directory
                )
        
        features = featurizer.import_graph_def(input_tensor=image_tensors, is_training=is_training)
        
        # Construct classifier
        with tf.name_scope('classifier'):
            classifier = self.construct_classifier(architecture)
            preds = classifier(features)
        
        # Initialize weights
        sess = tf.get_default_session()
        tf.global_variables_initializer().run()
        
        if not is_frozen:
            print('Restoring weights from featurizer into session')
            featurizer.restore_weights(sess)
        
        if not(weights_file is None):
            print("loading classifier weights from", weights_file)
            classifier.load_weights(weights_file)
        elif starting_weights_directory is not None:
            print("loading classifier weights from", starting_weights_directory+'/class_weights_best.h5')
            classifier.load_weights(starting_weights_directory+'/class_weights_best.h5')
            
        return in_images, image_tensors, features, preds, featurizer, classifier 


    def train_architecture(self, architecture, dry=False):
        self.train(architecture, dry=dry)


    def train(self, architecture='Resnet50', dry=False):
        self.tags.append(architecture)
        self.set_new_training_dir(dry=dry)

        self.print_and_log('Start training', strftime('%y-%m-%d %H:%M:%S'))
        self.print_and_log('Architecture:', architecture)
        self.print_and_log('training_dir:', self.training_dir)

        # Launch the training
        tf.reset_default_graph()
        sess = tf.Session(graph=tf.get_default_graph())

        with sess.as_default():

            in_images, image_tensors, \
            features, preds, \
            featurizer, classifier = self.construct_model(
                quantized = False,
                # starting_weights_directory = custom_weights_dir,
                starting_weights_directory = None,
                saved_model_dir = self.saved_model_dir,
                is_training = True,
                size = self.data_size,
                architecture = architecture
                )

            self.print_and_log('featurizer:', featurizer)
            self.print_and_log('classifier:', classifier)

            # It's necessary to specify global (all) variables when using 
            # the saver in this instance.
            # Since we are using batch norm layers, whose variables aren't 
            # saved by default, we include them this way.

            saver = tf.train.Saver(tf.global_variables(), max_to_keep = 100)

            # print('saver:', saver)
            # return
            
            if dry:
                return

            loss_over_epoch, accuracy_over_epoch, \
            auc_over_epoch, val_loss_over_epoch, \
            val_accuracy_over_epoch, val_auc_over_epoch = \
                self.train_model(
                    preds,
                    in_images,
                    self.data.train_files,
                    self.data.val_files,
                    # is_retrain      = True,
                    is_retrain      = False,
                    train_epoch     = 4,
                    classifier      = classifier,
                    saver           = saver,
                    checkpoint_path = self.custom_weights_dir,
                    chunk_size      = 64
                    ) 

            _, _, features, preds, featurizer, classifier = \
                self.construct_model(
                    quantized = False,
                    saved_model_dir = self.saved_model_dir,
                    starting_weights_directory = self.custom_weights_dir,
                    is_training = False,
                    size = 64,
                    architecture = architecture
                    )

            loss, accuracy, auc, preds_test, test_labels = test_model(
                preds,
                in_images,
                self.data.test_files
                )
            

    def finetune_quantized(self, training_dir = None, architecture='Resnet50'):
        if training_dir is None:
            self.set_training_dir_latest()
        else:
            self.set_training_dir(training_dir)

        # print(self.training_dir, self.custom_weights_dir, self.custom_weights_dir_q, self.saved_model_dir, self.results_dir, self.logfile)
        # return

        tf.reset_default_graph()
        sess = tf.Session(graph=tf.get_default_graph())

        num_epoch_finetune = 5

        with sess.as_default():
            self.print_and_log("Fine-tuning model with quantization")

            in_images, image_tensors, features, preds, quantized_featurizer, classifier = self.construct_model(
                quantized = True, 
                saved_model_dir = self.saved_model_dir, 
                starting_weights_directory = self.custom_weights_dir, 
                is_training = True, 
                size = self.data_size,
                architecture = architecture
                )

            saver = tf.train.Saver(tf.global_variables(), max_to_keep = 100)

            loss_over_epoch_ft, accuracy_over_epoch_ft, auc_over_epoch_ft, val_loss_over_epoch_ft, val_accuracy_over_epoch_ft, val_auc_over_epoch_ft = \
                self.train_model(
                    preds, in_images,
                    self.data.train_files, self.data.val_files,
                    is_retrain = True,
                    train_epoch = num_epoch_finetune, 
                    classifier = classifier,
                    saver = saver, 
                    checkpoint_path = self.custom_weights_dir_q,
                    chunk_size = 32
                    )



    def test_trained_model(self, training_dir=None):

        if training_dir is None:
            self.set_training_dir_latest()
        else:
            self.set_training_dir(training_dir)

        tf.reset_default_graph()
        sess = tf.Session(graph=tf.get_default_graph())

        print('Start testing', strftime('%y-%m-%d %H:%M:%S'))
        print('training_dir:', self.training_dir)
        print('custom_weights_dir:', self.custom_weights_dir)

        with sess.as_default():

            in_images, image_tensors, \
            features, preds, \
            featurizer, classifier = self.construct_model(
                quantized = False,
                starting_weights_directory = self.custom_weights_dir,
                # starting_weights_directory = None,
                saved_model_dir = self.saved_model_dir,
                # weights_file = '../../training8/custom-weights/class_weights-4.h5',
                is_training = True,
                size = self.data_size
                )

            print('featurizer:', featurizer)
            print('classifier:', classifier)

            loss, accuracy, auc, preds_test, test_labels = self.test_model(
                preds,
                in_images,
                self.data.test_files[:1]
                )


    def test_my_model(self):
        # resdir = '/home/thomas/acceltraining/training14/custom-weights'
        resdir = '/home/thomas/acceltraining/training14/custom-weights-quantized'
        modeldir = '/home/thomas/acceltraining/training14/models'

        tf.reset_default_graph()
        sess = tf.Session(graph=tf.get_default_graph())

        with sess.as_default():

            in_images, image_tensors, \
            features, preds, \
            featurizer, classifier = construct_model(
                quantized = True,
                # quantized = False,
                starting_weights_directory = resdir,
                # weights_file = osp.join(resdir, 'class_weights.h5'),
                saved_model_dir = modeldir,
                # is_frozen = True,
                # is_training = True,
                size = self.data_size
                )

            print('featurizer:', featurizer)
            print('classifier:', classifier)

            loss, accuracy, auc, preds_test, test_labels = self.test_model(
                preds,
                in_images,
                self.data.test_files[:3]
                )


    def test_javiers_model(self):
        resdir = '/home/thomas/acceltraining/weights-floatingpoint-224x224-fixval-best'

        tf.reset_default_graph()
        sess = tf.Session(graph=tf.get_default_graph())

        with sess.as_default():

            in_images, image_tensors, \
            features, preds, \
            featurizer, classifier = construct_model(
                quantized = True,
                starting_weights_directory = resdir,
                # weights_file = osp.join(resdir, 'class_weights.h5'),
                saved_model_dir = resdir,
                # is_frozen = True,
                # is_training = True,
                size = self.data_size
                )

            print('featurizer:', featurizer)
            print('classifier:', classifier)

            loss, accuracy, auc, preds_test, test_labels = self.test_model(
                preds,
                in_images,
                self.data.test_files[:1]
                )


    def test_javiers_model_quantized(self):
        resdir = '/home/thomas/acceltraining/weights-quantized-224x224-fixval-best-final'

        tf.reset_default_graph()
        sess = tf.Session(graph=tf.get_default_graph())

        with sess.as_default():

            in_images, image_tensors, \
            features, preds, \
            featurizer, classifier = construct_model(
                quantized = True,
                starting_weights_directory = resdir,
                # weights_file = osp.join(resdir, 'class_weights_best.h5'),
                saved_model_dir = resdir,
                # is_frozen = True,
                # is_training = True,
                size = self.data_size
                )

            print('featurizer:', featurizer)
            print('classifier:', classifier)

            loss, accuracy, auc, preds_test, test_labels = self.test_model(
                preds,
                in_images,
                self.data.test_files[:1]
                )



    def test_model(self, preds, in_images, test_files, chunk_size=64, shuffle=True):
        """Test the model"""
        
        in_labels = tf.placeholder(tf.float32, shape=(None, 2))
        
        cross_entropy = tf.reduce_mean(binary_crossentropy(in_labels, preds))
        accuracy = tf.reduce_mean(categorical_accuracy(in_labels, preds))
        auc = tf.metrics.auc(tf.cast(in_labels, tf.bool), preds)
       
        n_test_events = count_events(test_files)
        chunk_num = int(n_test_events/chunk_size)+1
        preds_all = []
        label_all = []
        
        sess = tf.get_default_session()
        sess.run(tf.local_variables_initializer())
        
        avg_accuracy = 0
        avg_auc = 0
        avg_test_loss = 0
        is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
        for img_chunk, label_chunk, real_chunk_size in tqdm(chunks(test_files, chunk_size, shuffle=shuffle),total=chunk_num):
            test_loss, accuracy_result, auc_result, preds_result = sess.run([cross_entropy, accuracy, auc, preds],
                            feed_dict={in_images: img_chunk,
                                       in_labels: label_chunk,
                                       K.learning_phase(): 0,
                                       is_training: False})
            avg_test_loss += test_loss * real_chunk_size / n_test_events
            avg_accuracy += accuracy_result * real_chunk_size / n_test_events
            avg_auc += auc_result[0]  * real_chunk_size / n_test_events 
            preds_all.extend(preds_result)
            label_all.extend(label_chunk)
        
        self.print_and_log("test_loss = ", "{:.3f}".format(avg_test_loss))
        self.print_and_log("Test Accuracy:", "{:.3f}".format(avg_accuracy), ", Area under ROC curve:", "{:.3f}".format(avg_auc))
        
        return avg_test_loss, avg_accuracy, avg_auc, np.asarray(preds_all).reshape(n_test_events,2), np.asarray(label_all).reshape(n_test_events,2)


    def train_model(
            self,
            preds,
            in_images,
            train_files,
            val_files,
            is_retrain      = False,
            train_epoch     = 10,
            classifier      = None,
            saver           = None,
            checkpoint_path = None,
            chunk_size      = 64
            ): 

        learning_rate = 0.0001 if is_retrain else 0.001

        # Specify the loss function
        in_labels = tf.placeholder(tf.float32, shape=(None, 2))   
        with tf.name_scope('xent'):
            cross_entropy = tf.reduce_mean(binary_crossentropy(in_labels, preds))
            
        with tf.name_scope('train'):  
            #optimizer_def = tf.train.GradientDescentOptimizer(learning_rate)
            #momentum = 0.9
            #optimizer_def = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
            optimizer_def = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = optimizer_def.minimize(cross_entropy)

        with tf.name_scope('metrics'):
            accuracy = tf.reduce_mean(categorical_accuracy(in_labels, preds))
            auc = tf.metrics.auc(tf.cast(in_labels, tf.bool), preds)
        
        sess = tf.get_default_session()
        # to re-initialize all variables 
        #sess.run(tf.group(tf.local_variables_initializer(),tf.global_variables_initializer()))
        # to re-initialize just local variables + optimizer variables
        sess.run(
            tf.group(
                tf.local_variables_initializer(),
                tf.variables_initializer(optimizer_def.variables())
                )
            )
        
        # Create a summary to monitor cross_entropy loss
        tf.summary.scalar("loss", cross_entropy)
        # Create a summary to monitor accuracy 
        tf.summary.scalar("accuracy", accuracy)
        # Create a summary to monitor auc 
        tf.summary.scalar("auc", auc[0])
        # create a summary to look at input images
        tf.summary.image("images", in_images, 3)
        
        # Create summaries to visualize batchnorm variables
        tensors_per_node = [node.values() for node in tf.get_default_graph().get_operations()]
        bn_tensors = [tensor for tensors in tensors_per_node for tensor in tensors if 'BatchNorm/moving_mean:0' in tensor.name or 'BatchNorm/moving_variance:0' in tensor.name or  'BatchNorm/gamma:0' in tensor.name or 'BatchNorm/beta:0' in tensor.name]
        for var in bn_tensors:
            tf.summary.histogram(var.name.replace(':','_'), var)
            
        #grads = tf.gradients(cross_entropy, tf.trainable_variables())
        #grads = list(zip(grads, tf.trainable_variables()))
        
        # Summarize all gradients
        #for grad, var in grads:
        #    tf.summary.histogram(var.name.replace(':','_') + '/gradient', grad)

        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        n_train_events = count_events(train_files)
        train_chunk_num = int(n_train_events / chunk_size)+1
        
        train_writer = tf.summary.FileWriter(checkpoint_path + '/logs/train', sess.graph)
        val_writer = tf.summary.FileWriter(checkpoint_path + '/logs/val', sess.graph)

        loss_over_epoch = []
        accuracy_over_epoch = []
        auc_over_epoch = []
        
        val_loss_over_epoch = []
        val_accuracy_over_epoch = []
        val_auc_over_epoch = []
        best_val_loss = 999999
        
        is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')


        for epoch in range(train_epoch):

            avg_loss = 0
            avg_accuracy = 0
            avg_auc = 0
            preds_temp = []
            label_temp = []
            i = 0

            for img_chunk, label_chunk, real_chunk_size in tqdm(chunks(train_files, chunk_size),total=train_chunk_num):
                _, loss, summary, accuracy_result, auc_result = sess.run([
                        optimizer, 
                        cross_entropy, 
                        merged_summary_op,
                        accuracy, auc
                        ],
                    feed_dict = {
                        in_images: img_chunk,
                        in_labels: label_chunk,
                        K.learning_phase(): 1,
                        is_training: True
                    })

                avg_loss += loss * real_chunk_size / n_train_events
                avg_accuracy += accuracy_result * real_chunk_size / n_train_events
                avg_auc += auc_result[0] * real_chunk_size / n_train_events
                train_writer.add_summary(summary, epoch * train_chunk_num + i)
                i += 1
            
            self.print_and_log("Epoch:", (epoch + 1), "loss = ", "{:.3f}".format(avg_loss))
            self.print_and_log("Training Accuracy:", "{:.3f}".format(avg_accuracy), ", Area under ROC curve:", "{:.3f}".format(avg_auc))

            loss_over_epoch.append(avg_loss)
            accuracy_over_epoch.append(avg_accuracy)
            auc_over_epoch.append(avg_auc)

            n_val_events = count_events(val_files)
            val_chunk_num = int(n_val_events / chunk_size)+1
            
            avg_val_loss = 0
            avg_val_accuracy = 0
            avg_val_auc = 0
            i = 0
            
            for img_chunk, label_chunk, real_chunk_size in tqdm(chunks(val_files, chunk_size),total=val_chunk_num):

                val_loss, val_accuracy_result, val_auc_result, summary = sess.run(
                    [cross_entropy, accuracy, auc, merged_summary_op],
                    feed_dict = {
                        in_images: img_chunk,
                        in_labels: label_chunk,
                        K.learning_phase(): 0,
                        is_training: False
                        })

                avg_val_loss += val_loss * real_chunk_size / n_val_events
                avg_val_accuracy += val_accuracy_result * real_chunk_size / n_val_events
                avg_val_auc += val_auc_result[0] * real_chunk_size / n_val_events
                val_writer.add_summary(summary, epoch * val_chunk_num + i)
                i += 1
                
            self.print_and_log("Epoch:", (epoch + 1), "val_loss = ", "{:.3f}".format(avg_val_loss))
            self.print_and_log("Validation Accuracy:", "{:.3f}".format(avg_val_accuracy), ", Area under ROC curve:", "{:.3f}".format(avg_val_auc))
        
            val_loss_over_epoch.append(avg_val_loss)
            val_accuracy_over_epoch.append(avg_val_accuracy)
            val_auc_over_epoch.append(avg_val_auc)
            
            if saver is not None and checkpoint_path is not None and classifier is not None:
                saver.save(sess, checkpoint_path+'/resnet50_bw', write_meta_graph=True, global_step = epoch)
                saver.save(sess, checkpoint_path+'/resnet50_bw', write_meta_graph=True)
                classifier.save_weights(checkpoint_path+'/class_weights-%s.h5'%epoch)
                classifier.save(checkpoint_path+'/class_model-%s.h5'%epoch)
                classifier.save_weights(checkpoint_path+'/class_weights.h5')
                classifier.save(checkpoint_path+'/class_model.h5')
                if avg_val_loss < best_val_loss:
                    self.print_and_log("new best model: epoch {0}".format(epoch))
                    best_val_loss = avg_val_loss
                    saver.save(sess, checkpoint_path+'/resnet50_bw_best', write_meta_graph=True)
                    classifier.save_weights(checkpoint_path+'/class_weights_best.h5')
                    classifier.save(checkpoint_path+'/class_model_best.h5')
                    
            
        return loss_over_epoch, accuracy_over_epoch, auc_over_epoch, val_loss_over_epoch, val_accuracy_over_epoch, val_auc_over_epoch


# for convenience in interpreter
t = Trainer()
def get_t():
    return t
