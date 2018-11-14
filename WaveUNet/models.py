import os
import numpy as np
import tensorflow as tf

import util

class Hyperdash(tf.keras.callbacks.Callback):

    def __init__(self, entries, exp):
        super(Hyperdash, self).__init__()
        self.entries = entries
        self.exp = exp

    def on_epoch_end(self, epoch, logs=None):
        for entrie in self.entries:
            log = logs.get(entrie)
            if log is not None:
                self.exp.metric(entrie, log)

class TPUCrossShardOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(self, opt=None, **kwargs):
        cls = getattr(tf.keras.optimizers, opt)
        opt = cls(**kwargs)
        super(TPUCrossShardOptimizer, self).__init__()
        self.opt = opt
        self.opt_configs = kwargs

    def get_updates(self, loss, params):
        return self.opt.get_updates(loss, params)

    def get_gradients(self, loss, params):
        grads = self.opt.get_gradients(loss, param)
        grads = [tf.contrib.tpu.cross_replica_sum(g) for g in grads]
        return grads

    def get_weights(self):
        return self.opt.get_weights()

    def set_weights(self, weights):
        self.opt.set_weights(weights)

    def get_config(self):
        config = {
                'opt': self.opt.__class__.__name__,
        }
        return dict(list(config.items()) + list(self.opt_configs.items()))

    def __getattr__(self, key):
        return getattr(self.opt, key)

class LearingUpsampler(tf.keras.layers.Layer):

    '''
    Implements a trainable upsampling layer by interpolation by a factor of two, from N samples to N*2 - 1.
    Interpolation of intermediate feature vectors v_1 and v_2 (of dimensionality F) is performed by
     w \cdot v_1 + (1-w) \cdot v_2, where \cdot is point-wise multiplication, and w an F-dimensional weight vector constrained to [0,1]
    :param input: Input features of shape [batch_size, 1, width, F]
    :param padding:
    :param level:
    :return:
    '''

    def __init__(self, level=0, **kwargs):
        self.level = level
        super(LearingUpsampler, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['level'] = self.level
        return config

    def build(self, input_shape):
        self.features = input_shape.as_list()[3]
        self.num_entries = input_shape.as_list()[2]
        # Create a trainable weight variable for this layer.
        self.weights0 = self.add_weight(name='interp_%d_weights'%self.level,
                                      shape=(self.features,),
                                      initializer='uniform',
                                      trainable=True)
        super(LearingUpsampler, self).build(input_shape)

    def call(self, x):
        weights_scaled = tf.sigmoid(self.weights0)
        counter_weights = 1.0 - weights_scaled
        conv_weights = tf.expand_dims(tf.concat([tf.expand_dims(tf.diag(weights_scaled), axis=0), 
                                                tf.expand_dims(tf.diag(counter_weights), axis=0)], axis=0), axis=0)
        intermediate_vals = tf.nn.conv2d(x, conv_weights, strides=[1,1,1,1], padding='VALID')
        intermediate_vals = tf.transpose(intermediate_vals, [2, 0, 1, 3])
        out = tf.transpose(x, [2, 0, 1, 3])
        out = tf.concat([out, intermediate_vals], axis=0)
        indices = list()
        num_outputs = (2*self.num_entries - 1)
        for idx in range(num_outputs):
            if idx % 2 == 0:
                indices.append(idx // 2)
            else:
                indices.append(self.num_entries + idx//2)

        out = tf.gather(out, indices)
        out = tf.transpose(out, [1, 2, 0, 3])
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_entries*2-1, input_shape[3])



class CropAndConcat(tf.keras.layers.Layer):

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('A `CropAndConcat` layer should be called '
                             'on exactly 2 inputs')
        self.input_shape0 = [s.as_list() for s in input_shape]
        super(CropAndConcat, self).build(input_shape)

    def call(self, x):
        diff = self.input_shape0[0][1] - self.input_shape0[1][1]
        if diff % 2 == 0:
            crop_start = diff // 2
            crop_end = diff - crop_start
        else:
            crop_start = diff // 2
            crop_end = crop_start + 2
        x1 = tf.keras.layers.Lambda(lambda y: y[:,crop_start:-crop_end,:],
                                output_shape=(self.input_shape0[1][1], self.input_shape0[0][2]))(x[0])
        x2 = x[1]
        return tf.keras.layers.concatenate([x1, x2], axis=2)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], input_shape[0][2]+input_shape[1][2])

class Crop(tf.keras.layers.Layer):

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('A `CropAndConcat` layer should be called '
                             'on exactly 2 inputs')
        self.input_shape0 = input_shape
        super(Crop, self).build(input_shape)

    def call(self, x):
        diff = self.input_shape0[0][1] - self.input_shape0[1][1]
        if diff % 2 == 0:
            crop_start = diff // 2
            crop_end = diff - crop_start
        else:
            crop_start = int(np.floor(diff / 2))
            crop_end = crop_start + 2
        return tf.keras.layers.Lambda(lambda y: y[:,crop_start:-crop_end,:],
                                  output_shape=(self.input_shape0[0][0], self.input_shape0[1][1], self.input_shape0[0][2]))(x[0])

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], input_shape[0][2])


class UnetAudioSeparator():
    
    def __init__(self, config, useTPU=False):
        
        self.config = config
        self.useTPU = useTPU
        
        self.num_layers = config["model"]["num_layers"]
        self.num_initial_filters = config["model"]["num_initial_filters"]
        self.filter_size = config["model"]["filter_size"]
        self.merge_filter_size = config["model"]["merge_filter_size"]

        self.num_samples = config["training"]["num_frames"]
        self.input_num_samples, self.output_num_samples = self.get_padding(self.num_samples)
        self.num_channels = 2
        self.num_sources = config["model"]["num_sources"]

        self.epoch_num = 0

        self.model = self.setup_model()

    def setup_model(self):

        self.checkpoints_path = os.path.join(self.config['training']['path'], 'checkpoints')

        model = self.build_model()

        if not self.useTPU:
            self.tb_path = os.path.join(self.config['tensorboard']['path'],"%d"%self.epoch_num)
            os.makedirs(self.tb_path, exist_ok=True)

        #model.summary()
        
        if self.useTPU:
            # TPU
            tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
            strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu_cluster_resolver)
            model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

        if os.path.exists(self.checkpoints_path) and util.dir_contains_files(self.checkpoints_path):
            checkpoints = os.listdir(self.checkpoints_path)
            checkpoints.sort(key=lambda x: os.stat(os.path.join(self.checkpoints_path, x)).st_mtime)
            last_checkpoint = checkpoints[-1]
            last_checkpoint_path = os.path.join(self.checkpoints_path, last_checkpoint)
            m = re.match(r'checkpoint\.([0-9]+)-', last_checkpoint)
            if m:
                self.epoch_num = int(m.group(1))
                print('Loading model from epoch: %d' % self.epoch_num)
                model.load_weights(last_checkpoint_path)
            else:
                self.epoch_num = 0
                print('Building new model...')
        else:
            print('Building new model...')
            os.makedirs(self.checkpoints_path, exist_ok=True)
            self.epoch_num = 0

        losses = {}
        for i in range(self.num_sources):
            losses['data_output%d'%(i+1)] = self.get_out_loss()

        model.compile(optimizer=self.get_optimizer(),
                      loss=losses, 
                      metrics=self.get_metrics())

        return model

    def get_padding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input and output, 
        so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape
        :return: Input_shape, output_shape, where each is a list [batch_size, time_steps, channels]
        '''

        # Check if desired shape is possible as output shape - go from output shape towards lowest-res feature map
        rem = float(shape) # Cut off batch size number and channel
        #rem = rem +  self.filter_size - 1
        for i in range(self.num_layers):
            rem = rem + self.merge_filter_size - 1
            rem = (rem + 1) / 2 # out = in + in - 1 <=> in = (out+1)/

        rem = np.ceil(rem)
        # Round resulting feature map dimensions up to nearest integer
        x = np.asarray(rem,dtype=np.int64)
        assert(x >= 2)

        input_shape = x

        # Extra conv
        input_shape = input_shape + self.filter_size - 1

        for i in range(self.num_layers):
            input_shape = 2*input_shape - 1 # Decimation
            input_shape = input_shape + self.filter_size - 1 # Conv

        n = input_shape
        for i in range(self.num_layers):
            n = n - self.filter_size + 1
            n = int(np.ceil(n / 2))
        n = n - self.filter_size + 1

        output_shape = n

        for i in range(self.num_layers):
            output_shape = 2*output_shape - 1 #Upsampling
            output_shape = output_shape - self.merge_filter_size + 1 # Conv

        return input_shape, output_shape

    def get_padded_target_field_indices(self):
        diff = self.input_num_samples - self.output_num_samples
        diff = diff // 2
        return range(diff,diff+self.output_num_samples)

    def get_optimizer(self):
        if self.useTPU:
            return TPUCrossShardOptimizer('Adam', lr=self.config['learning']['learning_rate'])
        else:
            return tf.keras.optimizers.Adam(lr=self.config['learning']['learning_rate'])

    def get_out_loss(self):

        return lambda y_true, y_pred: tf.keras.losses.mean_absolute_error(y_true, y_pred)

    def get_callbacks(self):
        
        callbacks =  [
            tf.keras.callbacks.EarlyStopping(patience=self.config['training']['early_stopping_patience'], verbose=1,
                                          monitor='loss'),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoints_path,
                                                         'checkpoint.{epoch:05d}-{val_loss:.3f}.h5'))
        ]
        if not self.useTPU:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.tb_path))

        return callbacks

    def fit_model(self, train_set_generator, num_steps_train, test_set_generator, num_steps_test, num_epochs, exp=None):

        print('Fitting model with %d training num steps and %d test num steps...' % (num_steps_train, num_steps_test))

        callbacks = self.get_callbacks()
        if exp is not None:
            callbacks.append(Hyperdash(['loss','val_loss'], exp))

        self.model.fit_generator(train_set_generator,
                                 num_steps_train,
                                 epochs=num_epochs,
                                 validation_data=test_set_generator,
                                 validation_steps=num_steps_test,
                                 callbacks=callbacks,
                                 initial_epoch=self.epoch_num)

    def separate_batch(self, inputs):
        return self.model.predict_on_batch(inputs)

    def get_metrics(self):

        return [
            tf.keras.metrics.mean_absolute_error,
        ]


    def build_model(self):

        data_input = tf.keras.layers.Input(
                shape=(self.input_num_samples, self.num_channels),
                name='data_input')

        enc_outputs = []

        data_out = data_input
        for i in range(self.num_layers):
            with tf.name_scope('DownConv%d'%i) as scope:
                data_out = tf.keras.layers.Conv1D(self.num_initial_filters + (self.num_initial_filters * i), 
                                                self.filter_size,
                                                padding='valid')(data_out)
                data_out = tf.keras.layers.LeakyReLU()(data_out)
                enc_outputs.append(data_out)
                data_out = tf.keras.layers.Lambda(lambda x: x[:,::2,:], 
                                                output_shape=lambda x: (x[0],int(x[1]/2),x[2]),
                                                name='decimate%d'%i)(data_out)

        with tf.name_scope('U_block') as scope:
            data_out = tf.keras.layers.Conv1D(self.num_initial_filters + (self.num_initial_filters * self.num_layers), 
                                            self.filter_size,
                                            padding='valid')(data_out)
            data_out = tf.keras.layers.LeakyReLU()(data_out)

        for i in range(self.num_layers):
            with tf.name_scope('UpConv%d'%i) as scope:
                data_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1), 
                                                output_shape=lambda x: (x[0], 1, x[1], x[2]))(data_out)
                data_out = LearingUpsampler(i, name='upsample%d'%i)(data_out)
                data_out = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), 
                                                output_shape=lambda x: (x[0], x[2], x[3]))(data_out)
                data_out = CropAndConcat()([enc_outputs[-i-1], data_out])
                data_out = tf.keras.layers.Conv1D(self.num_initial_filters + self.num_initial_filters * (self.num_layers - i - 1), 
                                                self.merge_filter_size,
                                                padding='valid')(data_out)
                data_out = tf.keras.layers.LeakyReLU()(data_out)

        
        with tf.name_scope('Final') as scope:
            data_out = CropAndConcat()([data_input, data_out])

        independent = self.config['model']['independent_output']
        if independent:
            with tf.name_scope('Output') as scope:
                outputs = []
                for i in range(self.num_sources):
                    out = tf.keras.layers.Conv1D(self.num_channels, 1, activation='tanh', name='data_output%d'%(i+1))(data_out)
                    outputs.append(out)

                outputs.append(out)
        else:
            with tf.name_scope('Output') as scope:
                cropped_input = Crop()([data_input, data_out])
                outputs = []
                for i in range(self.num_sources - 1):
                    out = tf.keras.layers.Conv1D(self.num_channels, 1, activation='tanh', name='data_output%d'%(i+1))(data_out)
                    outputs.append(out)
                    cropped_input = keras.layers.Subtract()([cropped_input, out])

                out = keras.layers.Lambda(lambda x: x, name='data_output%d'%self.num_sources)(cropped_input)
                outputs.append(out)

        return tf.keras.models.Model(inputs=[data_input], outputs=outputs)

