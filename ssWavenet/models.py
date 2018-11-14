import os
import re
import util
import numpy as np
import tensorflow as tf

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

class AddSingletonDepth(tf.keras.layers.Layer):

    def call(self, x, mask=None):
        x = tf.expand_dims(x, -1)  # add a dimension of the right

        if tf.rank(x) == 4:
            return tf.transpose(x, prem=(0, 3, 1, 2))
        else:
            return x

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            return input_shape[0], 1, input_shape[1], input_shape[2]
        else:
            return input_shape[0], input_shape[1], 1


class Slice(tf.keras.layers.Layer):

    def __init__(self, selector=None, desired_output_shape=None, **kwargs):
        if selector[0] is None or type(selector[0]) is dict:
            s = selector
            if s[0] is None:
                s1 = Ellipsis
            else:
                s1 = slice(s[0]['start'], s[0]['stop'], s[0]['step'])
            if s[1] is None:
                s2 = Ellipsis
            else:
                s2 = slice(s[1]['start'], s[1]['stop'], s[1]['step'])
            selector = (s1, s2)
        self.selector = selector
        self.desired_output_shape = desired_output_shape
        super(Slice, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        if type(self.selector[0]) is slice:
            s1 = {'start': self.selector[0].start, 'stop': self.selector[0].stop, 'step': self.selector[0].step}
        if self.selector[0] is Ellipsis:
            s1 = None
        if type(self.selector[1]) is slice:
            s2 = {'start': self.selector[1].start, 'stop': self.selector[1].stop, 'step': self.selector[1].step}
        if self.selector[1] is Ellipsis:
            s2 = None
        config['selector'] = [s1, s2]
        config['desired_output_shape'] = self.desired_output_shape
        return config

    def call(self, x, mask=None):

        selector = self.selector
        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            x = tf.transpose(x, perm=[0, 2, 1])
            selector = (self.selector[1], self.selector[0])

        y = x[selector]

        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            y = tf.transpose(y, perm=[0, 2, 1])

        return y


    def compute_output_shape(self, input_shape):

        output_shape = (None,)
        for i, dim_length in enumerate(self.desired_output_shape):
            if dim_length == None:
                output_shape = output_shape + (input_shape[i+1],)
            else:
                output_shape = output_shape + (dim_length,)
        return output_shape

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

class SeparationWavenet():
    
    def __init__(self, config, useTPU=False):
        
        self.config = config
       
        self.useTPU = useTPU

        self.num_stacks = self.config['model']['num_stacks']
        if type(self.config['model']['dilations']) is int:
            self.dilations = [2 ** i for i in range(0, self.config['model']['dilations'] + 1)]
        elif type(self.config['model']['dilations']) is list:
            self.dilations = self.config['model']['dilations']
        self.receptive_field_length = util.compute_receptive_field_length(config['model']['num_stacks'], self.dilations,
                                                                          config['model']['filters']['lengths']['res'],
                                                                          1)
        self.target_field_length = config['model']['target_field_length']
        self.input_length = self.receptive_field_length + (self.target_field_length - 1)
        
        self.target_padding = config['model']['target_padding']
        self.padded_target_field_length = self.target_field_length + 2 * self.target_padding
        self.half_target_field_length = int(self.target_field_length / 2)
        self.half_receptive_field_length = int(self.receptive_field_length / 2)
        self.num_residual_blocks = len(self.dilations) * self.num_stacks
        self.samples_of_interest_indices = self.get_padded_target_field_indices()
        self.target_sample_indices = self.get_target_field_indices()

        self.num_sources = config['model']['num_sources']

        self.epoch_num = 0

        self.model = self.setup_model()

    def setup_model(self):

        self.checkpoints_path = os.path.join(self.config['training']['path'], 'checkpoints')

        model = self.build_model()

        if not self.useTPU:
            self.tb_path = os.path.join(self.config['tensorboard']['path'],"%d"%self.epoch_num)
            os.makedirs(self.tb_path, exist_ok=True)

        #model.summary()

        losses = {}
        loss_weights = []
        for i in range(self.num_sources):
            losses['data_output%d'%(i+1)] = self.get_out_loss()
            loss_weights.append(self.config['learning']['loss_weights'][i])

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

        model.compile(optimizer=self.get_optimizer(),
                      loss=losses,
                      loss_weights=loss_weights,
                      metrics=self.get_metrics())
        
        return model

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

    def get_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length,
                     target_sample_index + self.half_target_field_length + 1)

    def get_padded_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length - self.target_padding,
                     target_sample_index + self.half_target_field_length + self.target_padding + 1)

    def get_target_sample_index(self):
        return int(np.floor(self.input_length / 2.0))

    def get_metrics(self):

        return [
            tf.keras.metrics.mean_absolute_error,
        ]

    def build_model(self):

        data_input = tf.keras.layers.Input(
                shape=(self.input_length,),
                name='data_input')

        data_expanded = AddSingletonDepth()(data_input)

        with tf.name_scope('Initial') as scope:
            data_out = tf.keras.layers.Convolution1D(self.config['model']['filters']['depths']['res'],
                                                  self.config['model']['filters']['lengths']['res'], padding='same',
                                                  use_bias=False,
                                                  name='initial_causal_conv')(data_expanded)

        skip_connections = []
        res_block_i = 0
        for stack_i in range(self.num_stacks):
            layer_in_stack = 0
            with tf.name_scope('Stack%d'%stack_i) as scope:
                for dilation in self.dilations:
                    res_block_i += 1
                    data_out, skip_out = self.dilated_residual_block(data_out, res_block_i, layer_in_stack, dilation, stack_i)
                    if skip_out is not None:
                        skip_connections.append(skip_out)
                    layer_in_stack += 1

        with tf.name_scope('Add_Skip') as scope:
            data_out = tf.keras.layers.Add()(skip_connections)
            data_out = tf.keras.layers.LeakyReLU()(data_out)

        for i in range(len(self.config['model']['filters']['depths']['final'])):
            with tf.name_scope('Conv%i'%i) as scope:
                data_out = tf.keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][i],
                                                      self.config['model']['filters']['lengths']['final'][i],
                                                      padding='same',
                                                      use_bias=False)(data_out)

                data_out = tf.keras.layers.LeakyReLU()(data_out)

        with tf.name_scope('Final') as scope:
            data_out = tf.keras.layers.Convolution1D(self.num_sources, 1)(data_out)
            data_out = tf.keras.layers.Activation('tanh')(data_out)

        outputs = []
        for i in range(self.num_sources):
            with tf.name_scope('Output%d'%(i+1)) as scope:
                data_out_sep = Slice((Ellipsis, slice(i, i+1)), (self.padded_target_field_length, 1),
                                       name='slice_data_output%d'%(i+1))(data_out)
                data_out_sep = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, 2),
                                                    output_shape=lambda shape: (shape[0], shape[1]), 
                                                    name='data_output%d'%(i+1))(data_out_sep)
                outputs.append(data_out_sep)

        #return keras.engine.Model(inputs=[data_input], outputs=[data_out_vocals_1, data_out_vocals_2])
        return tf.keras.models.Model(inputs=[data_input], outputs=outputs)

    def dilated_residual_block(self, data_x, res_block_i, layer_i, dilation, stack_i):

        with tf.name_scope('Dilated%d'%dilation) as scope:
            original_x = data_x

            data_out = tf.keras.layers.Conv1D(2 * self.config['model']['filters']['depths']['res'],
                                                        self.config['model']['filters']['lengths']['res'],
                                                        dilation_rate=dilation, padding='same',
                                                        use_bias=False,
                                                        name='res_%d_dilated_conv_d%d_s%d' % (
                                                        res_block_i, dilation, stack_i),
                                                        activation=None)(data_x)

            data_out_1 = Slice(
                (Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
                (self.input_length, self.config['model']['filters']['depths']['res']),
                name='res_%d_data_slice_1_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

            data_out_2 = Slice(
                (Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                                 2 * self.config['model']['filters']['depths']['res'])),
                (self.input_length, self.config['model']['filters']['depths']['res']),
                name='res_%d_data_slice_2_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

            tanh_out = tf.keras.layers.Activation('tanh')(data_out_1)
            sigm_out = tf.keras.layers.Activation('sigmoid')(data_out_2)

            data_x = tf.keras.layers.Multiply(name='res_%d_gated_activation_%d_s%d' % (res_block_i, layer_i, stack_i))(
                [tanh_out, sigm_out])

            data_x = tf.keras.layers.Convolution1D(
                self.config['model']['filters']['depths']['res'] + self.config['model']['filters']['depths']['skip'], 1,
                padding='same', use_bias=False)(data_x)

            res_x = Slice((Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
                                 (self.input_length, self.config['model']['filters']['depths']['res']),
                                 name='res_%d_data_slice_3_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

            skip_x = Slice((Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                                                   self.config['model']['filters']['depths']['res'] +
                                                   self.config['model']['filters']['depths']['skip'])),
                                  (self.input_length, self.config['model']['filters']['depths']['skip']),
                                  name='res_%d_data_slice_4_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

            skip_x = Slice((slice(self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1),
                                   Ellipsis), (self.padded_target_field_length, self.config['model']['filters']['depths']['skip']),
                                  name='res_%d_keep_samples_of_interest_d%d_s%d' % (res_block_i, dilation, stack_i))(skip_x)

            res_x = tf.keras.layers.Add()([original_x, res_x])

            return res_x, skip_x

