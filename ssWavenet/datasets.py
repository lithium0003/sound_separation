import numpy as np
import soundfile as sf
import concurrent.futures

def wav_to_float(x):

    try:
        max_value = np.iinfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    except:
        max_value = np.finfo(x.dtype).max
        min_value = np.finfo(x.dtype).min
    x = x.astype('float64', casting='safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    return x

def read_wav(filename):
    audio_signal, sample_rate = sf.read(filename)

    if audio_signal.dtype != 'float64':
        audio_signal = wav_to_float(audio_signal)

    amp = abs(np.max(audio_signal) - np.min(audio_signal))/2
    audio_signal = audio_signal / amp * 0.5

    return audio_signal, sample_rate

def write_wav(x, filename, sample_rate):

    if type(x) != np.ndarray:
        x = np.array(x)

    sf.write(filename, x, sample_rate)

class VoiceDataset():

    def __init__(self, config, model):
        self.model = model
        self.sequences = {'train':{},'val':{}}
        for i in range(model.num_sources):
            self.sequences['train']['vocal%d'%(i+1)] = []
            self.sequences['val']['vocal%d'%(i+1)] = []
        self.sequences['train']['mixture'] = []
        self.sequences['val']['mixture'] = []
        
        self.filepath = config['dataset']
        self.batch_size = config['training']['batch_size']

    def load_dataset(self):
        self.load_songs()
        return self

    def load_songs(self):

        for dataset in ['train', 'val']:
            
            conditions = ['vocal%d'%(i+1) for i in range(self.model.num_sources)]
            conditions.append('mixture')
            for condition in conditions:
                
                for filename in self.filepath[dataset][condition]:
                    sequence, sample_rate = read_wav(filename)
                    self.sequences[dataset][condition].append(sequence)

    def get_next(self, dataset):
        sample_i = np.random.randint(0, len(self.sequences[dataset]['mixture']))
        mixture = self.sequences[dataset]['mixture'][sample_i]
        offset_1 = np.squeeze(np.random.randint(0, len(mixture) - self.model.input_length + 1, 1))

        mixture_fragment = mixture[offset_1:offset_1 + self.model.input_length]
        batch_inputs = mixture_fragment

        batch_outputs = []
        for i in range(self.model.num_sources):
            vocal = self.sequences[dataset]['vocal%d'%(i+1)][sample_i]
            vocal_fragment = vocal[offset_1:offset_1 + self.model.input_length]
            batch_outputs.append(vocal_fragment[self.model.get_padded_target_field_indices()])

        return batch_inputs, batch_outputs

    def get_random_batch_generator(self, dataset):

        if dataset not in ['train', 'val']:
            raise ValueError("Argument SET must be either 'train' or 'val'")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                batch_outputs = {}
                batch_inputs = []
                for i in range(self.model.num_sources):
                    batch_outputs['data_output%d'%(i+1)] = []

                outs = executor.map(lambda x: self.get_next(dataset), range(self.batch_size))
                for indata, outdata in outs:
                    batch_inputs.append(indata)
                    for i, chdata in enumerate(outdata):
                        batch_outputs['data_output%d'%(i+1)].append(chdata)

                batch_inputs = np.array(batch_inputs, dtype='float32')
                for i in range(self.model.num_sources):
                    batch_outputs['data_output%d'%(i+1)] = np.array(batch_outputs['data_output%d'%(i+1)], dtype='float32')

                batch = {'data_input': batch_inputs}, batch_outputs
                yield batch
