#! /usr/bin/env python3

import os
import json
import tqdm
import numpy as np

import models
import datasets

def load_config(config_filepath):
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        print('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)

def get_valid_output_folder_path(outputs_folder_path):
    j = 1
    while True:
        output_folder_name = 'samples_%d' % j
        output_folder_path = os.path.join(outputs_folder_path, output_folder_name)
        if not os.path.isdir(output_folder_path):
            os.makedirs(output_folder_path, exist_ok=True)
            break
        j += 1
    return output_folder_path

def separate(config, targets):
    model = models.SeparationWavenet(config)
    batch_size = config['training']['batch_size']

    samples_folder_path = os.path.join(config['training']['path'], 'samples')
    output_path = get_valid_output_folder_path(samples_folder_path)

    for target in targets:
        output_filename_prefix = os.path.basename(target)
        output_filename_prefix = output_filename_prefix[0:-4]

        sequence, sample_rate = datasets.read_wav(target)
        mixture = sequence

        num_output_samples = mixture.shape[0] - (model.receptive_field_length - 1)
        num_fragments = int(np.ceil(num_output_samples / model.target_field_length))
        num_batches = int(np.ceil(num_fragments / batch_size))

        vocals_output = {}
        for i in range(model.num_sources):
            vocals_output[i] = []

        num_pad_values = 0
        fragment_i = 0

        for batch_i in tqdm.tqdm(range(0, num_batches)):

            if batch_i == num_batches - 1:  # If its the last batch
                batch_size = num_fragments - batch_i * batch_size

            input_batch = np.zeros((batch_size, model.input_length))

            # Assemble batch
            for batch_fragment_i in range(0, batch_size):

                if fragment_i + model.target_field_length > num_output_samples:
                    remainder = mixture[fragment_i:]
                    current_fragment = np.zeros((model.input_length,))
                    current_fragment[:remainder.shape[0]] = remainder
                    num_pad_values = model.input_length - remainder.shape[0]
                else:
                    current_fragment = mixture[fragment_i:fragment_i + model.input_length]

                input_batch[batch_fragment_i, :] = current_fragment
                fragment_i += model.target_field_length

            separated_output_fragments = model.separate_batch({'data_input': input_batch})

            for i in range(model.num_sources):
                vocals_output_fragment = separated_output_fragments[i]
                vocals_output_fragment = vocals_output_fragment[:,
                                         model.target_padding: model.target_padding + model.target_field_length]
                vocals_output_fragment = vocals_output_fragment.flatten().tolist()

                vocals_output[i] = vocals_output[i] + vocals_output_fragment
        
        for i in range(model.num_sources):
            vocals_output[i] = np.array(vocals_output[i])
            if num_pad_values != 0:
                vocals_output[i] = vocals_output[i][:-num_pad_values]


            output_vocals_filename = output_filename_prefix + '_vocals%d.wav'%(i+1)
            output_vocals_filepath = os.path.join(output_path, output_vocals_filename)

            datasets.write_wav(vocals_output[i], output_vocals_filepath, sample_rate)

def main():
    config = load_config('config.json')
    targets = ['sound/cat/all01.wav',
            'sound/cat/all02.wav',
            ]

    separate(config, targets)

if __name__ == "__main__":
    main()
