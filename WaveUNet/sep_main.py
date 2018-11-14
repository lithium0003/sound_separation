#!/usr/bin/env python3
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
    model = models.UnetAudioSeparator(config)
    batch_size = config['training']['batch_size']

    samples_folder_path = os.path.join(config['training']['path'], 'samples')
    output_path = get_valid_output_folder_path(samples_folder_path)

    for target in targets:
        output_filename_prefix = os.path.basename(target)
        output_filename_prefix = output_filename_prefix[0:-4]

        sequence, sample_rate = datasets.read_wav(target)
        mix_audio = sequence

        source_time_frames = mix_audio.shape[0]
        input_time_frames = model.input_num_samples
        output_time_frames = model.output_num_samples

        pad_time_frames = (input_time_frames - output_time_frames) // 2
        mix_audio_padded = np.pad(mix_audio, [(pad_time_frames, pad_time_frames), (0,0)], mode="constant", constant_values=0.0)

        source_preds = [np.zeros(mix_audio.shape, np.float32) for _ in range(model.num_sources)]

        batch_i = 0
        input_batch = []
        out_time = []

        for source_pos in tqdm.tqdm(range(0, source_time_frames, output_time_frames)):
            if source_pos + output_time_frames > source_time_frames:
                source_pos = source_time_frames - output_time_frames

            mix_part = mix_audio_padded[source_pos:source_pos + input_time_frames,:]
            out_time.append(range(source_pos, source_pos + output_time_frames))
            input_batch.append(mix_part)
            batch_i += 1
            
            if batch_i == batch_size:
                separated_output_fragments = model.separate_batch({'data_input': np.array(input_batch)})

                for b in range(batch_size):
                    for i in range(model.num_sources):
                        source_preds[i][out_time[b]] = separated_output_fragments[i][b, :, :]

                batch_i = 0
                input_batch = []
                out_time = []
        
        if batch_i != 0:
            for _ in range(batch_i, batch_size):
                input_batch.append(np.zeros([input_time_frames,2], np.float32))

            separated_output_fragments = model.separate_batch({'data_input': np.array(input_batch)})
            
            for b in range(batch_i):
                for i in range(model.num_sources):
                    source_preds[i][out_time[b]] = separated_output_fragments[i][b, :, :]


        for i in range(model.num_sources):
            output_vocals_filename = output_filename_prefix + '_vocal%d.wav'%(i+1)
            output_vocals_filepath = os.path.join(output_path, output_vocals_filename)

            datasets.write_wav(source_preds[i], output_vocals_filepath, sample_rate)

def main():
    config = load_config('config.json')

    targets = ['sound/cat/all01.wav',
            'sound/cat/all02.wav',
            ]

    separate(config, targets)

if __name__ == "__main__":
    main()
