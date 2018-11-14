#!/usr/bin/env python3
import os
import json

import models
import datasets

#from hyperdash import Experiment

def load_config(config_filepath):
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        print('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)

def training(config):
    #exp = Experiment('Wave-U-Net Separation')

    model = models.UnetAudioSeparator(config)
    dataset = datasets.VoiceDataset(config, model).load_dataset()

    num_steps_train = config['training']['num_steps_train']
    num_steps_val = config['training']['num_steps_test']
    train_set_generator = dataset.get_random_batch_generator('train')
    val_set_generator = dataset.get_random_batch_generator('val')

    model.fit_model(train_set_generator, num_steps_train, val_set_generator, num_steps_val,
                    config['training']['num_epochs'])

    #exp.end()

def main():
    config = load_config('config.json')
    training(config)

if __name__ == "__main__":
    main()
