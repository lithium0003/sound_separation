import os

def dir_contains_files(path):

    for f in os.listdir(path):
        if not f.startswith('.'):
            return True
    return False

def compute_receptive_field_length(stacks, dilations, filter_length, target_field_length):

    half_filter_length = int((filter_length-1)/2)
    length = 0
    for d in dilations:
        length += d*half_filter_length
    length = 2*length
    length = stacks * length
    length += target_field_length
    return length

