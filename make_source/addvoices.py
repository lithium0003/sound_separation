import soundfile as sf
import numpy as np
import sys

targets = sys.argv[1:]
vocals = []
for target in targets:
    vocal, samplefq = sf.read(target)
    vocals.append(vocal)

condition = [[]]
for i in range(len(vocals)):
    new_cond = []
    for c in condition:
        new_cond.append([True] + c)
    for c in condition:
        new_cond.append([False] + c)
    condition = new_cond

for c in condition:
    num_sources = sum(1 for i in c if i)
    name_part = ''.join([str(i+1) if b else str(0) for (i, b) in enumerate(c)])

    mix_name = 'mixture_'+name_part+'.wav'
    mixture = np.zeros(vocals[0].shape)
    silence = np.zeros(vocals[0].shape)
    for (i,b) in enumerate(c):
        part_name = 'part%d_'%(i+1) + name_part+'.wav'
        if b:
            mixture += vocals[i]/num_sources
            sf.write(part_name, vocals[i]/num_sources, samplefq)
        else:
            sf.write(part_name, silence, samplefq)
    sf.write(mix_name, mixture, samplefq)

