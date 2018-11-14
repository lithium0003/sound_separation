import soundfile as sf
import numpy as np
import sys

target = sys.argv[1]
background = sys.argv[2]
result_wav = sys.argv[3]

mixture, samplefq = sf.read(target)
back, samplefq = sf.read(background)

corr = np.correlate(mixture[:samplefq*10,0], back[:samplefq*10,0], "full")
estimated_delay = corr.argmax() - (len(back[:samplefq*10,0]) - 1)
print("estimated delay is", estimated_delay)

shift = estimated_delay

length = min(mixture.shape[0], back.shape[0]) - abs(shift)
resultlen = back.shape[0]

if shift >= 0:
    mixture = mixture[shift:shift+length,:]
    back = back[:length,:]
else:
    ishift = -shift
    mixture = mixture[:length,:]
    back = back[ishift:ishift+length,:]

vocal = mixture - back

if shift > 0:
    vlen = vocal.shape[0]
    pad = np.zeros((resultlen - vlen, 2))
    vocal = np.concatenate((vocal, pad), axis=0)
else:
    vlen = vocal.shape[0]
    ishift = -shift
    pad = np.zeros((ishift, 2))
    vocal = np.concatenate((pad, vocal), axis=0)

sf.write(result_wav, vocal, samplefq)
