# sound_separation
This is the wave based sound source separation tests.

* ssWavenet/ is the source-separation-wavenet
* WaveUNet/ is Wave-U-Net

## Original
http://jordipons.me/apps/end-to-end-music-source-separation/

### Wave-U-Net based separation
https://github.com/f90/Wave-U-Net
### Wavenet based separation
https://github.com/francesclluis/source-separation-wavenet

## Instration
```
pip install tensorflow pysoundfile joblib tqdm
apt install libsndfile1
```

## Training
Prepare the traning source wave files and edit config.json.
``` config.json
       "dataset":{
                "train":{
                        "vocal1": [
                                "data/train/part1_12.wav"
                        ],
                        "vocal2": [
                                "data/train/part2_12.wav"
                        ],
                        "mixture": [
                                "data/train/mixture_12.wav"
                        ]
                },
                "val": {
                        "vocal1": [
                                "data/val/part1_12.wav"
                        ],
                        "vocal2": [
                                "data/val/part2_12.wav"
                        ],
                        "mixture": [
                                "data/val/mixture_12.wav"
                        ]
                }
        }
```
Run the main.py
```
./main.py
```

## Sound Separation
Edit sep_main.py file to specify targets.
``` sep_main.py
def main():
    config = load_config('config.json')

    targets = ['sound/cat/all01.wav',
            'sound/cat/all02.wav',
            ]

    separate(config, targets)
```
Run sep_main.py
```
./sep_main.py
```
