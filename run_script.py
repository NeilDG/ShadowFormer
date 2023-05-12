#Script to use for running heavy training.

import os

def main():
    # os.system("python train.py --warmup --win_size 8 --train_ps 256 --resume")
    os.system("python test_custom.py --save_images --win_size 8 --train_ps 256")
    # os.system("shutdown /s /t 1")

if __name__ == "__main__":
    main()
