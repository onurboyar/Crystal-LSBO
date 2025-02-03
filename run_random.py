# main.py
import argparse
from random_experiment import run_experiment

def main():
    parser = argparse.ArgumentParser(description='Crystal-LCA-LSBO experiments')
    parser.add_argument('combined_z_size', type=int, help='Combined latent space dimension')
    parser.add_argument('ckpt_name', type=str, help='Checkpoint file name for the combined VAE')
    args = parser.parse_args()
    run_experiment(args.combined_z_size, args.ckpt_name)

if __name__ == "__main__":
    main()

