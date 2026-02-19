#!/usr/bin/env python

# Please do *not* edit this script. Changes will be discarded so that we can train the models consistently.

# This file contains functions for training your model for the Challenge. You can run it as follows:
#
#   python train_model.py -d data -m model -v
#
# where 'data' is a folder containing the Challenge data, 'model' is a folder for saving your model, and , and -v is an optional
# verbosity flag.

import argparse
import importlib
import sys

# Parse arguments.
def get_parser():
    description = 'Train the Challenge model.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--data_folder', type=str, required=True)
    parser.add_argument('-m', '--model_folder', type=str, required=True)
    parser.add_argument('-tm', '--team_module', type=str, default='team_code')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

# Run the code.
def run(args):
    # Dynamically import the team's code.
    team_code = importlib.import_module(args.team_module)
    train_model = team_code.train_model

    train_model(args.data_folder, args.model_folder, args.verbose)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))