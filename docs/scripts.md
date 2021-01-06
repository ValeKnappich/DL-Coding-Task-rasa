# Script overview
## cross_validation.py
Usage: cross_validation.py [OPTIONS]

  Perform a k-fold cross-validation on one or multiple configurations. The
  results are averaged across runs and saved to disk.

Options:
  -f, --folds INTEGER  Number of folds
  -d, --data TEXT      File or directory containing the training data
  --help               Show this message and exit.

## inference.py
Usage: inference.py [OPTIONS]

Options:
  -d, --data TEXT   Path to dev.json
  -o, --out TEXT    Path of the output json
  -m, --model TEXT  Path to the trained model, defaults to the latest model
  --help            Show this message and exit.

## train.py
Usage: train.py [OPTIONS]

Options:
  -c, --config TEXT  Name of the chosen config
  -d, --data TEXT    File or directory containing the training data
  --help             Show this message and exit.

## transform_data.py
Usage: transform_data.py [OPTIONS]

Options:
  -i, --input TEXT   Input file in json format
  -o, --output TEXT  Output file in yaml format
  -c, --check-only   Only check for trailing whitespaces and dots inside of
                     entities and not output yaml

  -s, --save-check   Save check results to disk
  --help             Show this message and exit.

## utils.py
Usage: utils.py [OPTIONS]

  Helper functions shared across scripts.

Options:
  --help  Show this message and exit.

