import os
import sys
import argparse

# Ensure we run from the project root containing net/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

try:
    from net.DL_config import Config
    from main_net import evaluate
except ModuleNotFoundError as e:
    print('Module import failed. Ensure you run this from the seizeit2 directory.')
    raise e


def load_config_from_experiment(exp_name: str) -> Config:
   
    cfg_dir = os.path.join('net', 'save_dir', 'models', exp_name, 'configs')
    cfg_path = os.path.join(cfg_dir, exp_name + '.cfg')
    config = Config()
    if os.path.exists(cfg_path):
        config.load_config(config_path=cfg_dir, config_name=exp_name + '.cfg')
        print(f'Loaded config from {cfg_path}')
    else:
        print(f'Config file not found at {cfg_path}. Using fallback defaults.')
        # Minimal fallback defaults (adjust if needed)
        config.data_path = '/Users/rosalouisemarker/Desktop/Digital Media Project/dataset'
        config.save_dir = 'net/save_dir'
        config.fs = 250
        config.CH = 2
        config.cross_validation = 'fixed'
        config.batch_size = 128
        config.frame = 2
        config.stride = 1
        config.stride_s = 0.5
        config.boundary = 0.5
        config.factor = 5
        config.dropoutRate = 0.5
        config.nb_epochs = 80
        config.l2 = 0.01
        config.lr = 0.01
        # Infer model/dataset from exp_name when possible
        config.model = 'ChronoNet' if 'ChronoNet' in exp_name else 'ChronoNet'
        config.dataset = 'SZ2'
        config.sample_type = 'subsample'
        # add_to_name derived from suffix after last underscore if present
        parts = exp_name.split('_')
        config.add_to_name = parts[-1] if len(parts) > 1 else 'run'
    return config


def ensure_predictions_exist(config, exp_name: str) -> bool:
    pred_dir = os.path.join(config.save_dir, 'predictions', exp_name)
    if not os.path.isdir(pred_dir):
        print(f'Prediction directory missing: {pred_dir}')
        print('Run training + prediction first (python main_net.py) or generate predictions separately.')
        return False
    
    files = [f for f in os.listdir(pred_dir) if f.endswith('_preds.h5')]
    if not files:
        print(f'No prediction files (*.h5) found in {pred_dir}.')
        return False
    
    print(f'Found {len(files)} prediction files in {pred_dir}.')
    return True


def main():
    parser = argparse.ArgumentParser(description = 'Run evaluation for a trained experiment.')
    parser.add_argument('-n', '--name', required = True, help = 'Experiment name')
    args = parser.parse_args()

    exp_name = args.name
    config = load_config_from_experiment(exp_name)

    # Ensure save_dir is set (loaded config might override)
    if not getattr(config, 'save_dir', None):
        config.save_dir = 'net/save_dir'

    if not ensure_predictions_exist(config, exp_name):
        sys.exit(1)

    print(f'Running evaluation for experiment: {exp_name}')
    evaluate(config)
    print('Evaluation complete. See results in:')
    print(os.path.join(config.save_dir, 'results'))


if __name__ == '__main__':
    main()
