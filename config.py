from pathlib import Path
import yaml
import argparse
import sys

PROJECT_DIR = Path(__file__).resolve().parent

def get_path(relative_path: str) -> Path:
    return (
        Path(relative_path)
        if relative_path.startswith('/')
        else PROJECT_DIR / relative_path
    )

def load_config(argv=None):
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/genocae.yaml')
    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config_name = Path(args.filename).stem
    out_dir = get_path(config['logging_params']['save_dir']+config["data_params"]["data_name"]+"/"+config['model_params']['name']+"/"+config_name)
    return config, out_dir, config_name
