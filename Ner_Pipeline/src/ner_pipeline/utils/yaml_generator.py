
from pathlib import Path
from typing import Union, Dict
from argparse import ArgumentParser
from omegaconf import OmegaConf

def sample_yaml_generator(path: Union[str,Path], config: Dict):
    path = Path(path)
    path.mkdir(exist_ok=True) #if directory doesn't already exist
    conf = OmegaConf.create(config)
    name = conf.get("name", "example")
    version = conf.get("version_name", None)
    file_name = f"{name}_{version}.yaml" if version else f"{name}.yaml"
    full_path = path / file_name

    
    OmegaConf.save(config=conf, f=full_path)
    return full_path

if __name__ == "__main__":
    parser = ArgumentParser(description="Flexible YAML Generator from CLI")
    parser.add_argument("-o", "--save_path", type=str, required=True, help="Output directory to save generated YAML file")
    

    args, res = parser.parse_known_args()

    # Create OmegaConf from dotlist
    config = OmegaConf.from_dotlist(res)

    # Generate YAML
    yaml_path = sample_yaml_generator(path=args.save_path, config=config)
    print(f"Generated YAML file saved at: {yaml_path}")