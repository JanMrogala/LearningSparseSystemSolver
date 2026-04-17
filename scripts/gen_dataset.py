import hydra
from omegaconf import DictConfig

from lsss.data.generate import generate_dataset


@hydra.main(config_path="../configs", config_name="mvp", version_base="1.3")
def main(cfg: DictConfig) -> None:
    generate_dataset(cfg)


if __name__ == "__main__":
    main()
