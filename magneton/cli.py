import hydra
from hydra.utils import instantiate

from magneton.config import PipelineConfig
from magneton.pipeline import EmbeddingPipeline


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: PipelineConfig) -> None:
    """Main entry point for the protein embedding pipeline"""
    cfg = instantiate(cfg)
    pipeline = EmbeddingPipeline(cfg)

    if cfg.stage == "train":
        pipeline.run_training()
    elif cfg.stage == "eval":
        pipeline.run_evals()
    else:
        raise ValueError(f"Unknown stage: {cfg.stage}")


if __name__ == "__main__":
    main()