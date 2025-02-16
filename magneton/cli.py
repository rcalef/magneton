from pprint import pprint

import hydra
from hydra.utils import instantiate

from magneton.config import PipelineConfig
from magneton.constants import name_to_stage
from magneton.pipeline import EmbeddingPipeline

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: PipelineConfig) -> None:
    """Main entry point for the protein embedding pipeline"""
    cfg = instantiate(cfg)
    pprint(cfg, compact=False)
    return
    stages = sorted([name_to_stage[x] for x in cfg.stages])
    pipeline = EmbeddingPipeline(cfg)

    for stage in stages:
        if stage == "embed":
            pipeline.run_embedding()
        elif stage == "train":
            pipeline.run_training()
        elif stage == "visualize":
            pipeline.run_visualization()
        else:
            raise ValueError(f"Unknown stage: {stage}")
            # Default: run full pipeline
            # pipeline.run()


if __name__ == "__main__":
    main()