import hydra
from hydra.utils import instantiate

from magneton.config import PipelineConfig
from magneton.constants import name_to_stage, PipelineStage
from magneton.pipeline import EmbeddingPipeline

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: PipelineConfig) -> None:
    """Main entry point for the protein embedding pipeline"""
    cfg = instantiate(cfg)
    stages = sorted([name_to_stage[x] for x in cfg.stages])
    pipeline = EmbeddingPipeline(cfg)

    for stage in stages:
        if stage == PipelineStage.EMBED:
            pipeline.run_embedding()
        elif stage == PipelineStage.TRAIN:
            pipeline.run_training()
        elif stage == PipelineStage.VISUALIZE:
            pipeline.run_visualization()
        else:
            raise ValueError(f"Unknown stage: {stage}")
            # Default: run full pipeline
            # pipeline.run()


if __name__ == "__main__":
    main()