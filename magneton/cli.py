from pprint import pprint

import hydra
from hydra.utils import instantiate

from magneton.config import PipelineConfig
from magneton.pipeline import EmbeddingPipeline

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: PipelineConfig) -> None:
    """Main entry point for the protein embedding pipeline"""
    cfg = instantiate(cfg)
    pprint(cfg, compact=False)
    return
    pipeline = EmbeddingPipeline(cfg)

    # Determine which pipeline stage to run based on config
    if cfg.get('stage') == 'embed':
        pipeline.run_embedding()
    elif cfg.get('stage') == 'train':
        pipeline.run_training()
    elif cfg.get('stage') == 'visualize':
        pipeline.run_visualization()
    else:
        # Default: run full pipeline
        pipeline.run()

if __name__ == "__main__":
    main()