import hydra
from omegaconf import DictConfig
from .pipeline import EmbeddingPipeline

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the protein embedding pipeline"""
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