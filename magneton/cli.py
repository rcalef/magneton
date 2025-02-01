import click
import yaml
from pathlib import Path
from typing import Dict, Any
from .embedders.factory import EmbedderFactory
from .config.base_config import ESMConfig, GearNetConfig
from .pipeline import EmbeddingPipeline

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration"""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return config_dict

@click.group()
def cli():
    """Protein embedding pipeline CLI"""
    pass

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='embeddings', help='Output directory')
@click.option('--num-proteins', '-n', default=None, type=int, help='Number of proteins to process')
@click.option('--device', '-d', default='cuda', help='Device to use (cuda/cpu)')
def run_pipeline(config_path: str, output_dir: str, num_proteins: int, device: str):
    """Run the complete embedding pipeline"""
    # Load config
    config_dict = load_config(config_path)
    config_dict.update({
        'output_dir': output_dir,
        'num_proteins': num_proteins,
        'device': device
    })
    
    # Create pipeline
    pipeline = EmbeddingPipeline(config_dict)
    
    # Run pipeline stages
    pipeline.run()

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def embed(config_path: str):
    """Generate embeddings only"""
    config_dict = load_config(config_path)
    pipeline = EmbeddingPipeline(config_dict)
    pipeline.run_embedding()

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def train(config_path: str):
    """Train model only"""
    config_dict = load_config(config_path)
    pipeline = EmbeddingPipeline(config_dict)
    pipeline.run_training()

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
def visualize(config_path: str):
    """Generate visualizations only"""
    config_dict = load_config(config_path)
    pipeline = EmbeddingPipeline(config_dict)
    pipeline.run_visualization()

if __name__ == '__main__':
    cli() 