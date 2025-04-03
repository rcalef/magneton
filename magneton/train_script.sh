#!/bin/bash
#SBATCH --job-name=multi-node-training
#SBATCH --output=multi_node_output.txt
#SBATCH --error=multi_node_error.txt
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=70GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=40GB
#SBATCH --exclude=node058,node055,node057,node056

# Load necessary modules (adjust for your cluster)
cd /net/vast-storage/scratch/vast/kellislab/artliang/magneton
source .venv/bin/activate

# uv pip list
# ls

# Get node list
NODELIST=$(scontrol show hostname $SLURM_NODELIST)
echo "Allocated nodes: $NODELIST"

# Extract GPU availability per node
for NODE in $NODELIST; do
    echo "Checking GPUs on $NODE..."
    ssh $NODE "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv"
done

# Run PyTorch Lightning in a distributed multi-node setting
srun torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$(scontrol show hostname $SLURM_NODELIST | head -n 1) \
    --master_port=12355 \
    magneton/cli.py +stages=["train"] &> debug.txt

# If it’s a PyTorch Lightning model, ensure Trainer(strategy="ddp") is used.
# You might need to enable torch.distributed.init_process_group() inside your script.