#!/bin/bash
#SBATCH --job-name=bias_identification
#SBATCH --ntasks=3
#SBATCH --nodes=3
#SBATCH --mincpus=1
#SBATCH --gres=gpu:4
#SBATCH --mem=150G
#SBATCH --partition=a40
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --output=job_%x_%j.out
#SBATCH --error=job_%x_%j.err
#SBATCH --time 100:00:00

# these commands don't need to run for all workers, put them here
MAIN_HOST=`hostname -s`
# this is the current host
export MASTER_ADDR=$MAIN_HOST
# pick a random available port
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"

# NCCL options
# This is needed to print debug info from NCCL, can be removed if all goes well
export NCCL_DEBUG=INFO
# This is needed to avoid NCCL to use ifiniband, which the cluster does not have
export NCCL_IB_DISABLE=1
# This is to tell NCCL to use bond interface for network communication
if [[ "${SLURM_JOB_PARTITION}" == "t4v2" ]] || \
    [[ "${SLURM_JOB_PARTITION}" == "rtx6000" ]]; then
    echo export NCCL_SOCKET_IFNAME=bond0 on ${SLURM_JOB_PARTITION}
    export NCCL_SOCKET_IFNAME=bond0
fi

# note when number of tasks is greater than 1, srun is needed to launch
# all tasks with the same command
# you should also be careful about parameter expansion, make sure
# they are not expanded here

#mkdir -p workdir_${SLURM_JOB_ID}
#cp -r  workdir_${SLURM_JOB_ID}/
#cd  workdir_${SLURM_JOB_ID}

# Checkpointing, never forget to do this
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} checkpoints
touch checkpoints/DELAYPURGE

# this will execute "number of tasks" times in parallel, each with
# slightly different env variables for DDP training
# /opt/slurm/bin/srun --nodes=1 --mem=30G bash -c \
#     "python3 adv_graph_decodingtrust.py --setting adv --temp 0.1 >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

# /opt/slurm/bin/srun --nodes=1 --mem=30G bash -c \
#     "python3 adv_graph.py --setting adv --temp 0.5 >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

# /opt/slurm/bin/srun --nodes=1 --mem=30G bash -c \
#     "python3 adv_graph.py --setting adv --temp 1 >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"
# /opt/slurm/bin/srun --nodes=1 --mem=30G bash -c \
#     "python3 adv_graph.py --setting adv --explanation --temp 1 >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

# /opt/slurm/bin/srun --nodes=1 --mem=30G bash -c \
#     "python3 adv_graph.py --setting adv --explanation --temp 0.5 >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

# /opt/slurm/bin/srun --nodes=1 --mem=30G bash -c \
#     "python3 adv_graph.py --setting adv --explanation --temp 0.1 >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"


# /opt/slurm/bin/srun --nodes=1 --mem=50G bash -c \
#     "python3 adv_graph_decodingtrust.py --temp 0.3 --load_cache >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1" & 
# /opt/slurm/bin/srun --nodes=1 --mem=50G bash -c \
#     "python3 adv_graph_decodingtrust.py --explanation --temp 0.3 --load_cache >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1" & 
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --temp 0.1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --explanation --temp 0.1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1" 
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --temp 0.5 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --explanation --prompt_file decoding_trust_top_3.csv --temp 0.5 --run \${SLURM_PROCID} >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --temp 1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --explanation --temp 1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"


/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --temp 0.1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --explanation --temp 0.1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1" 
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --temp 0.5 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"


/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --explanation --prompt_file decoding_trust_top_3.csv --temp 0.5 --run \${SLURM_PROCID} >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --temp 1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --explanation --temp 1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"


/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py  --model_name /model-weights/Llama-2-7b-hf/ --temp 0.1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1" 
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --model_name /model-weights/Llama-2-7b-hf/ --explanation --temp 0.1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1" 
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --model_name /model-weights/Llama-2-7b-hf/ --temp 0.5 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --model_name /model-weights/Llama-2-7b-hf/ --explanation --prompt_file decoding_trust_top_3.csv --temp 0.5 --run \${SLURM_PROCID} >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --model_name /model-weights/Llama-2-7b-hf/ --temp 1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --model_name /model-weights/Llama-2-7b-hf/ --explanation --temp 1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1" 

    /opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Llama-2-7b-hf/ --temp 0.1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Llama-2-7b-hf/ --explanation --temp 0.1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1" 
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Llama-2-7b-hf/ --temp 0.5 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"


/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Llama-2-7b-hf/ --explanation --prompt_file decoding_trust_top_3.csv --temp 0.5 --run \${SLURM_PROCID} >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Llama-2-7b-hf/ --temp 1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --setting adv --model_name /model-weights/Llama-2-7b-hf/ --explanation --temp 1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"


/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py  --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --temp 0.1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1" 
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --explanation --temp 0.1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1" 
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --temp 0.5 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --explanation --prompt_file decoding_trust_top_3.csv --temp 0.5 --run \${SLURM_PROCID} >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"
/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --temp 1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

/opt/slurm/bin/srun --nodes=3 --mem=50G bash -c \
    "python3 adv_graph_decodingtrust.py --model_name /model-weights/Mistral-7B-Instruct-v0.1/ --explanation --temp 1 --prompt_file decoding_trust_top_3.csv --run \${SLURM_PROCID}>> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1" 