#!/bin/bash

#### Runs: train-bench.py
#### Benchmarks: Forward, Backward passes every 5 steps
####
#### Notes:  1. If you want to bechmark all the steps find and replace
####            `run_training ""` with `run_training "--logging-frequency 1"`

#SBATCH --account=a-large-sc
#SBATCH --time=00:45:00
#SBATCH --job-name=lsai-merger
#SBATCH --output=./logs/info/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
#SBATCH --partition=normal
#SBATCH --environment=/iopsstor/scratch/cscs/bkundu/lsai-merger/ngc_pt_jan.toml     # Vanilla 25.01 PyTorch NGC Image 
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs

# Exit immediately if a command exits with a non-zero status (good practice)
set -eo pipefail

echo "START TIME: $(date)"

MERGER_DIR="/iopsstor/scratch/cscs/$USER/lsai-merger"

cd $MERGER_DIR

if [ ! -d ".venv" ]; then
    python -m venv --system-site-packages .venv
fi

source .venv/bin/activate

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}    # or whatever module set
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

if pip show flash_attn_3 &> /dev/null; then
    echo "flash_attn_3 is already installed"
else
    echo "Installing flash_attn_3..."
    cd flash-attention/hopper
    MAX_JOBS=8 pip install -e .
    pip install torch-utils

    # Becnhmark
    python benchmark_mla_decode.py
    cd -
fi

export PYTHONPATH=$PWD/flash-attention/hopper

# nvidia-smi -i 0 --lock-gpu-clocks 1830,1830

# # Set up ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
FILE_SUFFIX="$SLURM_JOB_NAME-$SLURM_JOB_ID"

cd $MERGER_DIR/LSAI-Checkpointing  # TODO: do everything in MERGER_DIR

if [ ! -f "checkpointing/pccheck/libtest_ssd.so" ]; then
    ./install.sh
fi


get_srun_outfile() {
    local out_file_dir="$1"
    local out_file_prefix="$2"
    local out_file_suffix="$3"


    if [ -z "$out_file_suffix" ]; then
        TEMP_BENCH_NUMBER=1
        srun_outfile="$out_file_dir/$out_file_prefix-$TEMP_BENCH_NUMBER.out"
        while [ -f "$srun_outfile" ]; do
            TEMP_BENCH_NUMBER=$((TEMP_BENCH_NUMBER + 1))
            srun_outfile="$out_file_dir/$out_file_prefix-$TEMP_BENCH_NUMBER.out"
        done
    else
        srun_outfile="$out_file_dir/$out_file_prefix-$out_file_suffix.out"
    fi

    echo "$srun_outfile"
}

run_merger() {
    MODE="$1"
    SEQ_LEN="$2"
    TRAINING_ARGS="$3"

    [ -d "$MERGER_DIR/chkpts" ] || mkdir $MERGER_DIR/chkpts
    [ -d "$MERGER_DIR/results" ] || mkdir $MERGER_DIR/results

    SRUN_OUTFILE_SUFFIX="merger"
    CMD_PREFIX="numactl --membind=0-3"

    if [ "$MODE" == "base" ]; then
        TRAIN_FILE_NAME="training.train_checkp"
    elif [ "$MODE" == "pccheck" ]; then
        TRAIN_FILE_NAME="training.train_pccheck"
    else
        echo "Unknown mode: $MODE"
        exit 1
    fi

    bench_flash_attn() {
        local flash_attn=$1
        local out_file_prefix=""
        local srun_outfile=""

        if [ "$flash_attn" == "True" ]; then
            export USE_FLASH_ATTENTION="1"
            out_file_prefix="$MODE-flash"
            if [ "$MODE" == "base" ]; then
                if [ "$SEQ_LEN" == "2048" ]; then
                    time_out=105
                elif [ "$SEQ_LEN" == "1024" ]; then
                    time_out=105
                elif [ "$SEQ_LEN" == "512" ]; then
                    time_out=85
                elif [ "$SEQ_LEN" == "256" ]; then
                    time_out=60
                else
                    echo "Unknown SEQ_LEN: $SEQ_LEN"
                    exit 1
                fi
            elif [ "$MODE" == "pccheck" ]; then
                if [ "$SEQ_LEN" == "2048" ]; then
                    time_out=80
                elif [ "$SEQ_LEN" == "1024" ]; then
                    time_out=80
                elif [ "$SEQ_LEN" == "512" ]; then
                    time_out=60
                elif [ "$SEQ_LEN" == "256" ]; then
                    time_out=60
                else
                    echo "Unknown SEQ_LEN: $SEQ_LEN"
                    exit 1
                fi
            fi
        else
            export USE_FLASH_ATTENTION="0"
            out_file_prefix="$MODE-torch"
            
            if [ "$MODE" == "base" ]; then
                if [ "$SEQ_LEN" == "2048" ]; then
                    time_out=90
                elif [ "$SEQ_LEN" == "1024" ]; then
                    time_out=90
                elif [ "$SEQ_LEN" == "512" ]; then
                    time_out=60
                elif [ "$SEQ_LEN" == "256" ]; then
                    time_out=60
                else
                    echo "Unknown SEQ_LEN: $SEQ_LEN"
                    exit 1
                fi
            elif [ "$MODE" == "pccheck" ]; then
                if [ "$SEQ_LEN" == "2048" ]; then
                    time_out=85
                elif [ "$SEQ_LEN" == "1024" ]; then
                    time_out=85
                elif [ "$SEQ_LEN" == "512" ]; then
                    time_out=60
                elif [ "$SEQ_LEN" == "256" ]; then
                    time_out=60
                else
                    echo "Unknown SEQ_LEN: $SEQ_LEN"
                    exit 1
                fi
            fi
        fi

        srun_outfile=$(get_srun_outfile "$MERGER_DIR/logs" "$SRUN_OUTFILE_SUFFIX-$out_file_prefix" "$SEQ_LEN-part1")

        echo "Writing to: $srun_outfile"

        DTYPE="bf16"
        [ -d "$MERGER_DIR/results/$DTYPE" ] || mkdir $MERGER_DIR/results/$DTYPE

        TRAINING_BASE_CMD="python3 -m $TRAIN_FILE_NAME \
                              --seed 4 \
                              --model-dtype $DTYPE \
                              --checkpoint-dir $MERGER_DIR/chkpts \
                              --sequence-length $SEQ_LEN \
                              --checkpoint-freq 100"

        TRAINING_CMD="$TRAINING_BASE_CMD \
                              --loss-file $MERGER_DIR/results/$DTYPE/loss-$out_file_prefix-$SEQ_LEN-part1.csv \
                              $TRAINING_ARGS"
        srun --output $srun_outfile \
             --cpus-per-task $SLURM_CPUS_PER_TASK \
             bash -c "timeout -s SIGINT $time_out $CMD_PREFIX $TRAINING_CMD; ec=\$?; [ \$ec -eq 0 ] || [ \$ec -eq 124 ]"


        
        srun_outfile=$(get_srun_outfile "$MERGER_DIR/logs" "$SRUN_OUTFILE_SUFFIX-$out_file_prefix" "$SEQ_LEN-part2")

        echo "Writing to: $srun_outfile"

        TRAINING_CMD="$TRAINING_BASE_CMD \
                              --loss-file $MERGER_DIR/results/$DTYPE/loss-$out_file_prefix-$SEQ_LEN-part2.csv \
                              --load-checkpoint \
                              $TRAINING_ARGS"
        srun --output $srun_outfile \
             --cpus-per-task $SLURM_CPUS_PER_TASK \
             bash -c "$CMD_PREFIX $TRAINING_CMD"
    }

    bench_flash_attn "True"
    bench_flash_attn "False"
}


SEQ_LEN=256
TRAIN_ARGS=""

run_merger "base" "$SEQ_LEN" "$TRAIN_ARGS"
run_merger "pccheck" "$SEQ_LEN" "$TRAIN_ARGS"

cd $MERGER_DIR/results && python plot_loss.py --seq-len $SEQ_LEN

echo "END TIME: $(date)"
