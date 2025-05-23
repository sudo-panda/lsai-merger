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
    FLASH_ATTN_MODE="$4"

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
        else
            export USE_FLASH_ATTENTION="0"
            out_file_prefix="$MODE-torch"
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
        stop_at_step=410

        TRAINING_CMD="$TRAINING_BASE_CMD \
                              --loss-file $MERGER_DIR/results/$DTYPE/loss-$out_file_prefix-$SEQ_LEN-part1.csv \
                              $TRAINING_ARGS"
        # Remove the file if it exists as it interferes with the check for target line
        echo "" > $srun_outfile
        srun --output $srun_outfile \
             --cpus-per-task $SLURM_CPUS_PER_TASK \
             bash -c "$CMD_PREFIX $TRAINING_CMD" &
        srun_pid=$!

        srun_step_id=""
        target_line="INFO - Step: $stop_at_step"
        stopped_by_script=false
        while kill -0 $srun_pid 2>/dev/null; do
            if grep -q "$target_line" "$srun_outfile"; then
                stopped_by_script=true
                
                line=$(sacct -n -j "$SLURM_JOB_ID" \
                        --format=JobID,JobName,State | grep RUNNING | grep bash || true)
                srun_step_id=$(awk '{print $1}' <<< "$line")

                [[ -z "$srun_step_id" ]] && {
                    echo " ! Could not determine step-ID; giving up."
                    exit 1
                }

                # -----------------------------------------------------------------------
                # 2. Repeatedly scancel until the step is gone
                # -----------------------------------------------------------------------
                attempt=0
                while kill -0 "$srun_pid" 2>/dev/null; do
                    if ! squeue -h -j "$srun_step_id" &>/dev/null; then
                        break
                    fi

                    echo " > Attempt $attempt: scancel --signal=SIGKILL  $srun_step_id"
                    scancel --signal=SIGKILL "$srun_step_id" || true
                    sleep 1
                done
                break
            fi
            sleep 2
        done
        
        if $stopped_by_script; then
            status=0
        elif [[ ${status:-0} -ne 0 ]]; then
            echo " > srun failed with exit code ${status}"
            exit $status
        else
            echo " > srun finished sucessfully but $target_line not found."
            exit $status
        fi


        
        srun_outfile=$(get_srun_outfile "$MERGER_DIR/logs" "$SRUN_OUTFILE_SUFFIX-$out_file_prefix" "$SEQ_LEN-part2")

        echo "Writing to: $srun_outfile"

        TRAINING_CMD="$TRAINING_BASE_CMD \
                              --loss-file $MERGER_DIR/results/$DTYPE/loss-$out_file_prefix-$SEQ_LEN-part2.csv \
                              --load-checkpoint \
                              $TRAINING_ARGS"
        srun --output $srun_outfile \
             --cpus-per-task $SLURM_CPUS_PER_TASK \
             bash -c "$CMD_PREFIX $TRAINING_CMD"

        echo "--------------- FAv3: $flash_attn   Checkpt: $MODE   Seq Len: $SEQ_LEN   Complete! ----------------"
    }

    if [ -z "$FLASH_ATTN_MODE" ]; then
        bench_flash_attn "True"
        bench_flash_attn "False"
    elif [ "$FLASH_ATTN_MODE" == "flash" ]; then
        bench_flash_attn "True"
    elif [ "$FLASH_ATTN_MODE" == "torch" ]; then
        bench_flash_attn "False"
    else
        echo "Unknown FLASH_ATTN_MODE: $FLASH_ATTN_MODE"
        exit 1
    fi
}


SEQ_LEN=$1 # 256, 512, 1024, 2048
if [ -z "$SEQ_LEN" ]; then
    SEQ_LEN=2048
fi
TRAIN_ARGS=""


# # # #  run_merger "Checkpoint mode" "$SEQ_LEN" "$TRAIN_ARGS" [ "flash" | "torch" ]
run_merger "base" "$SEQ_LEN" "$TRAIN_ARGS"
run_merger "pccheck" "$SEQ_LEN" "$TRAIN_ARGS"

cd $MERGER_DIR/results && python plot_loss.py --seq-len $SEQ_LEN

echo "END TIME: $(date)"
