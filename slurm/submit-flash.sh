#!/bin/bash

# -----------------------------
# SLURM Configuration
# -----------------------------

# --- Task name
#SBATCH -J FLASH

# --- Resource allocation
# #SBATCH --nodes 1                       # Number of nodes to distribute tasks on
#SBATCH --ntasks 32                     # Number of tasks distributed on nodes (distributed parallelism (MPI))
# #SBATCH --cpus-per-task 1               # Threads per task running on a single node (shared-memory parallelism (OpenMP))
#SBATCH --mem-per-cpu=2000MB

#SBATCH --partition ComputeNew
# #SBATCH --nodelist node06

# --- Mail address (uncomment once to get mails)
# #SBATCH --mail-type=end,fail
# #SBATCH --mail-user=gaetan.jalin@tu-darmstadt.de

# --- Working directory
# #SBATCH -D /home/gjalin/

# --- Log files
#SBATCH --output flash-%j.log
#SBATCH --error flash.err

# --- Set up environment
MPI_PATH=/home/tools/compiler/intel/mpi/latest

# --- Script parameters
SIM_NAME="flash"
FLASH_BIN="flash4"
PAR_FILE="flash.par"

function parse_cmd {
    # --- Parse command line arguments
    for arg in "$@"; do
        case $arg in
            -b=* | --bin=*)
                FLASH_BIN="${arg#*=}"
                shift
                ;;
            --par-file=*)
                PAR_FILE="${arg#*=}"
                shift
                ;;
            --sim-name=*)
                SIM_NAME="${arg#*=}"
                shift
                ;;
            -b | --bin | --par-file | --sim-name)
                echo "Missing value for option ${arg}!"
                exit 1
                ;;
            *)
                echo "Unrecognised option: ${arg}!"
                exit 1
                ;;
        esac
    done
}

function init_logs {
    # --- Create log history if needed and save logs with starting date
    if [[ ! -d "logs" ]]; then
        mkdir logs
    fi

    # --- We create hard links to the slurm log files so that the latter can be
    # deleted safely at the end of the script for cleanliness
    # TODO Logs not deleted if job is cancelled with scancel?
    if [[ -f "logs/${SIM_NAME}-$(date --date="@${SLURM_JOB_START_TIME}" +%H%M).log" ]]; then
        declare -i iter=1
        while [[ -f "logs/${SIM_NAME}-$(date --date="@${SLURM_JOB_START_TIME}" +%H%M)-${iter}.log" ]]; do
            iter+=1
        done
        ln -f "flash-${SLURM_JOB_ID}.log" "logs/${SIM_NAME}-$(date --date="@${SLURM_JOB_START_TIME}" +%H%M)-${iter}.log"
    else
        ln -f "flash-${SLURM_JOB_ID}.log" "logs/${SIM_NAME}-$(date --date="@${SLURM_JOB_START_TIME}" +%H%M).log"
    fi
}

function run_flash {
    echo "Running FLASH on node(s) $SLURM_JOB_NODELIST ($SLURM_JOB_PARTITION)"
    [ -n "$SLURM_NTASKS" ] && echo -e "\twith $SLURM_NTASKS tasks"
    [ -n "$SLURM_CPUS_PER_TASK" ] && echo -e "\twith $SLURM_CPUS_PER_TASK CPUs per task"

    ${MPI_PATH}/bin/mpirun -n ${SLURM_NTASKS} ./${FLASH_BIN} -par_file ${PAR_FILE}
    wait

    echo "Stopping:"
    date
}

function cleanup {
    # Clean-up slurm log files
    rm flash-${SLURM_JOB_ID}.log
    #rm flash-${SLURM_JOB_ID}.err
}

parse_cmd "$@"
init_logs
run_flash
cleanup

echo "Done!"

