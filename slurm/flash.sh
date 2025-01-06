#!/bin/bash

# --- Global script options
SIM_NAME="flash"
FLASH_BIN="flash4"
PAR_FILE="flash.par"
AUTO_RESTART=true
CONFIRMATION=false
VERBOSE=false

# --- Global simulation properties
SIM_OUTPUT_DIR=""
SIM_BASENM=""
SIM_RESTART=""
SIM_CHK_NUMBER=0
SIM_PLT_NUMBER=0

function print_help {
cat << EOF
Usage: ./flash.sh [OPTION]...
Submit a FLASH job to slurm.

  -b, --bin=BINARY          Specify the path to the FLASH binary to execute.
                            Default is "flash4".
      --par-file=PATH       Specify the path to the FLASH parameter file.
                            Default is "flash.par".
      --auto-restart[=BOOL]
      --no-auto-restart     Whether to use the automatic restart feature.
                            If enabled, the script will force the simulation to
                            restart from the latest checkpoint file in the
                            output directory, if any already exist.
                            This is done by overwriting the restart,
                            checkpointFileNumber, and plotFileNumber variables
                            in the parameter file.
                            If disabled, the simulation will use the parameter
                            file as-is. Any existing checkpoint/plot file in
                            the output directory may therefore be overwritten.
                            BOOL may be any of true,on,enable/false,off,disable.
                            Default is true.
  -c, --confirm             Ask for confirmation before submitting.
  -v, --verbose             Print more debug information.
      --sim-name=NAME       Give this run a little name for the log files.
  -h, --help                Display this help and exit.
      --version             Output version information and exit.
EOF
}

function print_version {
cat << EOF
FLASH slurm script (CCSN Edition) 1.0
EOF
}

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
            --auto-restart=*)
                if [[ "${arg#*=}" == "true" ]] || [[ "${arg#*=}" == "on" ]] || [[ "${arg#*=}" == "enable" ]]; then
                    AUTO_RESTART=true
                elif [[ "${arg#*=}" == "false" ]] || [[ "${arg#*=}" == "off" ]] || [[ "${arg#*=}" == "disable" ]]; then
                    AUTO_RESTART=false
                else
                    echo "Invalid value for option --auto-restart: ${arg#*=}!"
                    exit
                fi
                shift
                ;;
            --auto-restart)
                AUTO_RESTART=true
                shift
                ;;
            --no-auto-restart)
                AUTO_RESTART=false
                shift
                ;;
            -c | --confirm)
                CONFIRMATION=true
                shift
                ;;
            -v | --verbose)
                VERBOSE=true
                shift
                ;;
            --sim-name=*)
                SIM_NAME="${arg#*=}"
                shift
                ;;
            -h | --help)
                print_help
                exit 0
                ;;
            --version)
                print_version
                exit 0
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

function check_param {
    # --- Check FLASH binary validity
    if [[ ! -x "$FLASH_BIN" ]]; then
        echo "Error: $FLASH_BIN is not a valid executable!"
        exit 1
    fi

    # --- Check parameter file existence
    if [[ ! -f "$PAR_FILE" ]]; then
        echo "Error: Parameter file $PAR_FILE not found!"
        exit 1
    fi

    # --- Check job script exists
    if [[ ! -f "submit-flash.sh" ]]; then
        echo "Error: submit-flash.sh not found!"
        exit 1
    fi
}

function read_par_file {
    SIM_OUTPUT_DIR=$(grep "output_directory" "$PAR_FILE" | sed -E "s/^.*=\s*//" | sed -E 's/"//g')
    SIM_BASENM=$(grep "basenm" "$PAR_FILE" | sed -E "s/^.*=\s*//" | sed -E 's/"//g')
    SIM_RESTART=$(grep "restart" "$PAR_FILE" | sed -E "s/^.*=\s*//")
    SIM_CHK_NUMBER=$(grep "checkpointFileNumber" "$PAR_FILE" | sed -E "s/^.*=\s*//")
    SIM_PLT_NUMBER=$(grep "plotFileNumber" "$PAR_FILE" | sed -E "s/^.*=\s*//")
}

function init_output_dir {
    # Directory's missing, create and leave
    if [[ ! -d "$SIM_OUTPUT_DIR" ]]; then
        $VERBOSE && echo "Output directory $SIM_OUTPUT_DIR not found! Creating now."
        mkdir "$SIM_OUTPUT_DIR" || exit
        # Unnecessary if there was no output directory
        AUTO_RESTART=false
        return 1
    fi
}

function init_autorestart {
    # --- Get simulation intended output directory and basename
    local CHK_FILES
    local SORTED_CHK_FILES
    local BASENAMES
    local LAST_CHK_FILE
    local NEXT_PLT_FILE

    # If nullglob is not set, globbing will return non sense if no file matches
    shopt -s nullglob
    CHK_FILES=("$SIM_OUTPUT_DIR"/*_hdf5_chk_[0-9]*)
    if [[ "${#CHK_FILES[@]}" -gt 0 ]]; then
        # Sort the different basenames present in the output folder
        BASENAMES=($(for FILE in "${CHK_FILES[@]}"; do
            # Extract file name and split basename
            basename "$FILE" | awk -F'hdf5_chk' '{print $1}'
        done | sort -u))

        # If no checkpoint file matches the basename from the parameter file, leave
        if [[ ! " ${BASENAMES[*]} " =~ [[:space:]]"$SIM_BASENM"[[:space:]] ]]; then
            $VERBOSE && echo "No existing simulation matching basename $SIM_BASENM found."
            return 0
        fi

        # Get checkpoint files with matched basename and sort numerically
        CHK_FILES=("$SIM_OUTPUT_DIR"/"${SIM_BASENM}"*_chk_[0-9]*)
        IFS=$'\n'
        SORTED_CHK_FILES=($(sort -n <<< "${CHK_FILES[*]}"))
        IFS=$' \t\n'

        # If only one chk file, try it, if more, compare with second to last for corruption
        #declare -i LAST_CHK_INDEX=$(ls -t "$SIM_OUTPUT_DIR" | grep "^${SIM_BASENM}hdf5_chk_[0-9]*$" | head -n1 | sed -E "s/^.*hdf5_chk_([0-9]*)/\1/")
        declare -i LAST_CHK_INDEX
        LAST_CHK_INDEX=$(sed -E "s/.*chk_([0-9]*)$/\1/" <<< "${SORTED_CHK_FILES[-1]}")
        LAST_CHK_FILE="${SORTED_CHK_FILES[-1]}"
        if [[ "${#CHK_FILES[@]}" -gt 1 ]]; then
            declare -i CHKSIZE_1
            declare -i CHKSIZE_2

            CHKSIZE_1=$(stat -c%s "${SORTED_CHK_FILES[-1]}")
            CHKSIZE_2=$(stat -c%s "${SORTED_CHK_FILES[-2]}")
            # Last checkpoint most likely corrupted, use previous one instead
            if [[ CHKSIZE_1 -lt CHKSIZE_2 ]]; then
                $VERBOSE && echo "Latest checkpoint file (${SORTED_CHK_FILES[-1]}) appears corrupted!"
                $VERBOSE && echo "Next available checkpoint is ${SORTED_CHK_FILES[-2]}."
                LAST_CHK_INDEX=$(sed -E "s/.*chk_([0-9]*)$/\1/" <<< "${SORTED_CHK_FILES[-2]}")
                LAST_CHK_FILE="${SORTED_CHK_FILES[-2]}"
            fi
        fi

        # Note: finding the latest plot file associated with the saved chk file relies on the files' timestamp.
        local CHK_FILE_REACHED=false
        declare -i NEXT_PLT_INDEX=0
        NEXT_PLT_FILE=""
        while read -r FILE; do
            if [[ "$FILE" =~ .*hdf5_plt_cnt.* ]] && [[ "$CHK_FILE_REACHED" = true ]]; then
                NEXT_PLT_INDEX=$(($(sed -E "s/.*hdf5_plt_cnt_([0-9]*)$/\1/" <<< "$FILE") + 1))
                NEXT_PLT_FILE="$FILE"
                break
            elif [[ "$FILE" == $(basename "$LAST_CHK_FILE") ]]; then
                CHK_FILE_REACHED=true
            fi
        done < <(ls -1t "$SIM_OUTPUT_DIR")

        $VERBOSE && echo "Auto-restart completed"
        $VERBOSE && [ -n "$LAST_CHK_FILE" ] && echo -e "\t@ $LAST_CHK_FILE" || echo -e "\tNo valid checkpoint found"
        $VERBOSE && [ -n "$NEXT_PLT_FILE" ] && echo -e "\t@ $NEXT_PLT_FILE" || echo -e "\tNo valid plot file found"

        # Update relevant restart variables in the parameter file
        sed -Ei "s/(restart\s*=\s*).*/\1.true./" "${PAR_FILE}"
        sed -Ei "s/(checkpointFileNumber\s*=\s*).*/\1${LAST_CHK_INDEX}/" "$PAR_FILE"
        sed -Ei "s/(plotFileNumber\s*=\s*).*/\1${NEXT_PLT_INDEX}/" "$PAR_FILE"
    else
        $VERBOSE && echo "No existing simulation found."
    fi
}

function print_status {
    # --- Print debug information
    echo "Executing $FLASH_BIN with parameter file $PAR_FILE."
    echo -e "\tOutput directory: $SIM_OUTPUT_DIR"
    echo -e "\tBasename        : $SIM_BASENM"
    echo -e "\tRestart         : $SIM_RESTART"
    echo -e "\tFrom checkpoint : $SIM_CHK_NUMBER"
    echo -e "\tNext plot file  : $SIM_PLT_NUMBER"

    if [[ "$CONFIRMATION" = true ]]; then
        read -p "Submit? ([y/n]) " -r
        echo
        if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
            echo "Exiting."
            exit 1
        fi
    fi
}

parse_cmd "$@"
check_param
read_par_file
init_output_dir
if [[ "$AUTO_RESTART" = true ]]; then
    init_autorestart
    # Check updates in par file
    read_par_file
fi
print_status

sbatch submit-flash.sh --bin="$FLASH_BIN" --par-file="$PAR_FILE" --sim-name="$SIM_NAME"
echo "Submitted!"

