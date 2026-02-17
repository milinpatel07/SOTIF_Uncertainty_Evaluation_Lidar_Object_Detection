#!/usr/bin/env bash
# =============================================================================
# Train K SECOND detector ensemble members with different random seeds.
#
# Prerequisites:
#   - OpenPCDet installed (https://github.com/open-mmlab/OpenPCDet)
#   - KITTI dataset downloaded and prepared
#   - CUDA-compatible GPU
#
# Usage:
#   bash scripts/train_ensemble.sh --seeds 0 1 2 3 4 5
#   bash scripts/train_ensemble.sh --seeds 0 1 2 3 4 5 --epochs 80 --batch_size 4
#
# Each member is trained independently with identical hyperparameters,
# differing only in random seed (Section 4.2 of the paper).
# =============================================================================

set -euo pipefail

# Default configuration
SEEDS=(0 1 2 3 4 5)
EPOCHS=80
BATCH_SIZE=4
CFG_FILE="tools/cfgs/kitti_models/second.yaml"
OPENPCDET_ROOT="${OPENPCDET_ROOT:-./OpenPCDet}"
OUTPUT_ROOT="output/ensemble"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            shift
            SEEDS=()
            while [[ $# -gt 0 ]] && ! [[ "$1" == --* ]]; do
                SEEDS+=("$1")
                shift
            done
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --cfg_file)
            CFG_FILE="$2"
            shift 2
            ;;
        --openpcdet_root)
            OPENPCDET_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "SOTIF Uncertainty Evaluation - Ensemble Training"
echo "=============================================="
echo "Seeds: ${SEEDS[*]}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Config: ${CFG_FILE}"
echo "OpenPCDet root: ${OPENPCDET_ROOT}"
echo "Output: ${OUTPUT_ROOT}"
echo "=============================================="

# Check OpenPCDet installation
if [ ! -d "${OPENPCDET_ROOT}" ]; then
    echo "Error: OpenPCDet not found at ${OPENPCDET_ROOT}"
    echo "Install OpenPCDet first: https://github.com/open-mmlab/OpenPCDet"
    echo "Or set OPENPCDET_ROOT environment variable."
    exit 1
fi

cd "${OPENPCDET_ROOT}"

# Train each member
for SEED in "${SEEDS[@]}"; do
    TAG="seed_${SEED}"
    echo ""
    echo "----------------------------------------------"
    echo "Training member: ${TAG} (seed=${SEED})"
    echo "----------------------------------------------"

    python tools/train.py \
        --cfg_file "${CFG_FILE}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --extra_tag "${TAG}" \
        --fix_random_seed \
        --set OPTIMIZATION.SEED "${SEED}" \
        2>&1 | tee "${OUTPUT_ROOT}/${TAG}/train.log"

    echo "Member ${TAG} training complete."
done

echo ""
echo "=============================================="
echo "All ${#SEEDS[@]} ensemble members trained."
echo "Checkpoints saved to: ${OUTPUT_ROOT}/seed_*/ckpt/"
echo "=============================================="
echo ""
echo "Next step: Run ensemble inference"
echo "  python scripts/run_inference.py \\"
echo "    --ckpt_dirs ${OUTPUT_ROOT}/seed_*"
