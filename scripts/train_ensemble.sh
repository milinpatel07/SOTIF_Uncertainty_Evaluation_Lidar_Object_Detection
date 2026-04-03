#!/usr/bin/env bash
# =============================================================================
# Train K SECOND detector ensemble members with different random seeds.
#
# Prerequisites:
#   - OpenPCDet installed (https://github.com/open-mmlab/OpenPCDet)
#   - KITTI dataset downloaded and prepared (see scripts/prepare_kitti.py)
#   - CUDA-compatible GPU with sufficient VRAM (>= 4 GB)
#
# Usage:
#   bash scripts/train_ensemble.sh --seeds 0 1 2 3 4 5
#   bash scripts/train_ensemble.sh --seeds 0 1 2 3 4 5 --epochs 80 --batch_size 4
#   bash scripts/train_ensemble.sh --openpcdet_root /path/to/OpenPCDet
#
# Each member is trained independently with identical hyperparameters,
# differing only in random seed (Section 4.2 of the paper).
#
# Architecture: SECOND (Yan et al., 2018)
#   MeanVFE -> VoxelBackBone8x -> HeightCompression -> BaseBEVBackbone -> AnchorHeadSingle
#
# Default training configuration from OpenPCDet:
#   - Optimizer: adam_onecycle, LR=0.003
#   - 80 epochs, batch_size=4
#   - Car IoU=0.6/0.45 (matched/unmatched), Pedestrian/Cyclist IoU=0.5/0.35
#
# Random seed sets: random, numpy, torch, torch.cuda, cudnn.deterministic=True
# =============================================================================

set -euo pipefail

# Default configuration
SEEDS=(0 1 2 3 4 5)
EPOCHS=80
BATCH_SIZE=4
CFG_FILE="tools/cfgs/kitti_models/second.yaml"
OPENPCDET_ROOT="${OPENPCDET_ROOT:-./OpenPCDet}"
OUTPUT_ROOT="output/ensemble"
NUM_GPUS=1

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
        --output_root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/train_ensemble.sh [--seeds 0 1 2 ...] [--epochs N] [--batch_size N]"
            exit 1
            ;;
    esac
done

K=${#SEEDS[@]}

echo "=============================================="
echo "SOTIF Uncertainty Evaluation - Ensemble Training"
echo "=============================================="
echo "  Ensemble size (K): ${K}"
echo "  Seeds:             ${SEEDS[*]}"
echo "  Epochs:            ${EPOCHS}"
echo "  Batch size:        ${BATCH_SIZE}"
echo "  Config:            ${CFG_FILE}"
echo "  OpenPCDet root:    ${OPENPCDET_ROOT}"
echo "  Output:            ${OUTPUT_ROOT}"
echo "  GPUs:              ${NUM_GPUS}"
echo "=============================================="

# Check OpenPCDet installation
if [ ! -d "${OPENPCDET_ROOT}" ]; then
    echo ""
    echo "Error: OpenPCDet not found at ${OPENPCDET_ROOT}"
    echo ""
    echo "To install OpenPCDet:"
    echo "  git clone https://github.com/open-mmlab/OpenPCDet.git"
    echo "  cd OpenPCDet"
    echo "  pip install -r requirements.txt"
    echo "  python setup.py develop"
    echo ""
    echo "Or set OPENPCDET_ROOT environment variable:"
    echo "  export OPENPCDET_ROOT=/path/to/OpenPCDet"
    exit 1
fi

# Verify config file exists
if [ ! -f "${OPENPCDET_ROOT}/${CFG_FILE}" ]; then
    echo "Error: Config file not found: ${OPENPCDET_ROOT}/${CFG_FILE}"
    echo "Available KITTI configs:"
    ls "${OPENPCDET_ROOT}"/tools/cfgs/kitti_models/*.yaml 2>/dev/null || echo "  None found"
    exit 1
fi

cd "${OPENPCDET_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

# Track timing
START_TIME=$(date +%s)

# Train each member
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    TAG="seed_${SEED}"
    MEMBER_NUM=$((i + 1))

    echo ""
    echo "----------------------------------------------"
    echo "Training member ${MEMBER_NUM}/${K}: ${TAG} (seed=${SEED})"
    echo "Started at: $(date)"
    echo "----------------------------------------------"

    # Create output directory
    mkdir -p "${OUTPUT_ROOT}/${TAG}"

    # OpenPCDet training command
    # --fix_random_seed enables deterministic training
    # --set_random_seed overrides the default seed (666) with our seed
    # Note: Standard OpenPCDet uses seed 666; LiDAR-MIMO adds --set_random_seed
    # If using standard OpenPCDet without --set_random_seed, modify common_utils.py
    if [ "${NUM_GPUS}" -gt 1 ]; then
        # Multi-GPU training with torch.distributed
        python -m torch.distributed.launch \
            --nproc_per_node="${NUM_GPUS}" \
            tools/train.py \
            --cfg_file "${CFG_FILE}" \
            --batch_size "${BATCH_SIZE}" \
            --epochs "${EPOCHS}" \
            --extra_tag "${TAG}" \
            --fix_random_seed \
            --set OPTIMIZATION.SEED "${SEED}" \
            2>&1 | tee "${OUTPUT_ROOT}/${TAG}/train.log"
    else
        python tools/train.py \
            --cfg_file "${CFG_FILE}" \
            --batch_size "${BATCH_SIZE}" \
            --epochs "${EPOCHS}" \
            --extra_tag "${TAG}" \
            --fix_random_seed \
            --set OPTIMIZATION.SEED "${SEED}" \
            2>&1 | tee "${OUTPUT_ROOT}/${TAG}/train.log"
    fi

    echo "Member ${TAG} training complete at $(date)."
    echo ""
done

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=============================================="
echo "Ensemble training complete."
echo "  Members trained:  ${K}"
echo "  Total time:       ${HOURS}h ${MINUTES}m"
echo "  Checkpoints:      ${OUTPUT_ROOT}/seed_*/ckpt/"
echo "=============================================="
echo ""
echo "Next step: Run ensemble inference"
echo "  python scripts/run_inference.py \\"
echo "    --ckpt_dirs ${OUTPUT_ROOT}/seed_*"
echo ""
echo "Tip: To modify the random seed mechanism in standard OpenPCDet,"
echo "edit pcdet/utils/common_utils.py set_random_seed() to accept a"
echo "configurable seed, or use LiDAR-MIMO's OpenPCDet fork which"
echo "already supports --set_random_seed."
