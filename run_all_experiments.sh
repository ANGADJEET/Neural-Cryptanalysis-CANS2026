#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  Run All Experiments & Push Results
#  Branch: phase/implementation-plan
#
#  Usage:
#    chmod +x run_all_experiments.sh
#    ./run_all_experiments.sh              # run everything
#    ./run_all_experiments.sh --quick      # reduced samples for testing
#    ./run_all_experiments.sh --skip-push  # don't git push at the end
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Config ─────────────────────────────────────────────────────────
QUICK=false
SKIP_PUSH=false
DEVICE="cuda"
LOG_DIR="./results/_logs"

# Defaults (full run)
SAMPLES=500000
EPOCHS=30
N_SEEDS=5
MINE_EPOCHS=200
EVO_GENS=30

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK=true
            SAMPLES=50000
            EPOCHS=10
            N_SEEDS=2
            MINE_EPOCHS=50
            EVO_GENS=5
            echo "⚡ Quick mode: reduced samples/epochs/seeds"
            ;;
        --skip-push)
            SKIP_PUSH=true
            ;;
        --cpu)
            DEVICE="cpu"
            ;;
    esac
done

mkdir -p "$LOG_DIR"

# ─── Helpers ────────────────────────────────────────────────────────
TOTAL=0
PASSED=0
FAILED=0
FAILED_NAMES=""
START_TIME=$(date +%s)

run_exp() {
    local name="$1"
    shift
    local logfile="$LOG_DIR/${name}.log"

    TOTAL=$((TOTAL + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$TOTAL] $name"
    echo "  Log: $logfile"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local t0=$(date +%s)

    if "$@" 2>&1 | tee "$logfile"; then
        local t1=$(date +%s)
        local elapsed=$(( t1 - t0 ))
        echo "  ✓ $name PASSED (${elapsed}s)"
        PASSED=$((PASSED + 1))
    else
        local t1=$(date +%s)
        local elapsed=$(( t1 - t0 ))
        echo "  ✗ $name FAILED (${elapsed}s) — see $logfile"
        FAILED=$((FAILED + 1))
        FAILED_NAMES="$FAILED_NAMES $name"
    fi
}

# ─── Experiments ────────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════════"
echo "  Neural-Cryptanalysis: Full Experiment Suite"
echo "  Device: $DEVICE | Samples: $SAMPLES | Seeds: $N_SEEDS"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════════"

# --- Phase 1: Critical reviewer defenses ---

# E06: Conditional MI with MINE calibration
for cipher in speck32 simon32 present; do
    run_exp "e06_${cipher}" \
        python experiments/exp06_conditional_mi.py \
            --cipher "$cipher" --samples "$SAMPLES" \
            --mine-epochs "$MINE_EPOCHS" --device "$DEVICE" \
            --seed 42 --n-seeds "$N_SEEDS"
done

# E09b: ΔP-invariance sweep
for cipher in speck32 simon32; do
    run_exp "e09b_${cipher}" \
        python experiments/exp09b_delta_invariance.py \
            --cipher "$cipher" --samples "$SAMPLES" \
            --epochs "$EPOCHS" --device "$DEVICE" \
            --seed 42 --n-seeds "$N_SEEDS"
done

# --- Phase 2: Deepening the contribution ---

# E21: Transfer MI characterization
for cipher in speck32 simon32 present; do
    run_exp "e21_${cipher}" \
        python experiments/exp21_transfer_mi.py \
            --cipher "$cipher" --samples "$SAMPLES" \
            --epochs "$EPOCHS" --mine-epochs "$MINE_EPOCHS" \
            --device "$DEVICE" --seed 42 --n-seeds "$N_SEEDS"
done

# E22: Cross-round saliency
for cipher in speck32 simon32 present; do
    run_exp "e22_${cipher}" \
        python experiments/exp22_cross_saliency.py \
            --cipher "$cipher" --samples "$SAMPLES" \
            --epochs "$EPOCHS" --device "$DEVICE" \
            --seed 42 --n-seeds "$N_SEEDS"
done

# --- Phase 5: Evolutionary ΔP search ---

run_exp "e25_speck32_r7" \
    python experiments/exp25_evolutionary_diff.py \
        --cipher speck32 --rounds 7 \
        --pop-size 50 --generations "$EVO_GENS" \
        --eval-samples "$SAMPLES" --eval-epochs "$EPOCHS" \
        --device "$DEVICE" --seed 42

run_exp "e25_simon32_r9" \
    python experiments/exp25_evolutionary_diff.py \
        --cipher simon32 --rounds 9 \
        --pop-size 50 --generations "$EVO_GENS" \
        --eval-samples "$SAMPLES" --eval-epochs "$EPOCHS" \
        --device "$DEVICE" --seed 42

# --- Phase 6: PRESENT key recovery ---

run_exp "e12b_present_r4" \
    python experiments/exp12b_present_key_recovery.py \
        --rounds 4 --samples "$SAMPLES" \
        --epochs "$EPOCHS" --device "$DEVICE" \
        --seed 42 --n-seeds "$N_SEEDS"

run_exp "e12b_present_r5" \
    python experiments/exp12b_present_key_recovery.py \
        --rounds 5 --samples "$SAMPLES" \
        --epochs "$EPOCHS" --device "$DEVICE" \
        --seed 42 --n-seeds "$N_SEEDS"

# --- Phase 7: SIMON32-IRK comparison ---

run_exp "e26_simon_irk" \
    python experiments/exp26_simon_irk_transfer.py \
        --samples "$SAMPLES" --epochs "$EPOCHS" \
        --device "$DEVICE" --seed 42 --n-seeds "$N_SEEDS"

# --- Existing pending fixes (P1/P2) ---

run_exp "pending_fixes" \
    python experiments/run_pending_fixes.py \
        --task all --device "$DEVICE" --samples "$SAMPLES" \
        --seeds "$N_SEEDS"

# ─── Summary ────────────────────────────────────────────────────────

END_TIME=$(date +%s)
TOTAL_TIME=$(( END_TIME - START_TIME ))
HOURS=$(( TOTAL_TIME / 3600 ))
MINS=$(( (TOTAL_TIME % 3600) / 60 ))

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  EXPERIMENT SUITE COMPLETE"
echo "  Passed: $PASSED / $TOTAL"
echo "  Failed: $FAILED"
if [ -n "$FAILED_NAMES" ]; then
    echo "  Failed experiments:$FAILED_NAMES"
fi
echo "  Total time: ${HOURS}h ${MINS}m"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════════"

# ─── Git commit & push ──────────────────────────────────────────────

echo ""
echo "Committing results..."
git add -A results/ "$LOG_DIR/"
git commit -m "Experiment results: $PASSED/$TOTAL passed ($(date +%Y-%m-%d))

Experiments run:
- E06 conditional MI (speck32, simon32, present)
- E09b ΔP-invariance (speck32, simon32)
- E21 transfer MI (speck32, simon32, present)
- E22 cross-saliency (speck32, simon32, present)
- E25 evolutionary ΔP (speck32 7r, simon32 9r)
- E12b PRESENT key recovery (4r, 5r)
- E26 SIMON32-IRK transfer
- Pending fixes (MINE control, E09 t-test, E11 rerun)

Config: samples=$SAMPLES, epochs=$EPOCHS, seeds=$N_SEEDS
Device: $DEVICE | Total time: ${HOURS}h ${MINS}m
Failed: $FAILED ($FAILED_NAMES)" || echo "Nothing to commit"

if [ "$SKIP_PUSH" = false ]; then
    echo "Pushing to origin..."
    git push origin phase/implementation-plan
    echo "✓ Pushed to origin/phase/implementation-plan"
else
    echo "Skipping push (--skip-push)"
fi

echo ""
echo "✓ Done!"
