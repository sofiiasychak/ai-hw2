#!/usr/bin/env bash
# Run experiment lanes on GPUs.
set -uo pipefail

usage() {
  echo "usage: bash scripts/run_lanes.sh \"IDS FOR GPU 0\" [\"IDS FOR GPU 1\"]"
  echo "       bash scripts/run_lanes.sh --ddp ID   (one job on GPUs 0,1)"
  echo "env RUN_ARGS: extra runner arguments, e.g. RUN_ARGS=--smoke"
  exit 2
}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
OUT_ROOT="$(python -c 'from hw2.paths import out_root; print(out_root())')" || exit 1
LOGS="$OUT_ROOT/logs"
mkdir -p "$LOGS"
STATUS="$LOGS/status_$(date +%Y%m%d_%H%M%S).txt"

LOCK="$LOGS/.git.lock"
rmdir "$LOCK" 2>/dev/null

# Serialise pulls across lanes.
pull() {
  local tries=0 rc
  while ! mkdir "$LOCK" 2>/dev/null; do
    ((++tries > 60)) && return 1
    sleep 1
  done
  git pull -q --ff-only
  rc=$?
  rmdir "$LOCK"
  return $rc
}

# Print config path for ID.
config_for() {
  shopt -s nullglob
  local matches=(configs/*/"$1".yaml configs/*/"$1"_*.yaml)
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "no unique config for $1: ${matches[*]:-none}" >&2
    return 1
  fi
  echo "${matches[0]}"
}

# Print runner module for config.
runner_for() {
  case "$(basename "$(dirname "$1")")" in
    yolo | rtdetr) echo hw2.train.run_ultralytics ;;
    frcnn | retinanet) echo hw2.train.run_torchvision ;;
    *) echo "unknown family for $1" >&2; return 1 ;;
  esac
}

# Run one job, never fail.
run_job() {
  local id="$1" gpus="$2" device="$3" log="$LOGS/$1.log" cfg module rc
  pull || echo "[$id] git pull failed, using current checkout" >&2
  if cfg="$(config_for "$id")" && module="$(runner_for "$cfg")"; then
    echo "[$id] start $(date '+%F %T') gpu $gpus $cfg -> $log"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpus" python -m "$module" --config "$cfg" --device "$device" ${RUN_ARGS:-} >"$log" 2>&1
    rc=$?
  else
    rc=127
    echo "[$id] skipped: config or runner missing" | tee "$log"
  fi
  echo "[$id] end $(date '+%F %T') exit $rc"
  echo "$id gpu=$gpus exit=$rc" >>"$STATUS"
}

# Run lane jobs in order.
run_lane() {
  local gpus="$1" id
  shift
  for id in $1; do
    run_job "$id" "$gpus" 0
  done
}

if [[ "${1:-}" == "--ddp" ]]; then
  [[ $# -eq 2 ]] || usage
  run_job "$2" 0,1 0,1
else
  [[ $# -ge 1 && "${1:-}" != -* ]] || usage
  gpu=0
  for lane in "$@"; do
    run_lane "$gpu" "$lane" &
    gpu=$((gpu + 1))
  done
  wait
fi

echo "== job status ($STATUS)"
cat "$STATUS" 2>/dev/null
