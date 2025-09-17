#!/usr/bin/env bash
set -euo pipefail

# Compose up using top-level file plus all service fragments located under docker/compose.
# Supports: list, up <service>, and pass-through docker compose args.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SERVICES_DIR="$PROJECT_ROOT/docker/compose"

usage() {
  cat <<EOF
Usage: $0 [command] [docker-compose args]

Commands:
  list                Show available service fragment files and derived service names
  up <service> [args] Run docker compose with all fragments and the given service
  up                  Run docker compose with all fragments (no specific service)
  help                Show this help

Examples:
  $0 list
  $0 up celeba128_vqvae_rvq --build
  $0 up
EOF
}

if [ ! -d "$SERVICES_DIR" ]; then
  echo "Services directory not found: $SERVICES_DIR"
  exit 1
fi

# gather fragment files (recursive)
SERVICE_FILES=()
while IFS= read -r -d '' f; do
  SERVICE_FILES+=("-f" "$f")
done < <(find "$SERVICES_DIR" -type f \( -name "*.yaml" -o -name "*.yml" \) -print0 | sort -z)

if [ ${#SERVICE_FILES[@]} -eq 0 ]; then
  echo "No service fragments found in $SERVICES_DIR"
  exit 1
fi

cmd="${1:-}"
case "$cmd" in
  list)
    echo "Found compose fragment files:"
    for ((i=0;i<${#SERVICE_FILES[@]};i+=2)); do
      echo " - ${SERVICE_FILES[i+1]}"
    done
    echo
    echo "Derived service names (filename without extension):"
    find "$SERVICES_DIR" -type f \( -name "*.yaml" -o -name "*.yml" \) -print0 | sort -z | \
      xargs -0 -n1 basename | sed -E 's/\.[yY][aA]?[mM][lL]$//' | sed 's/.*/ - &/'
    exit 0
    ;;
  help|--help|-h)
    usage
    exit 0
    ;;
  up)
    shift || true
    # assemble docker compose command
    DOCKER_CMD=(docker compose -f "$PROJECT_ROOT/docker-compose.yaml" "${SERVICE_FILES[@]}" up)
    # if service name was provided, append it
    if [ $# -gt 0 ]; then
      DOCKER_CMD+=("$@")
    fi
    echo "Running: ${DOCKER_CMD[*]}"
    exec "${DOCKER_CMD[@]}"
    ;;
  "")
    usage
    exit 0
    ;;
  *)
    echo "Unknown command: $cmd"
    usage
    exit 2
    ;;
esac
#!/usr/bin/env bash
set -euo pipefail
#!/usr/bin/env bash
set -euo pipefail

# Compose up using top-level file plus all service fragments.
# Usage: ./compose-up.sh [additional docker-compose args]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SERVICES_DIR="$PROJECT_ROOT/docker/compose"

# Find all yaml files under services dir (recursive)
SERVICE_FILES=()
while IFS= read -r -d '' f; do
  SERVICE_FILES+=("-f" "$f")
done < <(find "$SERVICES_DIR" -type f \( -name "*.yaml" -o -name "*.yml" \) -print0 | sort -z)

if [ ${#SERVICE_FILES[@]} -eq 0 ]; then
  echo "No service fragments found in $SERVICES_DIR"
  echo "You can run: docker compose up <service-name>"
  exit 1
fi

cmd=(docker compose -f "$PROJECT_ROOT/docker-compose.yaml" "${SERVICE_FILES[@]}" up "$@")

echo "Running: ${cmd[*]}"
exec "${cmd[@]}"
