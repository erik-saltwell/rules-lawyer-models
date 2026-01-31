# Only apply in interactive shells
case $- in
  *i*) ;;
  *) return 0 ;;
esac

# Guard: block `uv sync` interactively. Override with UV_ALLOW_SYNC=1
uv() {
  if [[ "${1:-}" == "sync" && -z "${UV_ALLOW_SYNC:-}" ]]; then
    echo "Blocked: use uv-shell (runs: uv sync --frozen --inexact)." >&2
    echo "Override (not recommended): UV_ALLOW_SYNC=1 uv sync ..." >&2
    return 1
  fi
  command uv "$@"
}
