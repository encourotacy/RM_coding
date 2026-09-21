sleep 5
cd "$(cd "$(dirname "$0")" && pwd)"
mkdir -p logs
screen \
    -L \
    -Logfile logs/$(date "+%Y-%m-%d_%H-%M-%S").screenlog \
    -d \
    -m \
    bash -c "./build/ovsentry_omni_mpc configs/sentry.yaml --no-display"
