#!/bin/bash
# Check status of all cortexes

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
STATE_DIR="$PROJECT_ROOT/data/cortex_state"

echo "========================================"
echo "BioMimetic AI - Cortex Status"
echo "========================================"
echo ""

if [ ! -d "$STATE_DIR" ]; then
    echo "No cortex state directory found."
    echo "Cortexes may not have run yet."
    exit 0
fi

for state_file in "$STATE_DIR"/*.json; do
    if [ -f "$state_file" ]; then
        cortex_name=$(basename "$state_file" .json)
        echo "--- $cortex_name ---"
        
        # Extract key fields using jq if available, otherwise python
        if command -v jq &> /dev/null; then
            status=$(jq -r '.status' "$state_file" 2>/dev/null || echo "unknown")
            last_run=$(jq -r '.last_run' "$state_file" 2>/dev/null || echo "0")
            duration=$(jq -r '.duration_seconds' "$state_file" 2>/dev/null || echo "0")
            items=$(jq -r '.items_processed // .axioms_tested // 0' "$state_file" 2>/dev/null || echo "0")
        else
            status=$(python3 -c "import json; print(json.load(open('$state_file')).get('status', 'unknown'))")
            last_run=$(python3 -c "import json; print(json.load(open('$state_file')).get('last_run', 0))")
            duration=$(python3 -c "import json; print(json.load(open('$state_file')).get('duration_seconds', 0))")
            items=$(python3 -c "import json; d=json.load(open('$state_file')); print(d.get('items_processed') or d.get('axioms_tested', 0))")
        fi
        
        # Convert timestamp to human readable
        if [ "$last_run" != "0" ]; then
            last_run_human=$(date -d "@$last_run" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "unknown")
        else
            last_run_human="never"
        fi
        
        # Status emoji
        if [ "$status" = "success" ]; then
            status_icon="✓"
        elif [ "$status" = "error" ]; then
            status_icon="✗"
        elif [ "$status" = "timeout" ]; then
            status_icon="⏱"
        else
            status_icon="?"
        fi
        
        echo "  Status: $status_icon $status"
        echo "  Last run: $last_run_human"
        echo "  Duration: ${duration}s"
        echo "  Processed: $items items"
        echo ""
    fi
done

echo "========================================"
echo "To view logs:"
echo "  tail -f $PROJECT_ROOT/logs/<cortex_name>.log"
echo ""
echo "To view all cortex activity:"
echo "  ./scripts/monitor_cortexes.sh"
