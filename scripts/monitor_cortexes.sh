#!/bin/bash
# Monitor all cortex logs in real-time

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"

echo "Monitoring BioMimetic AI cortexes..."
echo "Press Ctrl+C to stop"
echo ""

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Touch log files to ensure they exist
touch "$LOG_DIR/episodic_consolidation.log"
touch "$LOG_DIR/axiom_spot_check.log"
touch "$LOG_DIR/contact_learning.log"
touch "$LOG_DIR/axiom_evaluation.log"
touch "$LOG_DIR/cron.log"

# Use multitail if available, otherwise fall back to tail
if command -v multitail &> /dev/null; then
    multitail \
        -l "tail -f $LOG_DIR/episodic_consolidation.log" \
        -l "tail -f $LOG_DIR/axiom_spot_check.log" \
        -l "tail -f $LOG_DIR/contact_learning.log" \
        -l "tail -f $LOG_DIR/axiom_evaluation.log" \
        -l "tail -f $LOG_DIR/cron.log"
else
    # Fallback: tail all logs
    tail -f \
        "$LOG_DIR/episodic_consolidation.log" \
        "$LOG_DIR/axiom_spot_check.log" \
        "$LOG_DIR/contact_learning.log" \
        "$LOG_DIR/axiom_evaluation.log" \
        "$LOG_DIR/cron.log"
fi
