#!/bin/bash
# Install BioMimetic AI cron jobs

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PATH="$PROJECT_ROOT/venv"

echo "Installing BioMimetic AI cron jobs..."
echo "Project root: $PROJECT_ROOT"

# Ensure scripts are executable
chmod +x "$SCRIPT_DIR"/cron/*.py

# Create cron entries
TEMP_CRON=$(mktemp)

# Preserve existing cron jobs (if any)
crontab -l > "$TEMP_CRON" 2>/dev/null || true

# Add header
echo "" >> "$TEMP_CRON"
echo "# BioMimetic AI Cortex Schedule" >> "$TEMP_CRON"
echo "# Installed: $(date)" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Episodic consolidation - Every 10 minutes
echo "# Episodic Memory Consolidation (60s timeout)" >> "$TEMP_CRON"
echo "*/10 * * * * $VENV_PATH/bin/python3 $SCRIPT_DIR/cron/episodic_consolidation.py >> $PROJECT_ROOT/logs/cron.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Axiom spot check - Every 15 minutes
echo "# Axiom Spot Check (3min timeout)" >> "$TEMP_CRON"
echo "*/15 * * * * $VENV_PATH/bin/python3 $SCRIPT_DIR/cron/axiom_spot_check.py >> $PROJECT_ROOT/logs/cron.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Contact learning - Every 30 minutes
echo "# Contact Learning - Learn about DM contacts (5min timeout)" >> "$TEMP_CRON"
echo "*/30 * * * * $VENV_PATH/bin/python3 $SCRIPT_DIR/cron/contact_learning.py >> $PROJECT_ROOT/logs/cron.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Full axiom evaluation - Every 4 hours
echo "# Full Axiom Evaluation (15min timeout)" >> "$TEMP_CRON"
echo "0 */4 * * * $VENV_PATH/bin/python3 $SCRIPT_DIR/cron/full_axiom_evaluation.py >> $PROJECT_ROOT/logs/cron.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Install crontab
crontab "$TEMP_CRON"
rm "$TEMP_CRON"

echo "✓ Cron jobs installed successfully"
echo ""
echo "Installed jobs:"
crontab -l | grep -A1 "BioMimetic"
echo ""
echo "View logs:"
echo "  tail -f $PROJECT_ROOT/logs/cron.log"
echo "  tail -f $PROJECT_ROOT/logs/episodic_consolidation.log"
echo "  tail -f $PROJECT_ROOT/logs/axiom_spot_check.log"
echo "  tail -f $PROJECT_ROOT/logs/axiom_evaluation.log"
