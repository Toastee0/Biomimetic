#!/bin/bash
# Remove BioMimetic AI cron jobs

set -e

echo "Removing BioMimetic AI cron jobs..."

TEMP_CRON=$(mktemp)

# Get current crontab, filter out BioMimetic entries
crontab -l 2>/dev/null | grep -v "BioMimetic" | grep -v "episodic_consolidation" | grep -v "axiom_spot_check" | grep -v "contact_learning" | grep -v "full_axiom_evaluation" > "$TEMP_CRON" || true

# Install filtered crontab
crontab "$TEMP_CRON"
rm "$TEMP_CRON"

echo "✓ BioMimetic AI cron jobs removed"
echo ""
echo "Remaining cron jobs:"
crontab -l
