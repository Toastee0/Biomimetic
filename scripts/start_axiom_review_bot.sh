#!/bin/bash
# Start Axiom Review Bot for human-in-the-loop training

cd /home/toastee/BioMimeticAi

echo "Starting Axiom Review Bot..."
echo "This bot will DM you for axiom reviews"
echo ""

# Activate venv
source venv/bin/activate

# Run bot
python src/discord/bot_axiom_review.py
