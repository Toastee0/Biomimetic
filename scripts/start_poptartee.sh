#!/bin/bash
# Start PopTartee bot

cd ~/BioMimeticAi
source venv/bin/activate

# Kill any existing instances
pkill -f "python.*bot.py" 2>/dev/null

# Start bot
echo "Starting PopTartee..."
python -u src/discord/bot.py >> logs/poptartee.log 2>&1 &

echo "PopTartee started. Check logs with: tail -f ~/BioMimeticAi/logs/poptartee.log"
