#!/bin/bash
# Stop PopTartee bot

echo "Stopping PopTartee..."
pkill -f "python.*bot.py"
echo "PopTartee stopped."
