#!/bin/bash
# Quick start script for complete axiom training system

echo "🧠 Starting Axiom Training System"
echo "=================================="
echo ""

# Check if DISCORD_OWNER_ID is set
if ! grep -q "DISCORD_OWNER_ID" config/.env; then
    echo "⚠️  DISCORD_OWNER_ID not found in config/.env"
    echo "Please add your Discord User ID:"
    echo "  echo 'DISCORD_OWNER_ID=YOUR_ID_HERE' >> config/.env"
    echo ""
    exit 1
fi

cd /home/toastee/BioMimeticAi
source venv/bin/activate

echo "Starting Review Bot in background..."
nohup python src/discord/bot_axiom_review.py > logs/review_bot.log 2>&1 &
REVIEW_PID=$!
echo "  Review Bot PID: $REVIEW_PID"

echo ""
echo "Starting Self-Training Loop..."
python src/tensor_axiom/self_training_loop.py --iterations 0 --interval 300

# Cleanup on exit
kill $REVIEW_PID 2>/dev/null
