#!/usr/bin/env python3
"""
Axiom Review Bot - DM-based interface for human review of axiom system

This bot:
- DMs you when axioms need review
- Presents axiom details and test results
- Accepts your feedback (approve/reject/improve)
- Integrates with self-training loop
"""

import os
import sys
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.daemon.textgen_client import TextGenClient
from src.tensor_axiom.axiom_library import AxiomLibrary
from src.tensor_axiom.review_queue import ReviewQueue

# Load environment
load_dotenv("/home/toastee/BioMimeticAi/config/.env")

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OWNER_ID = os.getenv("DISCORD_OWNER_ID")  # Your Discord user ID

# Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.dm_messages = True

# Create bot
bot = commands.Bot(command_prefix="!", intents=intents)
textgen = TextGenClient()

# Initialize axiom systems
library = AxiomLibrary("data/axioms/base_axioms.json")
queue = ReviewQueue("data/axioms/review_queue.json", library)

# State
pending_review = None  # Currently displayed axiom
owner_user = None  # Owner DM channel


@bot.event
async def on_ready():
    """Bot startup"""
    global owner_user
    
    print(f"Axiom Review Bot online! Logged in as {bot.user}")
    print(f"Connected to {len(bot.guilds)} server(s)")
    
    # Get owner user for DMs
    if OWNER_ID:
        try:
            owner_user = await bot.fetch_user(int(OWNER_ID))
            print(f"Found owner: {owner_user.name}")
        except Exception as e:
            print(f"Could not find owner user: {e}")
    
    if textgen.test_connection():
        print("LLM inference connected")
    else:
        print("WARNING: LLM inference not available")
    
    # Load axiom library
    library.load()
    queue.load()
    
    print(f"Loaded {len(library.axioms)} axioms")
    print(f"Review queue: {len(queue.queue)} items")
    
    # Start periodic check
    check_review_queue.start()
    
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="axiom cortex 🧠"
        )
    )


@tasks.loop(minutes=5)
async def check_review_queue():
    """Periodically check for items needing review"""
    global pending_review
    
    if not owner_user:
        return
    
    # Reload queue
    queue.load()
    
    # Get high priority items
    high_priority = [item for item in queue.queue if item['priority'] >= 3]
    
    if high_priority and not pending_review:
        # Send notification
        try:
            await owner_user.send(
                f"🔔 **Axiom Review Notification**\n\n"
                f"There are **{len(high_priority)}** high-priority axioms needing review.\n"
                f"Use `!review` to see the queue."
            )
        except Exception as e:
            print(f"Could not send DM: {e}")


@bot.event
async def on_message(message):
    """Handle DM messages"""
    # Only process DMs from owner
    if message.author.id != int(OWNER_ID or 0):
        await bot.process_commands(message)
        return
    
    # Only process DMs
    if not isinstance(message.channel, discord.DMChannel):
        await bot.process_commands(message)
        return
    
    # Skip bot's own messages
    if message.author == bot.user:
        return
    
    # Process commands
    await bot.process_commands(message)


@bot.command(name="review")
async def review(ctx):
    """Show review queue"""
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    queue.load()
    
    if not queue.queue:
        await ctx.send("✅ Review queue is empty! All axioms performing well.")
        return
    
    # Sort by priority
    sorted_queue = sorted(queue.queue, key=lambda x: x['priority'], reverse=True)
    
    msg = f"**Axiom Review Queue** ({len(queue.queue)} items)\n\n"
    
    for i, item in enumerate(sorted_queue[:10], 1):
        priority_emoji = "🔴" if item['priority'] >= 3 else "🟡" if item['priority'] >= 2 else "🟢"
        msg += f"{priority_emoji} **{i}.** `{item['axiom_id']}`\n"
        msg += f"   Reason: {item['reason']}\n"
        msg += f"   Added: {item['timestamp'][:10]}\n\n"
    
    if len(queue.queue) > 10:
        msg += f"_...and {len(queue.queue) - 10} more_\n\n"
    
    msg += f"Use `!inspect <number>` to review an axiom"
    
    await ctx.send(msg)


@bot.command(name="inspect")
async def inspect(ctx, index: int = None):
    """Inspect a specific axiom from the review queue"""
    global pending_review
    
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    queue.load()
    library.load()
    
    if not queue.queue:
        await ctx.send("Review queue is empty!")
        return
    
    if index is None:
        index = 1
    
    if index < 1 or index > len(queue.queue):
        await ctx.send(f"Invalid index. Use 1-{len(queue.queue)}")
        return
    
    # Sort by priority and get item
    sorted_queue = sorted(queue.queue, key=lambda x: x['priority'], reverse=True)
    item = sorted_queue[index - 1]
    axiom_id = item['axiom_id']
    
    if axiom_id not in library.axioms:
        await ctx.send(f"Axiom `{axiom_id}` not found in library!")
        return
    
    axiom = library.axioms[axiom_id]
    pending_review = axiom_id
    
    # Build detailed report
    msg = f"**Axiom Review: `{axiom_id}`**\n\n"
    msg += f"**Name:** {axiom['name']}\n"
    msg += f"**Layer:** {axiom.get('layer', 'unknown')} (priority {axiom['priority']})\n"
    msg += f"**Description:** {axiom['description']}\n\n"
    
    if 'formula' in axiom:
        msg += f"**Formula:** `{axiom['formula']}`\n\n"
    
    # Performance metrics
    metrics = axiom.get('performance_metrics', {})
    msg += f"**Performance:**\n"
    msg += f"- Success Rate: {metrics.get('success_rate', 0):.1%}\n"
    msg += f"- Confidence: {metrics.get('avg_confidence', 0):.2f}\n"
    msg += f"- Test Count: {metrics.get('test_count', 0)}\n"
    msg += f"- Last Tested: {metrics.get('last_tested', 'Never')[:19]}\n\n"
    
    # Review queue reason
    msg += f"**Review Reason:** {item['reason']}\n"
    msg += f"**Priority:** {item['priority']}/5\n\n"
    
    await ctx.send(msg)
    
    # Send test scenarios in separate message
    scenarios = axiom.get('test_scenarios', [])
    if scenarios:
        scenario_msg = f"**Test Scenarios for `{axiom_id}`:**\n\n"
        for i, scenario in enumerate(scenarios[:3], 1):
            scenario_msg += f"**{i}.** {scenario.get('input', 'N/A')}\n"
            scenario_msg += f"   Expected: {scenario.get('expected_behavior', 'N/A')}\n"
            scenario_msg += f"   Criteria: {scenario.get('success_criteria', 'N/A')}\n\n"
        
        await ctx.send(scenario_msg)
    
    # Commands
    await ctx.send(
        f"**Actions:**\n"
        f"`!approve` - Axiom is correct, remove from queue\n"
        f"`!reject` - Axiom is flawed, needs rework\n"
        f"`!suggest` - Get LLM suggestions for improvement\n"
        f"`!skip` - Skip to next axiom"
    )


@bot.command(name="approve")
async def approve(ctx):
    """Approve the currently inspected axiom"""
    global pending_review
    
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    if not pending_review:
        await ctx.send("No axiom currently under review. Use `!inspect` first.")
        return
    
    axiom_id = pending_review
    
    # Remove from queue
    queue.load()
    queue.queue = [item for item in queue.queue if item['axiom_id'] != axiom_id]
    queue.save()
    
    # Update metrics
    library.load()
    if axiom_id in library.axioms:
        metrics = library.axioms[axiom_id].get('performance_metrics', {})
        metrics['human_approval_rate'] = metrics.get('human_approval_rate', 0.0) + 0.1
        library.update_metrics(axiom_id, metrics)
        library.save()
    
    await ctx.send(f"✅ Approved `{axiom_id}` and removed from review queue.")
    
    pending_review = None


@bot.command(name="reject")
async def reject(ctx, *, reason: str = "Needs improvement"):
    """Reject the currently inspected axiom"""
    global pending_review
    
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    if not pending_review:
        await ctx.send("No axiom currently under review. Use `!inspect` first.")
        return
    
    axiom_id = pending_review
    
    # Update queue with rejection reason
    queue.load()
    for item in queue.queue:
        if item['axiom_id'] == axiom_id:
            item['status'] = 'rejected'
            item['notes'] = reason
            break
    queue.save()
    
    await ctx.send(f"❌ Rejected `{axiom_id}`: {reason}\n\nMarked for rework.")
    
    pending_review = None


@bot.command(name="suggest")
async def suggest(ctx):
    """Get LLM suggestions for improving the current axiom"""
    global pending_review
    
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    if not pending_review:
        await ctx.send("No axiom currently under review. Use `!inspect` first.")
        return
    
    await ctx.send("🤔 Generating improvement suggestions...")
    
    axiom_id = pending_review
    library.load()
    axiom = library.axioms[axiom_id]
    
    # Build prompt for suggestions
    prompt = f"""Analyze this axiom and suggest improvements:

AXIOM: {axiom['name']}
DESCRIPTION: {axiom['description']}
FORMULA: {axiom.get('formula', 'N/A')}
PRIORITY: {axiom['priority']}

PERFORMANCE:
- Success Rate: {axiom.get('performance_metrics', {}).get('success_rate', 0):.1%}
- Confidence: {axiom.get('performance_metrics', {}).get('avg_confidence', 0):.2f}

Provide 2-3 specific suggestions to improve this axiom's performance."""
    
    try:
        suggestions = textgen.generate(
            prompt,
            system_prompt="You are an axiom improvement advisor for a biomimetic AI reasoning system.",
            max_tokens=400,
            temperature=0.7
        )
        
        if suggestions:
            await ctx.send(f"**Suggestions for `{axiom_id}`:**\n\n{suggestions}")
        else:
            await ctx.send("Could not generate suggestions (LLM error)")
    
    except Exception as e:
        await ctx.send(f"Error generating suggestions: {e}")


@bot.command(name="skip")
async def skip(ctx):
    """Skip the current axiom"""
    global pending_review
    
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    if not pending_review:
        await ctx.send("No axiom currently under review.")
        return
    
    await ctx.send(f"Skipped `{pending_review}`. Use `!review` to see the queue.")
    pending_review = None


@bot.command(name="stats")
async def stats(ctx):
    """Show axiom system statistics"""
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    library.load()
    queue.load()
    
    stats = library.get_stats()
    
    msg = f"**Axiom System Statistics**\n\n"
    msg += f"**Library:**\n"
    msg += f"- Total Axioms: {stats['total_axioms']}\n"
    msg += f"- Tested: {stats['tested_axioms']}\n"
    msg += f"- Avg Success: {stats['average_success_rate']:.1%}\n\n"
    
    msg += f"**Review Queue:**\n"
    msg += f"- Pending: {len(queue.queue)}\n"
    msg += f"- High Priority: {sum(1 for x in queue.queue if x['priority'] >= 3)}\n\n"
    
    msg += f"**Categories:**\n"
    for cat, count in stats['categories'].items():
        msg += f"- {cat}: {count}\n"
    
    await ctx.send(msg)


if __name__ == "__main__":
    if not BOT_TOKEN:
        print("ERROR: DISCORD_BOT_TOKEN not found!")
        sys.exit(1)
    
    if not OWNER_ID:
        print("WARNING: DISCORD_OWNER_ID not set - bot will not send DMs")
    
    print("Starting Axiom Review Bot...")
    
    try:
        bot.run(BOT_TOKEN)
    except KeyboardInterrupt:
        print("\nBot shutdown requested")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
