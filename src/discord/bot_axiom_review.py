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
from discord.ui import Button, View
from dotenv import load_dotenv
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.daemon.textgen_client import TextGenClient
from src.tensor_axiom.axiom_library import AxiomLibrary
from src.tensor_axiom.review_queue import ReviewQueue
from src.core.dynamic_prompts import get_conversational_prompt, get_axiom_evaluation_prompt
from src.memory.episodic import EpisodicMemory
from src.memory.contact_memory import ContactMemory

# Load environment
load_dotenv("/home/toastee/BioMimeticAi/config/.env")

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OWNER_ID = os.getenv("DISCORD_OWNER_ID")  # Your Discord user ID

# Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.dm_messages = True
intents.members = True

# Create bot
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)
textgen = TextGenClient()
episodic = EpisodicMemory()
contacts = ContactMemory()

# Initialize axiom systems
library = AxiomLibrary("data/axioms/base_axioms.json")
queue = ReviewQueue("data/axioms/review_queue.json", library)

# State
pending_review = None  # Currently displayed axiom
owner_user = None  # Owner DM channel


class QuickInspectView(View):
    """Quick inspect buttons for review queue"""
    def __init__(self, queue_items: list):
        super().__init__(timeout=None)
        self.queue_items = queue_items
        
        # Add buttons for first 5 items
        for i, item in enumerate(queue_items[:5], 1):
            button = Button(
                label=f"#{i} {item.axiom_id[:20]}",
                style=discord.ButtonStyle.primary,
                custom_id=f"inspect_{i}"
            )
            button.callback = self.make_inspect_callback(i)
            self.add_item(button)
    
    def make_inspect_callback(self, index: int):
        async def callback(interaction: discord.Interaction):
            # Trigger inspect command
            await interaction.response.send_message(f"Loading axiom #{index}...")
            # Manually call inspect logic
            global pending_review
            queue.load()
            library.load()
            
            sorted_queue = sorted(queue.queue, key=lambda x: x.priority, reverse=True)
            if index < 1 or index > len(sorted_queue):
                await interaction.followup.send(f"❌ Invalid index. Queue has {len(sorted_queue)} items.")
                return
            
            item = sorted_queue[index - 1]
            pending_review = item.axiom_id
            axiom = library.get_axiom(item.axiom_id)
            
            if not axiom:
                await interaction.followup.send(f"❌ Axiom `{item.axiom_id}` not found in library.")
                return
            
            # Build inspect message
            msg = f"**📋 Axiom: `{item.axiom_id}`**\n\n"
            msg += f"**Name:** {axiom.get('name', 'N/A')}\n"
            msg += f"**Layer:** {axiom.get('layer', 'N/A')}\n"
            msg += f"**Description:** {axiom.get('description', 'N/A')}\n\n"
            
            perf = axiom.get('performance', {})
            if perf:
                msg += f"**Performance:**\n"
                msg += f"Success Rate: {perf.get('success_rate', 0):.1%},\n"
                msg += f"Confidence: {perf.get('avg_confidence', 0):.2f},\n"
                msg += f"Test Count: {perf.get('test_count', 0)},\n"
                msg += f"Last Tested: {perf.get('last_tested', 'Never')}\n\n"
            
            priority_emoji = "🔴" if item.priority >= 3 else "🟡" if item.priority >= 2 else "🟢"
            msg += f"**🔍 Why Flagged for Review:**\n{item.reason}\n\n"
            msg += f"**Priority Level:** {priority_emoji} {item.priority}/5\n"
            
            await interaction.followup.send(msg)
            
            # Send review buttons
            review_view = ReviewButtons(item.axiom_id)
            await interaction.followup.send(
                "**Actions:** Use buttons below or type a test scenario:",
                view=review_view
            )
        
        return callback


class ReviewButtons(View):
    """Interactive buttons for axiom review"""
    def __init__(self, axiom_id: str):
        super().__init__(timeout=None)
        self.axiom_id = axiom_id
    
    @discord.ui.button(label="✅ Approve", style=discord.ButtonStyle.success, custom_id="approve")
    async def approve_button(self, interaction: discord.Interaction, button: Button):
        global pending_review
        queue = ReviewQueue("data/axioms/review_queue.json", library)
        queue.load()
        
        if queue.approve(self.axiom_id):
            await interaction.response.send_message(f"✅ Approved `{self.axiom_id}` and removed from review queue.")
            pending_review = None
            
            # Show updated queue
            queue.load()
            if queue.queue:
                sorted_queue = sorted(queue.queue, key=lambda x: x.priority, reverse=True)
                msg = f"**Updated Review Queue** ({len(queue.queue)} items)\n\n"
                
                for i, item in enumerate(sorted_queue[:10], 1):
                    priority_emoji = "🔴" if item.priority >= 3 else "🟡" if item.priority >= 2 else "🟢"
                    msg += f"{priority_emoji} **{i}.** `{item.axiom_id}`\n"
                    msg += f"   Reason: {item.reason[:100]}\n\n"
                
                if len(queue.queue) > 10:
                    msg += f"_...and {len(queue.queue) - 10} more_\n"
                
                view = QuickInspectView(sorted_queue)
                await interaction.followup.send(msg, view=view)
            else:
                await interaction.followup.send("✅ Review queue is now empty! All axioms performing well.")
        else:
            await interaction.response.send_message(f"❌ Failed to approve `{self.axiom_id}`.")
    
    @discord.ui.button(label="❌ Reject", style=discord.ButtonStyle.danger, custom_id="reject")
    async def reject_button(self, interaction: discord.Interaction, button: Button):
        await interaction.response.send_message(f"Please provide rejection reason: `!reject [reason]`")
    
    @discord.ui.button(label="💡 Suggest", style=discord.ButtonStyle.primary, custom_id="suggest")
    async def suggest_button(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        
        try:
            from src.tensor_axiom.self_training_loop import SelfTrainingLoop
            trainer = SelfTrainingLoop()
            axiom = library.get_axiom(self.axiom_id)
            
            if not axiom:
                await interaction.followup.send(f"❌ Axiom `{self.axiom_id}` not found.")
                return
            
            suggestions = trainer.generate_improvement_suggestions(axiom)
            
            msg = f"**💡 Improvement Suggestions for `{self.axiom_id}`:**\n\n"
            msg += suggestions[:1800]
            
            await interaction.followup.send(msg)
        except Exception as e:
            await interaction.followup.send(f"Error generating suggestions: {e}")
    
    @discord.ui.button(label="🔄 Re-test", style=discord.ButtonStyle.secondary, custom_id="retest")
    async def retest_button(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        
        try:
            from src.tensor_axiom.self_training_loop import SelfTrainingLoop
            trainer = SelfTrainingLoop()
            result = trainer.test_axiom(self.axiom_id)
            
            msg = f"**Test Results for `{self.axiom_id}`:**\n\n"
            msg += f"- Scenarios: {result['total_scenarios']}\n"
            msg += f"- Passed: {result['passed']} ✅\n"
            msg += f"- Failed: {result['failed']} ❌\n"
            msg += f"- Avg Confidence: {result['avg_confidence']:.2f}\n\n"
            
            if result['failed'] == 0:
                msg += "✅ All tests passing! Consider approving.\n"
            else:
                msg += f"⚠️ Still has {result['failed']} failing scenario(s).\n"
                failed = [r for r in result['results'] if not r['success']]
                for i, f in enumerate(failed[:2], 1):
                    msg += f"\n**Failed #{i}:** {f['input'][:100]}\n"
            
            await interaction.followup.send(msg)
        except Exception as e:
            await interaction.followup.send(f"Error during re-test: {e}")
    
    @discord.ui.button(label="⏭️ Skip", style=discord.ButtonStyle.secondary, custom_id="skip")
    async def skip_button(self, interaction: discord.Interaction, button: Button):
        global pending_review
        await interaction.response.send_message(f"Skipped `{self.axiom_id}`.")
        pending_review = None
        
        # Show updated queue
        queue = ReviewQueue("data/axioms/review_queue.json", library)
        queue.load()
        if queue.queue:
            sorted_queue = sorted(queue.queue, key=lambda x: x.priority, reverse=True)
            msg = f"**Review Queue** ({len(queue.queue)} items)\n\n"
            
            for i, item in enumerate(sorted_queue[:10], 1):
                priority_emoji = "🔴" if item.priority >= 3 else "🟡" if item.priority >= 2 else "🟢"
                msg += f"{priority_emoji} **{i}.** `{item.axiom_id}`\n"
                msg += f"   Reason: {item.reason[:100]}\n\n"
            
            if len(queue.queue) > 10:
                msg += f"_...and {len(queue.queue) - 10} more_\n"
            
            view = QuickInspectView(sorted_queue)
            await interaction.followup.send(msg, view=view)


@bot.event
async def on_ready():
    """Bot startup"""
    global owner_user
    
    print(f"Axiom Review Bot online! Logged in as {bot.user}")
    print(f"Connected to {len(bot.guilds)} server(s)")
    print(f"Intents: {bot.intents}")
    print(f"Message Content Intent: {bot.intents.message_content}")
    print(f"Messages Intent: {bot.intents.messages}")
    print(f"DM Messages Intent: {bot.intents.dm_messages}")
    
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
    
    # Send startup help to owner
    if owner_user:
        try:
            queue_size = len(queue.queue)
            
            if queue_size > 0:
                # Queue has items - show full menu with review button
                help_msg = (
                    "🧠 **Axiom Review Bot Online**\n\n"
                    f"⚠️ **{queue_size} axiom{'s' if queue_size != 1 else ''} need review**\n\n"
                    "Use the buttons below to interact with the system."
                )
            else:
                # Empty queue - simpler greeting
                help_msg = (
                    "🧠 **Axiom Review Bot Online**\n\n"
                    "All axioms are performing well right now. I'll notify you if any need attention.\n\n"
                    "**What I do:**\n"
                    "• Monitor axiom performance continuously\n"
                    "• Ask clarifying questions when I'm uncertain\n"
                    "• Learn from your guidance to improve reasoning\n\n"
                    "Feel free to chat with me anytime, or use the buttons below."
                )
            
            # Create main menu buttons
            class MainMenuView(View):
                def __init__(self, show_review_button=True):
                    super().__init__(timeout=None)
                    self.show_review_button = show_review_button
                    
                    # Only add review button if there are items in queue
                    if show_review_button:
                        review_btn = Button(label="📋 Review Queue", style=discord.ButtonStyle.primary, custom_id="main_review")
                        review_btn.callback = self.review_button
                        self.add_item(review_btn)
                    
                    # Always show train button
                    train_btn = Button(label="🧪 Run Training", style=discord.ButtonStyle.success, custom_id="main_train")
                    train_btn.callback = self.train_button
                    self.add_item(train_btn)
                    
                    # Always show stats button
                    stats_btn = Button(label="📊 Stats", style=discord.ButtonStyle.secondary, custom_id="main_stats")
                    stats_btn.callback = self.stats_button
                    self.add_item(stats_btn)
                
                async def review_button(self, interaction: discord.Interaction):
                    await interaction.response.defer()
                    queue.load()
                    
                    if not queue.queue:
                        await interaction.followup.send("✅ Review queue is empty! All axioms performing well.")
                        return
                    
                    sorted_queue = sorted(queue.queue, key=lambda x: x.priority, reverse=True)
                    msg = f"**Axiom Review Queue** ({len(queue.queue)} items)\n\n"
                    
                    for i, item in enumerate(sorted_queue[:10], 1):
                        priority_emoji = "🔴" if item.priority >= 3 else "🟡" if item.priority >= 2 else "🟢"
                        msg += f"{priority_emoji} **{i}.** `{item.axiom_id}`\n"
                        msg += f"   Reason: {item.reason}\n"
                        msg += f"   Added: {item.timestamp[:10]}\n\n"
                    
                    if len(queue.queue) > 10:
                        msg += f"_...and {len(queue.queue) - 10} more_\n"
                    
                    view = QuickInspectView(sorted_queue)
                    await interaction.followup.send(msg, view=view)
                
                async def train_button(self, interaction: discord.Interaction):
                    await interaction.response.defer()
                    await interaction.followup.send("🧪 Running training cycle on all axioms...\nThis will test each axiom and flag any with issues.")
                    
                    try:
                        from src.tensor_axiom.self_training_loop import SelfTrainingLoop
                        trainer = SelfTrainingLoop()
                        
                        # Run one training iteration
                        summary = trainer.run_training_iteration()
                        
                        msg = "**Training Cycle Complete!**\n\n"
                        msg += f"📊 **Results:**\n"
                        msg += f"- Total Scenarios: {summary['total_scenarios']}\n"
                        msg += f"- Pass Rate: {summary['pass_rate']:.1%}\n"
                        msg += f"- Avg Confidence: {summary['avg_confidence']:.2f}\n"
                        msg += f"- Problematic Axioms: {len(summary['problematic'])}\n\n"
                        
                        if summary['problematic']:
                            msg += f"⚠️ Flagged for review: {', '.join([f'`{a}`' for a in summary['problematic'][:5]])}\n"
                            if len(summary['problematic']) > 5:
                                msg += f"_...and {len(summary['problematic']) - 5} more_\n"
                            
                            # Show queue with buttons
                            queue.load()
                            sorted_queue = sorted(queue.queue, key=lambda x: x.priority, reverse=True)
                            view = QuickInspectView(sorted_queue)
                            await interaction.followup.send(msg, view=view)
                        else:
                            msg += "✅ All axioms performing well!"
                            await interaction.followup.send(msg)
                    
                    except Exception as e:
                        await interaction.followup.send(f"❌ Training error: {e}")
                
                async def stats_button(self, interaction: discord.Interaction):
                    await interaction.response.defer()
                    
                    queue.load()
                    library.load()
                    
                    total_axioms = len(library.axioms)
                    queue_size = len(queue.queue)
                    
                    priority_counts = {"high": 0, "medium": 0, "low": 0}
                    for item in queue.queue:
                        if item.priority >= 3:
                            priority_counts["high"] += 1
                        elif item.priority >= 2:
                            priority_counts["medium"] += 1
                        else:
                            priority_counts["low"] += 1
                    
                    msg = "**📊 Axiom System Statistics**\n\n"
                    msg += f"**Library:**\n"
                    msg += f"- Total Axioms: {total_axioms}\n"
                    msg += f"- In Review: {queue_size}\n"
                    msg += f"- Performing Well: {total_axioms - queue_size}\n\n"
                    msg += f"**Review Queue Priority:**\n"
                    msg += f"- 🔴 Critical: {priority_counts['high']}\n"
                    msg += f"- 🟡 Medium: {priority_counts['medium']}\n"
                    msg += f"- 🟢 Low: {priority_counts['low']}\n"
                    
                    await interaction.followup.send(msg)
            
            await owner_user.send(help_msg, view=MainMenuView(show_review_button=(queue_size > 0)))
        except Exception as e:
            print(f"Could not send startup message: {e}")


@bot.event
async def on_message(message):
    """Handle both commands and natural language testing input"""
    global pending_review
    
    # Dump raw message info
    print("\n" + "="*80)
    print("[RAW MESSAGE PACKET]")
    print(f"  ID: {message.id}")
    print(f"  Author: {message.author} (ID: {message.author.id})")
    print(f"  Channel: {message.channel} (Type: {type(message.channel).__name__})")
    print(f"  Content: {repr(message.content[:200])}{'...' if len(message.content) > 200 else ''}")
    print(f"  Is DM: {isinstance(message.channel, discord.DMChannel)}")
    print(f"  Bot User: {bot.user} (ID: {bot.user.id if bot.user else 'None'})")
    print(f"  Owner ID: {OWNER_ID}")
    print(f"  Timestamp: {message.created_at}")
    print("="*80 + "\n")
    
    # Ignore own messages (but we logged them above for debugging)
    if message.author == bot.user:
        return
    
    # Update contact profile for DM users
    if isinstance(message.channel, discord.DMChannel) and not message.author.bot:
        try:
            profile = contacts.get_or_create_contact(
                user_id=str(message.author.id),
                username=message.author.name,
                display_name=message.author.display_name
            )
            print(f"[CONTACT] Updated profile for {message.author.name} (interactions: {profile['interaction_count']}, trust: {profile['trust_level']:.2f})")
        except Exception as e:
            print(f"[CONTACT ERROR] Failed to update contact: {e}")
    
    # Check if it's a DM
    if not isinstance(message.channel, discord.DMChannel):
        print(f"[NOT DM] Channel type: {type(message.channel).__name__}")
        await bot.process_commands(message)
        return
    
    # Check if it's from owner
    if message.author.id != int(OWNER_ID or 0):
        print(f"[NOT OWNER] Author ID: {message.author.id}, Owner ID: {OWNER_ID}")
        return
    
    print(f"[DM FROM OWNER] {message.author.name}: {message.content}")
    
    # Process commands first
    if message.content.startswith('!'):
        print("[COMMAND] Processing command")
        await bot.process_commands(message)
        return
    
    # If there's a pending review and user sends non-command text, treat it as test input
    if pending_review and len(message.content.strip()) > 0:
        print(f"[TESTING] Axiom {pending_review} with user input")
        await message.channel.send(f"🧪 Testing `{pending_review}` with your input...")
        
        try:
            from src.tensor_axiom.self_training_loop import SelfTrainingLoop
            trainer = SelfTrainingLoop()
            axiom = library.get_axiom(pending_review)
            
            if not axiom:
                await message.channel.send(f"❌ Axiom `{pending_review}` not found.")
                return
            
            # Use the user's input as a test scenario
            success, confidence, reasoning = trainer.evaluate_axiom_with_llm(
                axiom, 
                message.content
            )
            
            result_emoji = "✅" if success else "❌"
            msg = f"{result_emoji} **Test Result:**\n\n"
            msg += f"- Success: {'Yes' if success else 'No'}\n"
            msg += f"- Confidence: {confidence:.2f}\n\n"
            msg += f"**LLM Reasoning:**\n{reasoning[:1500]}"
            
            await message.channel.send(msg)
        
        except Exception as e:
            await message.channel.send(f"Error testing axiom: {e}")
    else:
        # No pending review - general conversation with AI
        print(f"[CONVERSATION] Generating response to: {message.content[:50]}")
        try:
            async with message.channel.typing():
                # Get recent conversation history for context
                recent_episodes = episodic.get_recent_episodes(
                    limit=5,
                    user_id=str(message.author.id)
                )
                
                # Build conversation context
                conversation_context = ""
                if recent_episodes:
                    conversation_context = "\n\nRecent conversation history:\n"
                    for ep in reversed(recent_episodes):  # Oldest first
                        conversation_context += f"User: {ep['user_message'][:100]}\n"
                        conversation_context += f"You: {ep['bot_response'][:100]}\n\n"
                
                # Build full prompt with context
                full_prompt = message.content
                if conversation_context:
                    full_prompt = f"{conversation_context}Current message:\n{message.content}"
                
                # Get dynamic system prompt with contact-aware context
                user_id = str(message.author.id)
                dynamic_system_prompt = get_conversational_prompt(user_id=user_id)
                
                print(f"[PROMPT] Using dynamic prompt with contact context for user {user_id}")
                
                # Use TextGenClient for conversational response
                response = textgen.generate(
                    full_prompt,
                    system_prompt=dynamic_system_prompt,
                    max_tokens=500,
                    temperature=0.8
                )
                
                print(f"[RESPONSE] Generated {len(response) if response else 0} chars")
                
                if response:
                    # Split long responses if needed
                    if len(response) > 1900:
                        chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
                        full_response = response  # Store full before chunking
                        for chunk in chunks:
                            await message.channel.send(chunk)
                    else:
                        await message.channel.send(response)
                        full_response = response
                    
                    # Store episode in memory
                    try:
                        episodic.store_episode(
                            user_id=str(message.author.id),
                            username=message.author.name,
                            user_message=message.content,
                            bot_response=full_response,
                            hemisphere="social",
                            salience_score=1.0
                        )
                        print(f"[MEMORY] Stored conversation episode")
                    except Exception as e:
                        print(f"[MEMORY ERROR] Failed to store episode: {e}")
                else:
                    await message.channel.send("🤔 I'm having trouble thinking right now... LLM connection issue.")
        
        except Exception as e:
            print(f"[ERROR] Conversation error: {e}")
            await message.channel.send(f"💭 Sorry, I encountered an error: {e}")
    
    # Always process commands after handling messages
    await bot.process_commands(message)


@tasks.loop(minutes=5)
async def check_review_queue():
    """Periodically check for items needing review"""
    global pending_review
    
    if not owner_user:
        return
    
    # Reload queue
    queue.load()
    
    # Get high priority items
    high_priority = [item for item in queue.queue if item.priority >= 3]
    
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
    sorted_queue = sorted(queue.queue, key=lambda x: x.priority, reverse=True)
    
    msg = f"**Axiom Review Queue** ({len(queue.queue)} items)\n\n"
    
    for i, item in enumerate(sorted_queue[:10], 1):
        priority_emoji = "🔴" if item.priority >= 3 else "🟡" if item.priority >= 2 else "🟢"
        msg += f"{priority_emoji} **{i}.** `{item.axiom_id}`\n"
        msg += f"   Reason: {item.reason}\n"
        msg += f"   Added: {item.timestamp[:10]}\n\n"
    
    if len(queue.queue) > 10:
        msg += f"_...and {len(queue.queue) - 10} more_\n\n"
    
    msg += f"Use `!inspect <number>` to review an axiom"
    
    # Add quick inspect buttons
    view = QuickInspectView(sorted_queue)
    await ctx.send(msg, view=view)


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
    sorted_queue = sorted(queue.queue, key=lambda x: x.priority, reverse=True)
    item = sorted_queue[index - 1]
    axiom_id = item.axiom_id
    
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
    msg += f"**🔍 Why Flagged for Review:**\n"
    msg += f"{item.reason}\n"
    msg += f"**Priority Level:** {item.priority}/5 {'🔴 CRITICAL' if item.priority >= 3 else '🟡 MEDIUM' if item.priority >= 2 else '🟢 LOW'}\n\n"
    
    # Show clarification questions if available
    if hasattr(item, 'clarification_questions') and item.clarification_questions:
        msg += f"**❓ Clarification Needed:**\n"
        for i, q in enumerate(item.clarification_questions, 1):
            msg += f"{i}. {q}\n"
        msg += f"\n💬 Type your answers in this DM to help refine the axiom.\n\n"
    
    # Show failed test details if available
    if item.failed_tests and len(item.failed_tests) > 0:
        msg += f"**Failed Tests:** {len(item.failed_tests)} scenario(s)\n\n"
    
    await ctx.send(msg)
    
    # Send failed test details if available
    if item.failed_tests and len(item.failed_tests) > 0:
        fail_msg = f"**❌ Failed Test Results:**\n\n"
        for i, test in enumerate(item.failed_tests[:2], 1):
            fail_msg += f"**Test {i}:**\n"
            if isinstance(test, dict):
                fail_msg += f"- Input: {test.get('input', 'N/A')[:100]}\n"
                fail_msg += f"- Reasoning: {test.get('reasoning', 'N/A')[:200]}\n\n"
        
        if len(item.failed_tests) > 2:
            fail_msg += f"_...and {len(item.failed_tests) - 2} more failures_\n\n"
        
        await ctx.send(fail_msg)
    
    # Send test scenarios in separate message
    scenarios = axiom.get('test_scenarios', [])
    if scenarios:
        scenario_msg = f"**Test Scenarios for `{axiom_id}`:**\n\n"
        for i, scenario in enumerate(scenarios[:3], 1):
            scenario_msg += f"**{i}.** {scenario.get('input', 'N/A')}\n"
            scenario_msg += f"   Expected: {scenario.get('expected_behavior', 'N/A')}\n"
            scenario_msg += f"   Criteria: {scenario.get('success_criteria', 'N/A')}\n\n"
        
        await ctx.send(scenario_msg)
    
    # Interactive buttons
    view = ReviewButtons(item.axiom_id)
    await ctx.send(
        f"**Actions:** Use buttons below or commands:\n"
        f"`!reject [reason]` - Mark as flawed with reason\n"
        f"`!test [your input]` - Test axiom with your scenario",
        view=view
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
    queue.queue = [item for item in queue.queue if item.axiom_id != axiom_id]
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
    
    # Show updated queue
    queue.load()
    if queue.queue:
        sorted_queue = sorted(queue.queue, key=lambda x: x.priority, reverse=True)
        msg = f"**Updated Review Queue** ({len(queue.queue)} items)\n\n"
        
        for i, item in enumerate(sorted_queue[:10], 1):
            priority_emoji = "🔴" if item.priority >= 3 else "🟡" if item.priority >= 2 else "🟢"
            msg += f"{priority_emoji} **{i}.** `{item.axiom_id}`\n"
            msg += f"   Reason: {item.reason[:100]}\n\n"
        
        if len(queue.queue) > 10:
            msg += f"_...and {len(queue.queue) - 10} more_\n"
        
        view = QuickInspectView(sorted_queue)
        await ctx.send(msg, view=view)
    else:
        await ctx.send("✅ Review queue is now empty! All axioms performing well.")


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
        if item.axiom_id == axiom_id:
            item.status = 'rejected'
            item.reviewer_notes = reason
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
            system_prompt=get_axiom_improvement_prompt(),
            max_tokens=400,
            temperature=0.7
        )
        
        if suggestions:
            await ctx.send(f"**Suggestions for `{axiom_id}`:**\n\n{suggestions}")
        else:
            await ctx.send("Could not generate suggestions (LLM error)")
    
    except Exception as e:
        await ctx.send(f"Error generating suggestions: {e}")


@bot.command(name="retest")
async def retest(ctx):
    """Re-run tests on the current axiom"""
    global pending_review
    
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    if not pending_review:
        await ctx.send("No axiom currently under review. Use `!inspect` first.")
        return
    
    await ctx.send(f"🔄 Re-testing `{pending_review}`...")
    
    try:
        # Import and run test
        from src.tensor_axiom.self_training_loop import SelfTrainingLoop
        trainer = SelfTrainingLoop()
        
        # Test just this axiom
        result = trainer.test_axiom(pending_review)
        
        # Report results
        msg = f"**Test Results for `{pending_review}`:**\n\n"
        msg += f"- Scenarios: {result['total_scenarios']}\n"
        msg += f"- Passed: {result['passed']} ✅\n"
        msg += f"- Failed: {result['failed']} ❌\n"
        msg += f"- Avg Confidence: {result['avg_confidence']:.2f}\n\n"
        
        if result['failed'] == 0:
            msg += "✅ All tests passing! Consider approving this axiom.\n"
        else:
            msg += f"⚠️ Still has {result['failed']} failing scenario(s).\n"
            
            # Show failed scenarios
            failed = [r for r in result['results'] if not r['success']]
            for i, f in enumerate(failed[:2], 1):
                msg += f"\n**Failed #{i}:**\n"
                msg += f"- Input: {f['input'][:100]}\n"
                msg += f"- Reasoning: {f['reasoning'][:150]}\n"
        
        await ctx.send(msg)
        
    except Exception as e:
        await ctx.send(f"Error during re-test: {e}")


@bot.command(name="skip")
async def skip(ctx):
    """Skip the current axiom"""
    global pending_review
    
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    if not pending_review:
        await ctx.send("No axiom currently under review.")
        return
    
    await ctx.send(f"Skipped `{pending_review}`.")
    pending_review = None
    
    # Show updated queue
    queue.load()
    if queue.queue:
        sorted_queue = sorted(queue.queue, key=lambda x: x.priority, reverse=True)
        msg = f"**Review Queue** ({len(queue.queue)} items)\n\n"
        
        for i, item in enumerate(sorted_queue[:10], 1):
            priority_emoji = "🔴" if item.priority >= 3 else "🟡" if item.priority >= 2 else "🟢"
            msg += f"{priority_emoji} **{i}.** `{item.axiom_id}`\n"
            msg += f"   Reason: {item.reason[:100]}\n\n"
        
        if len(queue.queue) > 10:
            msg += f"_...and {len(queue.queue) - 10} more_\n"
        
        view = QuickInspectView(sorted_queue)
        await ctx.send(msg, view=view)


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
    msg += f"- High Priority: {sum(1 for x in queue.queue if x.priority >= 3)}\n\n"
    
    msg += f"**Categories:**\n"
    for cat, count in stats['categories'].items():
        msg += f"- {cat}: {count}\n"
    
    await ctx.send(msg)


@bot.command(name="profile")
async def profile(ctx):
    """Show your contact profile"""
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    user_id = str(ctx.author.id)
    profile = contacts.get_contact(user_id)
    
    if not profile:
        await ctx.send("No profile found. Start chatting to create one!")
        return
    
    summary = contacts.get_relationship_summary(user_id)
    
    # Add context notes if any
    if profile.get('context_notes'):
        summary += f"\n**Notes:**\n{profile['context_notes'][-500:]}"  # Last 500 chars
    
    await ctx.send(summary)


@bot.command(name="note")
async def add_note(ctx, *, note_text: str = None):
    """Add a context note to your profile (!note <text>)"""
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    if not note_text:
        await ctx.send("Usage: `!note <your note text>`")
        return
    
    user_id = str(ctx.author.id)
    success = contacts.add_context_note(user_id, note_text)
    
    if success:
        await ctx.send(f"✅ Note added to your profile!")
    else:
        await ctx.send(f"❌ Failed to add note. Do you have a profile?")


@bot.command(name="contacts")
async def list_contacts(ctx):
    """List all contacts the bot knows"""
    if ctx.author.id != int(OWNER_ID or 0):
        return
    
    all_contacts = contacts.list_contacts(limit=20)
    
    if not all_contacts:
        await ctx.send("No contacts yet!")
        return
    
    msg = f"**📇 Contact List ({len(all_contacts)} contacts)**\n\n"
    
    for contact in all_contacts:
        username = contact['username']
        trust = contact['trust_level']
        interactions = contact['interaction_count']
        rel_type = contact.get('relationship_type', 'unknown')
        
        trust_emoji = "🟢" if trust > 0.7 else "🟡" if trust > 0.4 else "🔴"
        msg += f"{trust_emoji} **{username}** - {interactions} msgs, trust: {trust:.2f}"
        if rel_type != 'unknown':
            msg += f" ({rel_type})"
        msg += "\n"
    
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
