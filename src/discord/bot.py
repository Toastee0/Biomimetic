#!/usr/bin/env python3
"""PopTartee Discord Bot - Production Version with Memory System"""

import os
import sys
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.daemon.textgen_client import TextGenClient
from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.memory.identity import IdentityCore

# Load environment
load_dotenv("/home/toastee/BioMimeticAi/config/.env")

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

# Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

# Create bot
bot = commands.Bot(command_prefix="!", intents=intents)
textgen = TextGenClient()

# Initialize memory systems
episodic_memory = EpisodicMemory()
semantic_memory = SemanticMemory()
identity = IdentityCore()

# Load comprehensive system prompt from config
SYSTEM_PROMPT_FILE = "/home/toastee/BioMimeticAi/config/system_prompt.txt"
try:
    with open(SYSTEM_PROMPT_FILE, "r") as f:
        BASE_SYSTEM_PROMPT = f.read()
    print(f"Loaded comprehensive system prompt ({len(BASE_SYSTEM_PROMPT)} chars)")
except FileNotFoundError:
    BASE_SYSTEM_PROMPT = "You are PopTartee, a helpful AI assistant."
    print("WARNING: Using fallback system prompt")

# Add identity context to system prompt
IDENTITY_CONTEXT = identity.get_context_summary()
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + IDENTITY_CONTEXT

print(f"Memory systems initialized")
print(f"  - Episodes stored: {episodic_memory.get_episode_count()}")
print(f"  - Identity: {identity.get_name()}")

@bot.event
async def on_ready():
    """Bot startup event"""
    print(f"PopTartee is online! Logged in as {bot.user}")
    print(f"Connected to {len(bot.guilds)} server(s)")
    print(f"Listening to channel: {CHANNEL_ID if CHANNEL_ID else 'ALL'}")

    if textgen.test_connection():
        print("Text generation API connected successfully")
    else:
        print("WARNING: Text generation API connection failed!")

    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="neurons fire 🧠"
        )
    )

@bot.event
async def on_message(message):
    """Handle incoming messages with memory integration"""

    if message.author == bot.user:
        return

    if CHANNEL_ID and str(message.channel.id) != CHANNEL_ID:
        await bot.process_commands(message)
        return

    bot_mentioned = bot.user in message.mentions
    is_dm = isinstance(message.channel, discord.DMChannel)

    if bot_mentioned or is_dm or not CHANNEL_ID:
        async with message.channel.typing():
            try:
                user_id = str(message.author.id)
                username = message.author.name
                user_message = message.content

                if bot_mentioned:
                    user_message = user_message.replace(f"<@{bot.user.id}>", "").strip()

                if not user_message:
                    return

                # Update user in semantic memory
                semantic_memory.store_user(user_id, username)

                # Retrieve relevant memories using cue-based recall (biomimetic!)
                triggered_memories = episodic_memory.retrieve_by_cue(
                    user_message,
                    user_id=user_id,
                    limit=3
                )

                # Build context from triggered memories
                memory_context = ""
                if triggered_memories:
                    memory_context = "\n\n[Recalled memories triggered by this conversation:]"
                    for mem in triggered_memories:
                        memory_context += f"\n- User once said: \"{mem['user_message']}\""
                        memory_context += f"\n  You replied: \"{mem['bot_response']}\""

                # Construct full prompt with memory context
                full_prompt = SYSTEM_PROMPT
                if memory_context:
                    full_prompt += memory_context

                # Generate response
                response = textgen.generate(
                    user_message,
                    system_prompt=full_prompt,
                    max_tokens=500,
                    temperature=0.8
                )

                if response:
                    # Store this exchange in episodic memory
                    episodic_memory.store_episode(
                        user_id=user_id,
                        username=username,
                        user_message=user_message,
                        bot_response=response,
                        hemisphere="social",  # MVP: always social for now
                        salience_score=1.0 if bot_mentioned else 0.7
                    )

                    # Send response
                    if len(response) > 2000:
                        chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                        for chunk in chunks:
                            await message.reply(chunk)
                    else:
                        await message.reply(response)
                else:
                    await message.reply("Hmm, I seem to be having trouble thinking right now. Give me a moment...")

            except Exception as e:
                print(f"Error processing message: {e}")
                import traceback
                traceback.print_exc()
                await message.reply("Oops, something went wrong in my neural circuits!")

    await bot.process_commands(message)

@bot.command(name="ping")
async def ping(ctx):
    """Check bot responsiveness"""
    latency = round(bot.latency * 1000)
    await ctx.reply(f"Pong! Latency: {latency}ms")

@bot.command(name="status")
async def status(ctx):
    """Show bot status"""
    models = textgen.list_models()
    model_name = models[0].split("/")[-1] if models else "Unknown"

    # Get memory stats
    total_episodes = episodic_memory.get_episode_count()
    user_episodes = episodic_memory.get_episode_count(user_id=str(ctx.author.id))

    status_msg = f"""**PopTartee Status**

**Model**: {model_name}
**Latency**: {round(bot.latency * 1000)}ms
**Servers**: {len(bot.guilds)}
**Version**: MVP 0.1.0
**Memory**: {total_episodes} episodes stored ({user_episodes} with you)
**Hemisphere**: Social (RP-Max)
**System Prompt**: Comprehensive safety ({len(BASE_SYSTEM_PROMPT)} chars)

*Self-hosted on core server with 3090*"""

    await ctx.reply(status_msg)

@bot.command(name="memory")
async def memory_stats(ctx):
    """Show memory statistics"""
    user_id = str(ctx.author.id)

    # Get stats
    total_episodes = episodic_memory.get_episode_count()
    user_episodes = episodic_memory.get_episode_count(user_id=user_id)
    user_info = semantic_memory.get_user(user_id)

    trust = user_info['trust_score'] if user_info else 0.5
    interactions = user_info['interaction_count'] if user_info else 0

    msg = f"""**Memory Statistics**

**Total conversations**: {total_episodes} episodes
**Conversations with you**: {user_episodes} episodes
**Trust score**: {trust:.2f} / 1.0
**Interaction count**: {interactions}

**Memory system**: Biomimetic episodic recall
*Memories triggered by environmental cues, not chronological recall*"""

    await ctx.reply(msg)

@bot.command(name="about")
async def about(ctx):
    """Show information about PopTartee"""
    about_msg = """**About PopTartee**

I am a biomimetic AI assistant in early development, focused on learning persistent memory and natural human-AI collaboration.

**How to interact:**
- Mention me: `@PopTartee your message`
- DM me directly
- Or just chat in designated channels

**Current capabilities:**
- Natural conversation using Mistral-Small-22B RP-Max
- **Persistent memory** - I remember our conversations!
- **Cue-based recall** - Memories triggered by context (like biological memory)
- Honest, direct communication
- Self-aware of AI nature and limitations
- Comprehensive safety boundaries

**In development:**
- Dual-hemisphere architecture (analytical + social)
- Temporal attention decay
- Relationship tracking
- Memory consolidation

Commands: `!ping` `!status` `!memory` `!about`

*Built by toastee0 - Running on dedicated 3090 server*"""

    await ctx.reply(about_msg)

if __name__ == "__main__":
    if not BOT_TOKEN:
        print("ERROR: DISCORD_BOT_TOKEN not found in environment!")
        sys.exit(1)

    print("Starting PopTartee Discord bot with memory...")

    try:
        bot.run(BOT_TOKEN)
    except KeyboardInterrupt:
        print("Bot shutdown requested")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
