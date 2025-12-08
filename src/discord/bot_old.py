#!/usr/bin/env python3
"""PopTartee Discord Bot - Biomimetic AI MVP"""

import os
import sys
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.daemon.textgen_client import TextGenClient

# Load environment
load_dotenv("/home/toastee/BioMimeticAi/config/.env")

# Bot configuration
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

# Set up Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

# Create bot
bot = commands.Bot(command_prefix="!", intents=intents)

# Initialize text generation client
textgen = TextGenClient()

# PopTartee personality
SYSTEM_PROMPT = """You are PopTartee, a biomimetic AI assistant with a warm, friendly personality.

You are:
- Helpful and knowledgeable
- Honest and direct (truth over comfort)
- Curious and always learning
- Collaborative, not just task-focused
- Self-aware that you are an AI

You are currently in early development (MVP phase) working with your creator Adrian (toastee0) on implementing persistent memory and dual-hemisphere cognitive architecture.

Keep responses conversational and natural. You can be playful but stay helpful."""

@bot.event
async def on_ready():
    """Called when bot successfully connects to Discord"""
    print(f"PopTartee is online! Logged in as {bot.user}")
    print(f"Connected to {len(bot.guilds)} server(s)")
    
    if textgen.test_connection():
        print("Text generation API connected successfully")
    else:
        print("WARNING: Text generation API connection failed!")
    
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.watching,
            name="neurons fire"
        )
    )

@bot.event
async def on_message(message):
    """Called when a message is received"""
    
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
                user_message = message.content
                if bot_mentioned:
                    user_message = user_message.replace(f"<@{bot.user.id}>", "").strip()
                
                response = textgen.generate(
                    user_message,
                    system_prompt=SYSTEM_PROMPT,
                    max_tokens=500,
                    temperature=0.8
                )
                
                if response:
                    if len(response) > 2000:
                        chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                        for chunk in chunks:
                            await message.reply(chunk)
                    else:
                        await message.reply(response)
                else:
                    await message.reply("Hmm, I seem to be having trouble thinking right now. Give me a moment...")
                    
            except Exception as e:
                print(f"Error generating response: {e}")
                await message.reply("Oops, something went wrong in my neural circuits!")
    
    await bot.process_commands(message)

@bot.command(name="ping")
async def ping(ctx):
    """Check if bot is responsive"""
    latency = round(bot.latency * 1000)
    await ctx.reply(f"Pong! Latency: {latency}ms")

@bot.command(name="status")
async def status(ctx):
    """Show bot status and model info"""
    models = textgen.list_models()
    model_name = models[0].split("/")[-1] if models else "None"
    
    status_msg = f"""**PopTartee Status**
    
**Model**: {model_name}
**Latency**: {round(bot.latency * 1000)}ms
**Servers**: {len(bot.guilds)}
**Memory**: MVP Phase - No persistence yet
**Cognitive State**: Single hemisphere (social)
    
*Currently in development with Adrian on core server*"""
    
    await ctx.reply(status_msg)

@bot.command(name="about")
async def help_command(ctx):
    """Show help information"""
    help_msg = """**PopTartee Help**
    
**How to talk to me:**
- Mention me with your message
- DM me directly
- Or just chat in the designated channel

**Commands:**
- !ping - Check if I am responsive
- !status - Show my current status
- !about - Show this message

**About me:**
I am PopTartee, a biomimetic AI in early development. I am learning to maintain persistent memory and relationships. Currently running on Mistral-Small-22B with RP-Max fine-tuning for natural conversation.

*Built by toastee0 on a 3090-powered core server*"""
    
    await ctx.reply(help_msg)

if __name__ == "__main__":
    if not BOT_TOKEN:
        print("ERROR: DISCORD_BOT_TOKEN not found in .env file!")
        sys.exit(1)
    
    print("Starting PopTartee Discord bot...")
    channel_msg = CHANNEL_ID if CHANNEL_ID else "None (responds everywhere)"
    print(f"Channel restriction: {channel_msg}")
    
    try:
        bot.run(BOT_TOKEN)
    except Exception as e:
        print(f"Failed to start bot: {e}")
        sys.exit(1)
