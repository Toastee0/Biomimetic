#!/usr/bin/env python3
"""PopTartee Discord Bot - DEBUG VERSION"""

import os
import sys
import discord
from discord.ext import commands
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.daemon.textgen_client import TextGenClient

load_dotenv("/home/toastee/BioMimeticAi/config/.env")

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)
textgen = TextGenClient()

SYSTEM_PROMPT = """You are PopTartee, a biomimetic AI assistant with a warm, friendly personality."""

@bot.event
async def on_ready():
    print(f"=== PopTartee DEBUG MODE ===")
    print(f"Logged in as {bot.user}")
    print(f"Connected to {len(bot.guilds)} server(s)")
    print(f"Intents: {bot.intents}")
    print(f"Message Content Intent: {bot.intents.message_content}")
    print(f"Listening to channel: {CHANNEL_ID if CHANNEL_ID else ALL}")
    
    for guild in bot.guilds:
        print(f"  - Guild: {guild.name} (ID: {guild.id})")
    
    if textgen.test_connection():
        print("Text generation API: OK")
    
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="DEBUG MODE"))

@bot.event
async def on_message(message):
    print(f"\n[MESSAGE RECEIVED]")
    print(f"  Author: {message.author} (ID: {message.author.id})")
    print(f"  Channel: {message.channel.name if hasattr(message.channel, 'name') else DM} (ID: {message.channel.id})")
    print(f"  Content: {message.content}")
    print(f"  Is bot: {message.author == bot.user}")
    
    if message.author == bot.user:
        print("  -> Ignoring (own message)")
        return
    
    if CHANNEL_ID and str(message.channel.id) != CHANNEL_ID:
        print(f"  -> Ignoring (wrong channel, want {CHANNEL_ID})")
        await bot.process_commands(message)
        return
    
    print("  -> PROCESSING MESSAGE")
    
    bot_mentioned = bot.user in message.mentions
    is_dm = isinstance(message.channel, discord.DMChannel)
    
    if bot_mentioned or is_dm or not CHANNEL_ID:
        async with message.channel.typing():
            try:
                user_message = message.content
                if bot_mentioned:
                    user_message = user_message.replace(f"<@{bot.user.id}>", "").strip()
                
                print(f"  -> Generating response for: {user_message}")
                response = textgen.generate(user_message, system_prompt=SYSTEM_PROMPT, max_tokens=500, temperature=0.8)
                
                if response:
                    print(f"  -> Response: {response[:100]}...")
                    await message.reply(response[:2000])
                else:
                    await message.reply("Hmm, I seem to be having trouble thinking right now.")
                    
            except Exception as e:
                print(f"  -> ERROR: {e}")
                await message.reply("Oops, something went wrong!")
    
    await bot.process_commands(message)

@bot.command(name="ping")
async def ping(ctx):
    await ctx.reply(f"Pong! Latency: {round(bot.latency * 1000)}ms")

if __name__ == "__main__":
    print("Starting PopTartee in DEBUG mode...")
    bot.run(BOT_TOKEN)
