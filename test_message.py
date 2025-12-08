#!/usr/bin/env python3
"""Quick test to see if bot receives messages"""
import os
import discord
from dotenv import load_dotenv

load_dotenv("/home/toastee/BioMimeticAi/config/.env")

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    print(f"Intents: {client.intents}")
    print(f"Message content enabled: {client.intents.message_content}")

@client.event
async def on_message(message):
    print(f"[MESSAGE] {message.author}: {message.content} (Channel: {message.channel.id})")

client.run(os.getenv("DISCORD_BOT_TOKEN"))
