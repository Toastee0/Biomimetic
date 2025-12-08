#!/usr/bin/env python3
"""Test reading messages from Discord channel via API"""

import os
import discord
from dotenv import load_dotenv
import asyncio

load_dotenv("/home/toastee/BioMimeticAi/config/.env")

BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID"))

async def test_channel_read():
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True
    
    client = discord.Client(intents=intents)
    
    @client.event
    async def on_ready():
        print(f"Logged in as {client.user}")
        print(f"Guilds: {len(client.guilds)}")
        
        # Try to get the channel
        channel = client.get_channel(CHANNEL_ID)
        
        if channel:
            print(f"\nFound channel: {channel.name} (ID: {channel.id})")
            print(f"Channel type: {type(channel)}")
            print(f"Can read messages: {channel.permissions_for(channel.guild.me).read_message_history}")
            print(f"Can send messages: {channel.permissions_for(channel.guild.me).send_messages}")
            
            # Try to fetch recent messages
            print(f"\nFetching last 10 messages...")
            try:
                messages = []
                async for msg in channel.history(limit=10):
                    messages.append(msg)
                
                print(f"Found {len(messages)} messages:")
                for msg in reversed(messages):
                    print(f"  [{msg.created_at}] {msg.author.name}: {msg.content[:50]}")
                    
            except Exception as e:
                print(f"Error fetching messages: {e}")
        else:
            print(f"Channel {CHANNEL_ID} not found!")
            print("\nAvailable channels:")
            for guild in client.guilds:
                print(f"  Guild: {guild.name}")
                for ch in guild.text_channels:
                    print(f"    - {ch.name} (ID: {ch.id})")
        
        await client.close()
    
    await client.start(BOT_TOKEN)

if __name__ == "__main__":
    asyncio.run(test_channel_read())
