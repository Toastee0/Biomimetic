#!/usr/bin/env python3
"""Discord webhook sender - sends messages to Discord channel"""

import os
import sys
from discord_webhook import DiscordWebhook
from dotenv import load_dotenv

# Load environment variables
load_dotenv("/home/toastee/BioMimeticAi/config/.env")

class DiscordSender:
    def __init__(self):
        self.webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not self.webhook_url:
            raise ValueError("DISCORD_WEBHOOK_URL not found in environment")

    def send(self, message, username="BiomimAI-MVP"):
        """Send a message to Discord"""
        webhook = DiscordWebhook(
            url=self.webhook_url,
            content=message,
            username=username
        )
        response = webhook.execute()
        return response

    def send_test(self):
        """Send a test message"""
        return self.send("🤖 BiomimAI MVP online and ready for testing!")

if __name__ == "__main__":
    sender = DiscordSender()
    if len(sys.argv) > 1:
        # Send message from command line argument
        message = " ".join(sys.argv[1:])
        sender.send(message)
        print(f"Sent: {message}")
    else:
        # Send test message
        sender.send_test()
        print("Test message sent to Discord!")
