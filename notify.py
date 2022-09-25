import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import requests
import json


client = WebClient(token="xoxb-4113780431075-4113827224883-ddc7NvRNpTeDaxQjAXqWGw60")

def NotifySlack():
    client.chat_postMessage(channel="general", text="Training done!")
