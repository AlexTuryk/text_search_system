import configparser
import os

from telethon import TelegramClient
from telethon.types import MessageMediaPhoto, MessageMediaDocument
from telethon.tl import functions
from telethon.tl.types import InputMessagesFilterPhotos

# Reading Configs
config = configparser.ConfigParser()
config.read("config/config.ini")

# Setting configuration values
api_id = int(config['Telegram']['api_id'])
api_hash = str(config['Telegram']['api_hash'])
phone = config['Telegram']['phone']
username = config['Telegram']['username']

# Create the client and connect
client = TelegramClient(username, api_id, api_hash)
client.start()

local_folder_path = 'datasets/memes_from_twitter'
memes_link = 'https://t.me/+NsYoicvr5SpjOTRi'


async def download_images(channel_username, local_folder_path):
    count = 1
    await client.start()

    # Resolve the entity of the channel
    channel_entity = await client.get_entity(channel_username)

    # Create a filter to get only photos from messages
    photos_filter = InputMessagesFilterPhotos()

    async for message in client .iter_messages(channel_entity, filter=photos_filter):
        if message.media and hasattr(message.media, "photo"):
            # Download the largest available photo
            file = await client.download_media(message.media.photo, os.path.join(local_folder_path, f"image{count}"))
            count += 1
            if count % 100 == 0:
                print(f"Downloaded {count} images")


with client:
    client.loop.run_until_complete(download_images(memes_link, local_folder_path))