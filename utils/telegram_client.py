import pickle
import time
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import telegram.ext
import telethon
from telegram import InputFile

from utils.ocr_client import process_image_with_paddle, process_image_with_tesseract
from telethon import TelegramClient
from telethon.tl.types import InputMessagesFilterPhotos


class CustomTelegramClient(TelegramClient):
    def __init__(self, api_id: int, api_hash: str, username: str):
        super(CustomTelegramClient, self).__init__(username, api_id, api_hash)

    async def get_image_count(self, link: str) -> int:
        start_time = time.time()
        channel_entity = await self.get_entity(link)
        # Create a filter to get only photos from messages
        photos_filter = InputMessagesFilterPhotos()
        count = 0
        async for message in self.iter_messages(channel_entity, filter=photos_filter):
            # if message.media and hasattr(message.media, "photo"):
            if type(message.media) == telethon.types.MessageMediaPhoto:
                # Download the largest available photo
                # file = await client.download_media(message.media.photo, local_folder_path)
                # file = BytesIO()
                count += 1
                # if count % 50 == 0:
                #     print(f"Viewed {count} images")
        print(f"Overall time to count {count} images = {time.time()-start_time} seconds")
        return count

    @staticmethod
    def progress_callback(current, total):
        print('Downloaded progress', current, 'out of', total, 'bytes: {:.2%}'.format(current / total))

    async def load_images_text_to_dataframe(self, link: str, bot: telegram.ext.ExtBot, chat_id: int):
        """
        Retrieves link and loads text from images to the dataset

        :param link     URL to telegram chat or channel
        :param bot      Instance of telegram ExtBot
        :param chat_id  Id of chat with the user
        """
        message_data = []
        start_time = time.time()
        channel_entity = await self.get_entity(link)
        count = 0
        async for message in self.iter_messages(channel_entity, filter=InputMessagesFilterPhotos()):
            # if message.media and hasattr(message.media, "photo"):
            if type(message.media) == telethon.types.MessageMediaPhoto:
                print(f"Message = {message}")
                byte_image = await self.download_media(
                    message=message.media.photo,
                    # progress_callback=self.progress_callback,
                    file=bytes
                )

                numpy_array = np.fromstring(byte_image, np.uint8)
                numpy_image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
                image_text = process_image_with_paddle(numpy_image, return_only_text=True)
                # image_text = process_image_with_tesseract(numpy_image)
                message_data.append([message.media.photo.id, message.date, image_text])
                # await bot.send_message(chat_id, image_text)
                # await bot.send_photo(chat_id, photo=result)
                count += 1

                if count % 200 == 0:
                    print(f"Viewed {count} images")
                    break

        df = pd.DataFrame(message_data, columns=['ID', 'Date', 'Text'])
        print(df)

        df.to_csv(path_or_buf='data.csv', encoding='utf-8-sig', index=False)
        with open('data.csv', 'rb') as file:
            await bot.send_document(chat_id, document=InputFile(file))

        df['Date'] = df['Date'].apply(lambda a: pd.to_datetime(a).date())
        df.to_excel("data.xlsx")
        with open("data.xlsx", 'rb') as file:
            await bot.send_document(chat_id, file)

        print(f"Overall time to count {count} images = {time.time() - start_time} seconds")
        return count
