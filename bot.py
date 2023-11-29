import configparser
import logging

from paddleocr import PaddleOCR
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackContext, MessageHandler, filters

from utils.ocr_client import process_image_with_tesseract
from utils.telegram_client import CustomTelegramClient

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

GENDER = 0

def get_config():
    config = configparser.ConfigParser()
    config.read("config/config.ini")

    # Setting configuration values
    api_id = int(config['Telegram']['api_id'])
    api_hash = str(config['Telegram']['api_hash'])
    # phone = config['Telegram']['phone']
    # username = config['Telegram']['username']
    return api_id, api_hash


async def photo(update: Update, context: CallbackContext):

    file = (await context.bot.get_file(update.message.photo[-1].file_id)).file_path # .file_path -> retrieve the file path

    ocr = PaddleOCR(use_angle_cls=True, lang='uk')
    result = ocr.ocr(file, cls=True)
    txts = [line[1][0] for line in result[0]]
    response = 'I processed that and the result was %s' % (txts,)

    await context.bot.send_message(chat_id=update.message.chat_id, text=response)


async def link(update: Update, context: CallbackContext):
    print(f"Update = {update}")
    if update.message.text:
        link = update.message.text.split()[-1]
        print(f"Retrieved link: {link}")
        if link.startswith('http'):
            api_id, api_hash = get_config()

            client = CustomTelegramClient(api_id=api_id, api_hash=api_hash, username=update.message.from_user.username)
            await client.connect()
            # count_reply = "Retrieved {} images from {}"
            # await context.bot.send_message(
            #     chat_id=update.message.chat_id,
            #     text=count_reply.format(await client.get_image_count(link), link)
            # )
            image_data_reply = "Retrieved text from {} images by link {}"
            # await client.load_images(link, context.bot, update.message.chat_id)
            await context.bot.send_message(
                chat_id=update.message.chat_id,
                text=image_data_reply.format(await client.load_images_text_to_dataframe(link, context.bot, update.message.chat_id), link)
            )
            client.disconnect()
        else:
            await context.bot.send_message(chat_id=update.message.chat_id, text="Wrong link. Please try again")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="Надішліть будь-ласка посилання на чат для пошуку тексту")


def main():
    config = configparser.ConfigParser()
    config.read("config/config.ini")

    BOT_TOKEN = config["Telegram_Bot"]["bot_token"]
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler(['start', 'help'], start))
    # application.add_handler(MessageHandler(filters.PHOTO, photo))
    application.add_handler(MessageHandler(filters.Entity("url"), link))

    application.run_polling()


if __name__ == '__main__':
    main()
