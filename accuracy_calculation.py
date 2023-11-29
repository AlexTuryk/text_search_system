import os.path
import random
import string
import time

import cv2
import numpy as np
import pandas as pd
import pytesseract
import seaborn as sns
from PIL import Image, ImageFont, ImageDraw

from matplotlib import pyplot as plt, image as mpimg
from pytesseract import Output

from utils.image_preprocessing import get_grayscale, thresholding, opening, canny, median_blur
from utils.ocr_client import process_image_with_paddle, process_image_with_tesseract, get_accuracy_metrics


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def visualize_random_images(image_folder, num_images=9):
    # Зчитати шляхи до усіх зображень у папці
    all_images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]

    # Обрати випадкові 9 шляхів
    selected_images = random.sample(all_images, min(num_images, len(all_images)))

    # Створити сітку розміром 3x3 для візуалізації
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    plt.subplots_adjust(wspace=0, hspace=0)

    for i, ax in enumerate(axes.flat):
        if i < len(selected_images):
            # Зчитати та відобразити зображення
            img = mpimg.imread(selected_images[i])
            ax.imshow(img)
            ax.axis('off')  # Вимкнути відображення координатних вісей

    plt.show()


def plot_words_count_distribution(excel_path, title):
    # Read Excel file into a pandas DataFrame
    df = pd.read_excel(excel_path)

    # Check if the 'Words count' column exists in the DataFrame
    if 'Words count' not in df.columns:
        raise ValueError("The 'Words count' column does not exist in the DataFrame.")
    # df = df.drop(df.index[-1])
    # Calculate the count of values in the 'Words count' column
    words_count_distribution = df['Words count'].value_counts()

    # Plot the distribution
    plt.figure(figsize=(12, 8))
    sns.barplot(x=words_count_distribution.index, y=words_count_distribution.values, color='blue')
    plt.title(title)
    plt.xlabel('Words Count')
    plt.ylabel('Count')
    plt.show()


def plot_time_words_dependency(excel_path, y_column, title):
    df = pd.read_excel(excel_path)
    df.sort_values(by='Time')
    plt.figure(figsize=(10, 6))
    sns.set_style("darkgrid")
    sns.scatterplot(x='Time', y=y_column, data=df)
    plt.title(title)
    plt.ylabel('Кількість слів')
    plt.xlabel('Час розпізнавання')
    plt.show()


def plot_grouped_data(excel_path, word_count, correct_word_count, char_count, correct_char_count):
    df = pd.read_excel(excel_path, index_col=False)
    df.rename(
        columns={
            word_count: 'Усі слова', correct_word_count: 'Правильні слова',
            char_count: 'Усі символи', correct_char_count: 'Правильні символи'},
        inplace=True
    )

    word_count, correct_word_count, char_count, correct_char_count = 'Усі слова', 'Правильні слова', 'Усі символи', 'Правильні символи'
    word_df, char_df = df.copy(), df.copy()

    word_df['Категорія'] = pd.cut(df[word_count], bins=range(0, max(df[word_count]) + 5, 5), right=False)
    char_df['Категорія'] = pd.cut(df[char_count], bins=range(0, max(df[char_count]) + 50, 50), right=False)
    # Групування та обчислення сум
    grouped_word_df = word_df.groupby('Категорія').agg({correct_word_count: 'sum', word_count: 'sum'}).reset_index()
    grouped_char_df = char_df.groupby('Категорія').agg({correct_char_count: 'sum', char_count: 'sum'}).reset_index()
    # Обчислення відношення Correct Words до Words count
    grouped_word_df['Ratio'] = grouped_word_df[correct_word_count] / grouped_word_df[word_count]
    grouped_char_df['Ratio'] = grouped_char_df[correct_char_count] / grouped_char_df[char_count]

    grouped_word_df.dropna(inplace=True)
    grouped_char_df.dropna(inplace=True)
    # Виведення результату
    colors = {correct_word_count: 'green', word_count: 'cyan'}
    char_colors = {correct_char_count: 'green', char_count: 'blue'}
    # Створення стовпчикової діаграми
    plt.figure(figsize=(12, 24))
    plt.subplot(2, 1, 1)

    df_melted = pd.melt(grouped_word_df, id_vars=['Категорія'], value_vars=[correct_word_count, word_count],
                        var_name='Категорія слів', value_name='Кількість')
    ax1 = sns.barplot(x='Категорія', y='Кількість', hue='Категорія слів', data=df_melted, palette=colors, alpha=0.6)
    for p in ax1.patches:
        if p.get_height() == 0:
            continue
        ax1.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)
    plt.title('Розподіл кількості усіх слів та правильно розпізнаних')

    plt.subplot(2, 1, 2)
    char_df_melted = pd.melt(grouped_char_df, id_vars=['Категорія'], value_vars=[correct_char_count, char_count],
                        var_name='Категорія символів', value_name='Кількість')
    ax2 = sns.barplot(x='Категорія', y='Кількість', hue='Категорія символів', data=char_df_melted, palette=char_colors, alpha=0.6)

    for p in ax2.patches:
        if p.get_height() == 0:
            continue
        ax2.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)
    plt.title('Розподіл кількості усіх символів та правильно розпізнаних')
    plt.subplots_adjust(bottom=0.1, right=0.95, top=0.95, hspace=0.3)
    plt.show()


def preprocess_string(s: str) -> str:
    s = s.lower()
    result = s.translate(str.maketrans('', '', string.punctuation))
    return result


def visualize_images(images, captions):
    """
    Візуалізація 4 зображень у 2 рядки.

    Параметри:
    - images: Список шляхів до зображень або самі зображення.
    - captions: Список підписів для кожного зображення.
    """

    if len(images) != 4 or len(captions) != 4:
        raise ValueError("Потрібно передати рівно 4 зображення та 4 підписи.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            img = images[idx]

            # Перевірка, чи передано шляхи до зображень чи самі зображення
            if isinstance(img, str):
                img = mpimg.imread(img)

            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(captions[idx])

    plt.show()


def preprocess_image(image):
    gray = get_grayscale(image)
    thresh = thresholding(gray)
    opening_image = opening(gray)
    blur = median_blur(gray)
    # visualize_images([gray, thresh, opening_image, blur],
    #                  ['Сірі тони', 'Бінаризація Оцу', 'Морфологічна обробка', 'Медіанний фільтр'])
    return blur


def get_accuracy(annotations_path: str, model: str, result_df_name: str):
    """Performs text recognition and comparing result with annotation

    :param images_path:         Path to images
    :param annotations_path:    Path to .csv annotations
    :param model:               One from "paddle" "tesseract"
    :return:                    Int value of accuracy
    """
    print(f"Model: {model} result: {result_df_name}")
    df = pd.read_csv(annotations_path)
    results = []
    for index, row in df.iterrows():
        image = np.asarray(Image.open(row["Image_Path"]).convert('RGB'))
        start_time = time.time()
        image = preprocess_image(image)

        if model == "paddle":
            ocr_words_list = process_image_with_paddle(image, return_only_text=True)
        elif model == "tesseract":
            ocr_words_list = process_image_with_tesseract(image)

        # print(ocr_words_list)
        recognition_time = time.time() - start_time
        ocr_string = ' '.join(ocr_words_list)
        ground_truth, recognized = preprocess_string(row['Label']), preprocess_string(ocr_string)
        wer, cer = get_accuracy_metrics(ground_truth, recognized)
        results.append([row["Image_Path"], ground_truth, recognized, wer, cer, recognition_time])

    result_df = pd.DataFrame(
        results, columns=['Image_Path', 'Ground_Truth', 'Recognized', 'WER', 'CER', 'Time']
    )
    result_df['Chars count'] = df['Label'].apply(lambda words: sum(len(word) for word in words.split()))
    result_df['Words count'] = df['Label'].apply(lambda words: len(words.split()))
    result_df['Correct chars'] = result_df['Chars count'] - round(result_df['Chars count'] * result_df['CER'])
    result_df['Correct words'] = result_df['Words count'] - round(result_df['Words count'] * result_df['WER'])

    result_df.to_excel(model+'_'+result_df_name)
    correct_words, all_words = result_df['Correct words'].sum(), result_df['Words count'].sum()
    correct_chars, all_chars = result_df['Correct chars'].sum(), result_df['Chars count'].sum()

    print(f"Recognized {correct_words} from {all_words} words. WER = {correct_words/all_words}")
    print(f"Recognized {correct_chars} from {all_chars} characters. CER = {correct_chars/all_chars}\n")


def visualise_ocr_results(excel_path, ocr="pytesseract", show_worst=False):
    df = pd.read_excel(excel_path, index_col=False)
    df.sort_values(by="Correct words", ascending=show_worst, inplace=True)
    print(df)
    count = 0
    for index, row in df.iterrows():
        if count == 10:
            break
        image = np.asarray(Image.open(row["Image_Path"]).convert('RGB'))
        if ocr == "paddle":
            print("Preprocessing image...")
            process_image_with_paddle(image)
        else:
            result = pytesseract.image_to_data(image, output_type=Output.DICT, lang="ukr+eng")
            font = ImageFont.truetype("C:\\Windows\\Fonts\\consolaz.ttf", 25)
            # font = ImageFont.truetype("gulim.ttc", 20)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            print(result["text"])
            image = Image.open(row["Image_Path"])
            n_boxes = len(result['text'])
            for i in range(n_boxes):
                if int(result['conf'][i]) > 60:
                    (x, y, w, h) = (result['left'][i], result['top'][i], result['width'][i], result['height'][i])
                    draw = ImageDraw.Draw(image)
                    draw.text((x, y - 20), result["text"][i], font=font, fill=(255, 0, 0, 0))
                    draw.rectangle([x, y, x+w, y+h], outline=(0, 255, 0), width=2)

            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2_im_processed = cv2.resize(cv2_image, None, fx=0.5, fy=0.5)
            cv2.imshow('img', cv2_im_processed)
            cv2.waitKey(0)
        count+=1



base_path = "D:\Магістерська робота\Project\datasets\memes_from_twitter"

white_annotations = os.path.join(base_path, "white_background\\annotation.csv")
white_result = "white_background_result.xlsx"

colored_annotations = os.path.join(base_path, "colored_background\\annotation.csv")
colored_result = "colored_background_result.xlsx"

# get_accuracy(white_annotations, "tesseract", white_result)
# get_accuracy(colored_annotations, "tesseract", colored_result)
#
# get_accuracy(white_annotations, "paddle", white_result)
# get_accuracy(colored_annotations, "paddle", colored_result)


# visualize_images(os.path.join(base_path, "colored_background"))
result_white_path = os.path.join(base_path, "white_background\\tesseract_white_background_result.xlsx")
result_colored_path = os.path.join(base_path, "colored_background\\tesseract_colored_background_result.xlsx")

# plot_time_words_dependency(
#     excel_path=result_white_path,
#     y_column='WER',
#     title='Залежність часу розпізнавання відносно кількості слів'
# )
result_colored_path = r"D:\Магістерська робота\Project\results\lowercaase_&_without_punctutaion\paddle_white_background_result.xlsx"

visualise_ocr_results(result_colored_path, ocr="paddle", show_worst=True)
# plot_grouped_data(
#     result_colored_path, "Words count", "Correct words",
#     "Chars count", "Correct chars"
# )
# plot_words_count_distribution(result_white_path, "White dataset distribution")
# plot_words_count_distribution(result_colored_path, "Colored dataset distribution")
# print(get_accuracy_metrics("Один в полі не воїн", "дин палі не воїн і", visualise_alignment=True))

