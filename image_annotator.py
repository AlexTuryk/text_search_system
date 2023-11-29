import tkinter as tk
import pandas as pd
from PIL import Image, ImageTk
import os


class ImageAnnotator:
    def __init__(self, image_folder, dataset_path):
        self.image_folder = image_folder
        self.dataset_path = dataset_path
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        self.df = pd.DataFrame({
            'Image_Path': [os.path.join(image_folder, img) for img in self.image_files],
            'Label': [''] * len(self.image_files)
        })

        self.current_index = 0

        self.root = tk.Tk()
        self.root.title("Image Annotator")

        self.label_var = tk.StringVar()
        self.label_text = tk.Text(self.root, height=5, width=60)
        self.label_text.pack(pady=10)

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.annotation_count_label = tk.Label(self.root, text="Annotations: 0")
        self.annotation_count_label.pack()

        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack(pady=10)

        self.save_button = tk.Button(self.root, text="Save", command=self.save_dataset)
        self.save_button.pack(pady=10)

        self.load_image()
        self.root.mainloop()

    def load_image(self):
        if self.current_index < len(self.df):
            image_path = self.df.loc[self.current_index, 'Image_Path']
            image = Image.open(image_path)
            width, height = image.size
            print(width, height)
            if width > 900 and height > 900:
                image = image.resize((width//3, height//3))
            else:
                image = image.resize((width//2, height//2))  # Resize the image as needed
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.label_text.delete('1.0', tk.END)  # Clear previous text
            self.label_text.insert(tk.END, self.df.loc[self.current_index, 'Label'])
            self.annotation_count_label.config(text=f"Annotations: {self.current_index}/{len(self.df)}")
        else:
            self.label_text.delete('1.0', tk.END)
            self.label_text.insert(tk.END, "Annotation complete")
            self.annotation_count_label.config(text=f"Annotations: {len(self.df)}/{len(self.df)}")

    def next_image(self):
        label = self.label_text.get('1.0', tk.END).strip()
        self.df.loc[self.current_index, 'Label'] = label
        self.current_index += 1
        self.load_image()

    def save_dataset(self):
        self.df.to_csv(self.dataset_path, encoding='utf-8-sig', index=False)
        self.root.destroy()


if __name__ == "__main__":
    image_folder_path = "D:\Магістерська робота\Project\datasets\memes_from_twitter\colored_background"
    dataset_file_path = "annotation.csv"

    annotator = ImageAnnotator(image_folder_path, dataset_file_path)
