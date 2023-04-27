import tkinter as tk
from tkinter import *
import tkinter.font as tfont
from PIL import Image, ImageTk
import test as gr  # assuming your gender recognition code is in a file called gender_recognition.py
import analitical_graphs as ag
import time
class GenderRecognitionApp:
    def __init__(self, master):

        #fonts
        self.custom_font = tfont.Font(family="Times", size=14)
        self.custom_bg_color = "#f5f5f1"

        self.master = master
        master.title("Gender Recognition")
        master.iconbitmap("images/microphone.ico")
        

        self.menu_bar = Menu(master)
        master.config(menu=self.menu_bar)

        # create the "File" menu
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="New Session", command=self.new_session)
        self.file_menu.add_command(label="Save Session", command=self.save_session)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=master.quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        # create the "Help" menu
        self.help_menu = Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        
        
        
        
        
        self.label = tk.Label(master, text="Enter voice sample path:", font=self.custom_font, bg=self.custom_bg_color)
        self.label.pack(fill=tk.X, padx=20, pady=10)

        self.path_entry = tk.Entry(master, font=self.custom_font)
        self.path_entry.pack(fill=tk.X, padx=20, pady=10)

        self.recognize_button = tk.Button(master, text="Recognize", command=lambda:[self.recognize_gender(),self.analitical_graphs()], font=self.custom_font, bg=self.custom_bg_color)
        self.recognize_button.pack(pady=10)

        self.gender_label = tk.Label(master, text="",font=self.custom_font, bg=self.custom_bg_color)
        self.gender_label.pack(fill=tk.X, padx=20, pady=10)

        self.recognize_button = tk.Button(master, text="Graphs display", command=lambda:[ self.show_graph(master=master)], font=self.custom_font, bg=self.custom_bg_color)
        self.recognize_button.pack(pady=10)

        self.gender_label2 = tk.Label(master, text="",font=self.custom_font, bg=self.custom_bg_color)
        self.gender_label2.pack(fill=tk.X, padx=20, pady=10)

        self.clear_button = tk.Button(master, text="Clear", command=self.clear_widgets, font=self.custom_font)
        self.clear_button.pack(pady=10)
        
        self.image_frame = tk.Frame(master)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        

        self.gender_label.bind("<Configure>", self.update_font_size)



    def analitical_graphs(self):
        voice_sample_path = self.path_entry.get()
        gender = ag.main(voice_sample_path)
        self.gender_label2.configure(text=f"Graphs Ready {gender}")



    def show_graph(self,master):


        self.image_frame = tk.Frame(master)
        self.image_frame.pack(fill=tk.BOTH, expand=True)


        """self.image1 = Image.open("images/spectrum frequency.png")
        self.photo1 = ImageTk.PhotoImage(self.image1)
        self.image_label1 = Label(master, image=self.photo1)
        self.image_label1 = tk.Label(self.image_frame, image=self.photo1)
        self.image_label1.grid(row=0, column=0, padx=10, pady=10, sticky=tk.NSEW)"""

        self.image2 = Image.open('images/graphs/Signal.png')
        self.photo2 = ImageTk.PhotoImage(self.image2)
        self.image_label2 = Label(master, image=self.photo2)
        self.image_label2 = tk.Label(self.image_frame, image=self.photo2)
        self.image_label2.grid(row=0, column=0, padx=10, pady=10, sticky=tk.NSEW)

        self.image3 = Image.open('images/graphs/Spectrum.png')
        self.photo3 = ImageTk.PhotoImage(self.image3)
        self.image_label3 = Label(master, image=self.photo3)
        self.image_label3 = tk.Label(self.image_frame, image=self.photo3)
        self.image_label3.grid(row=0, column=2, padx=10, pady=10, sticky=tk.NSEW)

        # Configure the grid to resize with the window
        #self.image_frame.columnconfigure(0, weight=1, side=tk.LEFT, expand=True)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.columnconfigure(2, weight=1)
        self.image_frame.rowconfigure(0, weight=1)


    def recognize_gender(self):
        voice_sample_path = self.path_entry.get()
        gender = gr.main(voice_sample_path)
        self.gender_label.configure(text=f"The gender is {gender}")

    def clear_widgets(self):
        self.path_entry.delete(0, tk.END)
        self.gender_label.configure(text="")
        self.gender_label2.configure(text="")

    def update_font_size(self, event):
        # update the font size of the gender_label based on its width
        width = event.width
        font_size = max(int(width / 80), 10)  # adjust the divisor and minimum font size as needed
        self.custom_font.configure(size=font_size)

    def new_session(self):
        new_window = Toplevel(self.master)
        GenderRecognitionApp(new_window)

    def save_session(self):
        # implement the logic to save the current session
        pass

    def show_about(self):
        # implement the logic to show the "About" message box
        pass

if __name__ == '__main__':
    root = tk.Tk()
    width= root.winfo_screenwidth()
    height= root.winfo_screenheight()
    #setting tkinter window size
    root.geometry("%dx%d" % (width, height))
    app = GenderRecognitionApp(root)
    
    root.mainloop()