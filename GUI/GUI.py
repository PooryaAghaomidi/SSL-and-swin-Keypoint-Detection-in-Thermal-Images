import PIL.Image
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import *
from PIL import Image, ImageTk
from tkinter.messagebox import showerror
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

face_model = load_model('checkpoint_face', compile = True)
image_uploaded = False


def upload_image():
    global file_path, image_uploaded
    file_path = filedialog.askopenfilename()
    if file_path:
        image_uploaded = True  # Set the flag to True upon successful image upload
        display_image(file_path)
    else:
        showerror(title = "Error", message = 'No image uploaded!')


def start_generating():
    global image_uploaded
    if not image_uploaded:
        showerror(title = "Error", message = 'Please upload an image!')
    elif var_kp.get() == 0 and var_fd.get() == 0:
        showerror(title = "Error", message = 'Please select one of the options!')
    elif var_kp.get() == 0 and var_fd.get() == 1:
        train_img = image.img_to_array(myimage)
        train_img = train_img[:, 64:448]
        train_img /= 255.0
        img[0, :, :, :] = train_img
        img = np.array(img)

        lbl_c = classifier.predict(img)


def display_image(file_path):
    global myimage
    myimage = Image.open(file_path)
    myimage.thumbnail((screen_width * 0.6, screen_height * 0.8))

    img = ImageTk.PhotoImage(myimage)
    display_space.configure(image = img)
    display_space.image = img


def create_gui():
    # Create a Tkinter window
    root = tk.Tk()
    root.title("Face and Keypoint Detection")

    root.geometry('1166x718')
    root.resizable(0, 0)
    root.state('zoomed')

    # Function to create the layout
    def create_layout():
        global screen_width, screen_height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        bg_frame = Image.open('background.jpg')
        bg_frame = bg_frame.resize((screen_width, screen_height), Image.ANTIALIAS)  # Resize the image to fit the screen

        photo = ImageTk.PhotoImage(bg_frame)
        canvas = tk.Canvas(root, width = screen_width, height = screen_height)
        canvas.create_image(0, 0, anchor = tk.NW, image = photo)
        canvas.image = photo  # Keep a reference to the image to prevent garbage collection
        canvas.place(x = 0, y = 0)

        lgn_button = Image.open('upload.png')
        photo = ImageTk.PhotoImage(lgn_button)
        lgn_button_label = Label(root, image = photo, bg = '#242b3c')
        lgn_button_label.image = photo
        lgn_button_label.place(relx = .1, rely = .35)
        login = Button(lgn_button_label, text = 'Upload', width = 10, bd = 0, bg = '#39bee1', cursor = 'hand2', fg = 'black', command = upload_image)
        login.config(font = ('Arial', 12))
        login.place(relx = .5, rely = .5, anchor = CENTER)

        global var_fd
        var_fd = IntVar()
        chkbox = Checkbutton(root, text = "Extract face", variable = var_fd, bg = "#242b3c", font = (
            'Arial', 12), fg = 'white', selectcolor = "#22345f")
        chkbox.place(relx = .12, rely = .44)

        global var_kp
        var_kp = IntVar()
        chkbox = Checkbutton(root, text = "Extract keypoints", variable = var_kp, bg = "#242b3c", font = (
            'Arial', 12), fg = 'white', selectcolor = "#22345f")
        chkbox.place(relx = .12, rely = .48)

        # Start Generating Button
        lgn_button = Image.open('start.png')
        photo = ImageTk.PhotoImage(lgn_button)
        lgn_button_label = Label(root, image = photo, bg = '#242b3c')
        lgn_button_label.image = photo
        lgn_button_label.place(relx = .1, rely = .6)
        login = Button(lgn_button_label, text = 'Start', width = 10, bd = 0, bg = '#d4d756', cursor = 'hand2', fg = 'black', command = start_generating)
        login.config(font = ('Arial', 12))
        login.place(relx = .5, rely = .5, anchor = CENTER)

        # Space for Displaying Image / Output
        # This is a placeholder; you'll update this space with the image/output
        global display_space
        display_space = tk.Label(root, relief = tk.SUNKEN, bg = '#4e525b')
        display_space.place(relx = .36, rely = .08, width = screen_width * 0.6, height = screen_height * 0.8)

    create_layout()
    root.mainloop()


# Run the GUI
create_gui()