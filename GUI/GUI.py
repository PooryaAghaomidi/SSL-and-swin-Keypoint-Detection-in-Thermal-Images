import numpy as np
import PIL.Image
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib import patches
from tkinter import filedialog, ttk
from tkinter import *
from PIL import Image, ImageTk, ImageOps 
from tkinter.messagebox import showerror, showinfo
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

    
image_uploaded = False

def upload_image():
    global file_path, image_uploaded
    file_path = filedialog.askopenfilename()
    if file_path:
        image_uploaded = True  # Set the flag to True upon successful image upload
        display_image(file_path)
    else:
        showerror(title="Error", message='No image uploaded!')

def display_image(file_path):
    global myimage
    myimage = Image.open(file_path)
    myimage.thumbnail((screen_width*0.6, screen_height*0.8))

    img = ImageTk.PhotoImage(myimage)
    display_space.configure(image=img)
    display_space.image = img
    
    myimage = ImageOps.grayscale(myimage)
    
def start_generating(myimage): 
    showinfo("Message",  "Please wait! The model is generating the desired output ...")
    
    global image_uploaded
    if not image_uploaded:
        showerror(title="Error", message='Please upload an image!')
    elif var_kp.get() == 0 and var_fd.get() == 0:
        showerror(title="Error", message='Please select one of the options!')
    elif var_kp.get() == 0 and var_fd.get() == 1:
        try:
            face_model = load_model('models/checkpoint_face', compile=True)
        except:
            showerror(title="Error", message='Tensorflow version is wrong!') 

        img  = np.zeros((1, 384, 384, 1)).astype('double')
        mywidth = 384
        wpercent = (mywidth/float(myimage.size[1]))
        hsize = int((float(myimage.size[0])*float(wpercent)))
        myimage = myimage.resize((hsize, mywidth), PIL.Image.LANCZOS)
        
        my_d = int((hsize - mywidth)/2)
        
        train_img = image.img_to_array(myimage)
        train_img = train_img[:, my_d:my_d+mywidth]
        train_img /= 255.0
        img[0,:,:,:] = train_img
        img = np.array(img)
        
        lbl_c = face_model.predict(img)
        
        np.savetxt("result.csv", lbl_c, delimiter=",")
        display_face_result(img[0], lbl_c)
        
    elif var_kp.get() == 1 and var_fd.get() == 0:
        try:
            kp_model = load_model('models/kp_checkpoint', compile=True)
        except:
            showerror(title="Error", message='Tensorflow version is wrong!')
        
        img  = np.zeros((1, 384, 384, 1)).astype('double')
        mywidth = 384
        wpercent = (mywidth/float(myimage.size[1]))
        hsize = int((float(myimage.size[0])*float(wpercent)))
        myimage = myimage.resize((hsize, mywidth), PIL.Image.LANCZOS)
        
        my_d = int((hsize - mywidth)/2)
        
        train_img = image.img_to_array(myimage)
        train_img = train_img[:, my_d:my_d+mywidth]
        train_img /= 255.0
        img[0,:,:,:] = train_img
        img = np.array(img)
        
        lbl_c = kp_model.predict(img)        
        np.savetxt("result.csv", lbl_c, delimiter=",")        
        display_kp_result(img[0], lbl_c)
        
    elif var_kp.get() == 1 and var_fd.get() == 1:
        try:
            kp_model = load_model('models/kp_checkpoint', compile=True)
            face_model = load_model('models/checkpoint_face', compile=True)
        except:
            showerror(title="Error", message='Tensorflow version is wrong!')
        
        img  = np.zeros((1, 384, 384, 1)).astype('double')
        mywidth = 384
        wpercent = (mywidth/float(myimage.size[1]))
        hsize = int((float(myimage.size[0])*float(wpercent)))
        myimage = myimage.resize((hsize, mywidth), PIL.Image.LANCZOS)
        
        my_d = int((hsize - mywidth)/2)
        
        train_img = image.img_to_array(myimage)
        train_img = train_img[:, my_d:my_d+mywidth]
        train_img /= 255.0
        img[0,:,:,:] = train_img
        img = np.array(img)
        
        lbl_kp = kp_model.predict(img)
        lbl_fd = face_model.predict(img)
        
        np.savetxt("face_result.csv", lbl_fd, delimiter=",")
        np.savetxt("keypoint_result.csv", lbl_kp, delimiter=",")
        
        display_both_result(img[0], lbl_fd, lbl_kp)
        
    showinfo("Message",  "The output has been saved as 'result.png' and 'result.csv'")
    showinfo("Message",  "You can close the program or upload another image")

def display_both_result(image, box, lbl_kp):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap='gray')
    x = lbl_kp[0, 0:68]
    y = lbl_kp[0, 68: ]
    
    rect = patches.Rectangle((box[0][1], box[0][2]), (box[0][0] - box[0][1]), (box[0][3] - box[0][2]), linewidth=4, edgecolor='r', facecolor='none')
    
    ax.add_patch(rect)
    ax.imshow(image, cmap='gray')
    ax.scatter(y, x)
    plt.savefig('result.png')

    image = Image.open('result.png')
    image.thumbnail((screen_width*0.6, screen_height*0.8))
    img = ImageTk.PhotoImage(image)
    display_space.configure(image=img)
    display_space.image = img
    
def display_kp_result(image, lbl_c):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap='gray')
    x = lbl_c[0, 0:68]
    y = lbl_c[0, 68: ]

    ax.imshow(image, cmap='gray')
    ax.scatter(y, x)
    plt.savefig('result.png')

    image = Image.open('result.png')
    image.thumbnail((screen_width*0.6, screen_height*0.8))
    img = ImageTk.PhotoImage(image)
    display_space.configure(image=img)
    display_space.image = img
    
def display_face_result(image, box):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image, cmap='gray')
    rect = patches.Rectangle((box[0][1], box[0][2]), (box[0][0] - box[0][1]), (box[0][3] - box[0][2]), linewidth=4, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.savefig('result.png')

    image = Image.open('result.png')
    image.thumbnail((screen_width*0.6, screen_height*0.8))
    img = ImageTk.PhotoImage(image)
    display_space.configure(image=img)
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
        bg_frame = bg_frame.resize((screen_width, screen_height), Image.LANCZOS)  # Resize the image to fit the screen

        photo = ImageTk.PhotoImage(bg_frame)
        canvas = tk.Canvas(root, width=screen_width, height=screen_height)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image = photo  # Keep a reference to the image to prevent garbage collection
        canvas.place(x=0, y=0)
        
        lgn_button = Image.open('upload.png')
        photo      = ImageTk.PhotoImage(lgn_button)
        lgn_button_label = Label(root, image=photo, bg='#242b3c')
        lgn_button_label.image = photo
        lgn_button_label.place(relx=.1, rely=.35)
        login = Button(lgn_button_label, text='Upload', width=10, bd=0, bg='#39bee1', cursor='hand2', fg='black', command=upload_image)
        login.config(font=('Arial', 12))
        login.place(relx=.5, rely=.5, anchor= CENTER)
        
        global var_fd
        var_fd = IntVar()
        chkbox = Checkbutton(root, text="Extract face", variable=var_fd, bg="#242b3c", font=('Arial', 12), fg='white', selectcolor="#22345f")
        chkbox.place(relx=.12, rely=.44)

        global var_kp
        var_kp = IntVar()
        chkbox = Checkbutton(root, text="Extract keypoints", variable=var_kp, bg="#242b3c", font=('Arial', 12), fg='white', selectcolor="#22345f")
        chkbox.place(relx=.12, rely=.48)

        # Start Generating Button
        lgn_button = Image.open('start.png')
        photo      = ImageTk.PhotoImage(lgn_button)
        lgn_button_label = Label(root, image=photo, bg='#242b3c')
        lgn_button_label.image = photo
        lgn_button_label.place(relx=.1, rely=.6)
        login = Button(lgn_button_label, text='Start', width=10, bd=0, bg='#d4d756', cursor='hand2', fg='black', command=lambda : start_generating(myimage))
        login.config(font=('Arial', 12))
        login.place(relx=.5, rely=.5, anchor= CENTER)

        # Space for Displaying Image / Output
        # This is a placeholder; you'll update this space with the image/output
        global display_space
        display_space = tk.Label(root, relief=tk.SUNKEN, bg='#4e525b')
        display_space.place(relx=.36, rely=.08, width=screen_width*0.6, height=screen_height*0.8)

    create_layout()
    root.mainloop()

# Run the GUI
create_gui()