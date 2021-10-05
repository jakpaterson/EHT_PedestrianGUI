"""
Version 2.0
GUI to Run YOLOv4 Algorithm for Pedestrian Tracking
By: Zeyad and Jakson
Last Modified: June 2nd, 2021

Things to note:
TKINTER package is used
Geometry of window is set to 640x640

Formatting of the items is done using pack
- takes the relative x and y between 0 and 1 (base off of top left corner)
- anchor param set where in the box the text displays

"""
import csv
import os
import tkinter as tk
import time
import webbrowser
from datetime import datetime, timedelta
from tkinter import HORIZONTAL, CENTER, VERTICAL, Y, RIGHT, FALSE, BOTH, END, N
import tkinter.filedialog as fd
from tkinter.ttk import Progressbar

import cv2
from PIL import ImageTk, Image

# Initialize GUI window
import matplotlib.pyplot as plt
import pandas as pd
from absl import app

import video_clip
import weather_report

window = tk.Tk()
window.title('YOLOv4 Analysis GUI')
winWidth = 640
height = 640
geometry = str(winWidth) + "x" + str(height)
window.geometry(geometry)
window.config(bg="#6D2C9E")
window.resizable(False, False)

# global vars
output_folder = ""
input_file = ""
file_frame = ""
analysis_frame = ""
videoUp = False
folderUp = False
widthFactor = 0
heightFactor = 0


def main():
    # GUI title centered
    gui_title = tk.Label(window, text='YOLOv4 Pedestrian Detection Upload', bg="#6D2C9E", fg="white")
    gui_title.config(font=('Arial', 20))
    gui_title.place(relx=0.5, rely=0.04, anchor=CENTER)

    # Create frame (640x320)
    global file_frame
    file_frame = tk.Frame(window, width=winWidth, height=280).place(relx=0.5, rely=0.3, anchor=CENTER)
    analysis_title = tk.Label(window, text='Analyze Video Data', bg="#6D2C9E", fg="white")
    analysis_title.config(font=('Arial', 18))
    analysis_title.place(relx=0.5, rely=0.56, anchor=CENTER)
    analysis_section = tk.Frame(window, width=winWidth, height=120).place(relx=0.5, rely=0.7, anchor=CENTER)

    # First row labels for selecting files
    tk.Label(file_frame, text='Upload video in mp4 format ',
             font=('Helvetica', 11, 'bold')).place(relx=0.3, rely=0.15, anchor=CENTER)
    select_output = tk.Label(file_frame, text='Select destination of output ',
                             font=('Helvetica', 11, 'bold')).place(relx=0.7, rely=0.15, anchor=CENTER)
    # Second row buttons to open directory
    tk.Button(file_frame, text='Choose File',
              command=lambda: open_file()).place(relx=0.3, rely=0.2, anchor=CENTER)
    tk.Button(file_frame, text='Choose Folder',
              command=lambda: destination_folder()).place(relx=0.7, rely=0.2, anchor=CENTER)

    # Radiobutton for different configurations

    radiobutton_tag = tk.Label(file_frame, text='What do you want to detect? ', font=('Helvetica', 11, 'bold')).place(
        relx=0.5, rely=0.35, anchor=CENTER)
    var1 = tk.IntVar()
    R1 = tk.Radiobutton(window, text="Pedestrians", variable=var1, value=1).place(relx=0.25, rely=0.4, anchor=CENTER)
    R2 = tk.Radiobutton(window, text="Assistive Devices", variable=var1, value=2).place(relx=0.45, rely=0.4,
                                                                                        anchor=CENTER)
    R3 = tk.Radiobutton(window, text="Pedestrians + Assistive Devices", variable=var1, value=3).place(relx=0.70,
                                                                                                      rely=0.4,
                                                                                                      anchor=CENTER)

    # start button at bottom of page
    tk.Button(file_frame, text='Run YOLOv4 Process',
              command=lambda: upload_file()).place(relx=0.5, rely=0.45, anchor=CENTER)
    # Analysis section
    tk.Label(analysis_section, text="Options for data analysis: ", font=('Helvetica', 11, 'bold')).place(relx=0.2,
                                                                                                         rely=0.7,
                                                                                                         anchor=CENTER)
    tk.Button(analysis_section, text="Gather Weather Report",
              command=lambda: weather_report.weather_gui()).place(relx=0.5, rely=0.7, anchor=CENTER)
    tk.Button(analysis_section, text="Video Analysis",
              command=lambda: video_clip.get_clips()).place(relx=0.8, rely=0.7, anchor=CENTER)

    # # Footer with UHN and EHT Logos
    # # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    # kite_logo = ImageTk.PhotoImage(Image.open('uhn_kite_logo.png'))
    # # global panel variable is set to a canvas on render and is global to draw lines from the other methods
    # kite = tk.Canvas(window, width=winWidth, height=118, bg='white', bd=0, relief='ridge')
    # # image is placed from the top left origin and placed on the northwest/topleft section of the 'panel' canvas
    # kite.create_image(0, 0, anchor='nw', image=kite_logo)
    # kite.place(relx=0.5, rely=0.91, anchor=CENTER)
    #
    # # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    # eht_logo = ImageTk.PhotoImage(Image.open('EHT_Logo.png'))
    # # global panel variable is set to a canvas on render and is global to draw lines from the other methods
    # # image is placed from the top left origin and placed on the northwest/topleft section of the 'panel' canvas
    # kite.create_image(winWidth-120, 0, anchor='nw', image=eht_logo)
    # # Create window
    window.mainloop()


# Opens user directory to select mp4 file to analyze
def open_file():
    # open directory to select mp4 files only
    file_path = fd.askopenfile(mode='r', filetypes=[('MP4 Files', '*mp4')])
    # check if it ws selected
    if file_path is not None:
        # split directory string into it's folder names (each /)
        split_dir = file_path.name.split('/')
        # file name to display is the final item of the list
        file_name = split_dir[len(split_dir) - 1]
        # set global variable
        global input_file
        input_file = file_path.name
        print(str(file_path.name))

        # label to display the file name
        tk.Label(file_frame, text=file_name, padx=50).place(relx=0.3, rely=0.25, anchor=CENTER)
        global videoUp, folderUp
        videoUp = True
        if videoUp & folderUp:
            # select tracking region button
            tk.Button(file_frame, text='Select Tracking Region',
                      command=lambda: tracking_window()).place(relx=0.5, rely=0.3, anchor=CENTER)
        return file_path.name


# Opens user directory to select the folder to output yolov4 outputs
def destination_folder():
    # open directory to select an output folder
    folder_selected = fd.askdirectory()
    if folder_selected is not None:
        t = folder_selected.split(" ")
        if len(t) != 1:
            output_error = tk.Toplevel()
            geo = '240x240'
            output_error.title('ERROR')
            output_error.geometry(geo)
            tk.Label(output_error, text="Output folder directory cannot have spaces").place(relx=0.5, rely=0.5,
                                                                                            anchor=CENTER)
            tk.Button(output_error, text="Close", command=lambda: output_error.destroy()) \
                .place(relx=0.5, rely=0.9, anchor=CENTER)
        else:
            # split file tree into folders
            split_fold = folder_selected.split('/')
            # display name of the folder
            folder_name = split_fold[len(split_fold) - 1]
            # set global variable
            global output_folder
            output_folder = folder_selected
            tk.Label(file_frame, text="Folder selected: " + folder_name, padx=30).place(relx=0.7, rely=0.25,
                                                                                        anchor=CENTER)
            global videoUp, folderUp
            folderUp = True
            if videoUp & folderUp:
                # select tracking region button
                tk.Button(file_frame, text='Select Tracking Region',
                          command=lambda: tracking_window()).place(relx=0.5, rely=0.3, anchor=CENTER)
            pass


# runs python command to run yolov4 and sends the folders as parameters
def upload_file():
    if input_file != "" and output_folder != "":
        tk.Label(file_frame, text='Running YOLOv4 script, video output displaying shortly',
                 foreground='green').place(relx=0.5, rely=0.5, anchor=CENTER)

        # remove mp4 from filename
        #   split directory string into it's folder names
        split_dir = input_file.split('/')
        #   file name to display is the final item of the list
        testing = split_dir[len(split_dir) - 1]
        file_name = testing.split('.mp4')
        output_filename = output_folder + "/" + file_name[0] + "_OUTPUT.avi"
        # global coordinates
        coords = str(coordinates[0] * widthFactor) + ',' + str(coordinates[1] * widthFactor) + ',' + \
                 str(coordinates[2] * widthFactor) + ',' + str(coordinates[3] * widthFactor) + ',' + \
                 str(coordinates[4] * heightFactor) + ',' + str(coordinates[5] * heightFactor) + ',' + \
                 str(coordinates[6] * heightFactor) + ',' + str(coordinates[7] * heightFactor)
        # Run Python command using os.system (call tracker and send input, output w hardcoded model and tiny flags
        py_command = "python zeyad_tracker.py --video " + input_file + " --output " + output_filename + \
                     " --model yolov4" + " --tiny" + " --coordinates " + coords
        os.system(py_command)
    else:
        if input_file == "" and output_folder != "":
            tk.Label(file_frame, text='Select mp4 file',
                     foreground='red').place(relx=0.5, rely=0.5, anchor=CENTER)
        else:
            tk.Label(file_frame, text='Select output folder',
                     foreground='red').place(relx=0.5, rely=0.5, anchor=CENTER)


"""
Creating a new window to select the tracking region of the system

NOTES:
- Every tkinter object is loaded on the initial run unless within a conditional statement
- Global variables are used to save the tracking region blocks across the methods

"""

# GLOBAL VARS
coordinates = [0, 0, 0, 0, 0, 0, 0, 0]
panel = ''
track_win = ''
y_0 = ''
x_0 = ''
x_1 = ''
y_1 = ''
y_2 = ''
x_2 = ''
x_3 = ''
y_3 = ''


# main function to open tracking track_win and display the intersection to select points to track
def tracking_window():
    global track_win, input_file
    track_win = tk.Toplevel()
    track_win.title('Tracking window')
    track_width = 1200
    track_height = 650
    track_geometry = str(track_width) + "x" + str(track_height)
    track_win.geometry(track_geometry)

    # Title
    done_title = tk.Label(track_win, text='Select four points \n for the tracking region', fg="green")
    done_title.config(font=('Arial', 20))
    done_title.place(relx=0.2, rely=0.1, anchor=CENTER)
    # Displaying x-coordinates
    tk.Label(track_win, text='X-Coordinates: ').place(relx=0.15, rely=0.3, anchor=CENTER)

    # global variables must be set initially as empty strings then set to tkinter objects once rendered
    global x_0, x_1, x_2, x_3
    x_0 = tk.StringVar()
    x_1 = tk.StringVar()
    x_2 = tk.StringVar()
    x_3 = tk.StringVar()
    # to get the information typed into an Entry field, place command must be used in a separate line
    # text variables are the global variables
    x0 = tk.Entry(track_win, width=8, textvariable=x_0)
    x0.place(relx=0.15, rely=0.4, anchor=CENTER)
    x1 = tk.Entry(track_win, width=8, textvariable=x_1)
    x1.place(relx=0.15, rely=0.5, anchor=CENTER)
    x2 = tk.Entry(track_win, width=8, textvariable=x_2)
    x2.place(relx=0.15, rely=0.6, anchor=CENTER)
    x3 = tk.Entry(track_win, width=8, textvariable=x_3)
    x3.place(relx=0.15, rely=0.7, anchor=CENTER)
    # Displaying y-coordinates
    tk.Label(track_win, text='Y-Coordinates: ').place(relx=0.25, rely=0.3, anchor=CENTER)
    # global variables must be set initially as empty strings then set to tkinter objects once rendered
    global y_0, y_1, y_2, y_3
    y_0 = tk.StringVar()
    y_1 = tk.StringVar()
    y_2 = tk.StringVar()
    y_3 = tk.StringVar()

    y0 = tk.Entry(track_win, width=8, textvariable=y_0)
    y0.place(relx=0.25, rely=0.4, anchor=CENTER)
    y1 = tk.Entry(track_win, width=8, textvariable=y_1)
    y1.place(relx=0.25, rely=0.5, anchor=CENTER)
    y2 = tk.Entry(track_win, width=8, textvariable=y_2)
    y2.place(relx=0.25, rely=0.6, anchor=CENTER)
    y3 = tk.Entry(track_win, width=8, textvariable=y_3)
    y3.place(relx=0.25, rely=0.7, anchor=CENTER)
    # Button to draw a new box given inputted data
    box = tk.Button(track_win, text="Draw Region", command=lambda: draw_box(
        int(x0.get()), int(y0.get()), int(x1.get()), int(y1.get()),
        int(x2.get()), int(y2.get()), int(x3.get()), int(y3.get())
    ))
    box.place(relx=0.1, rely=0.8, anchor=CENTER)
    load_temp = tk.Button(track_win, text="Load Templates", command=lambda: open_templates())
    load_temp.place(relx=0.2, rely=0.8, anchor=CENTER)
    save_temp = tk.Button(track_win, text="Save New Template", command=lambda: save_template())
    save_temp.place(relx=0.26, rely=0.8, anchor='w')

    # Displaying intersection image to draw tracking region
    # Opens the Video file
    cap = cv2.VideoCapture(input_file)
    ret, frame = cap.read()
    print(type(frame))
    print(frame.shape)
    global output_folder, widthFactor, heightFactor
    # im = cv2.imread(output_folder + '/intersectionFrame.jpg')
    h, w, c = frame.shape
    if w > 640:
        resize_width = 640
        widthFactor = w / resize_width
    else:
        resize_width = w
    if h > 480:
        resize_height = 480
        heightFactor = h / resize_height
    else:
        resize_height = h

    dim = (resize_width, resize_height)
    # resize image
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_folder + '/intersectionFrame.jpg', resized)
    path = output_folder + '/intersectionFrame.jpg'
    # Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
    intersection = ImageTk.PhotoImage(Image.open(path))
    # global panel variable is set to a canvas on render and is global to draw lines from the other methods
    global panel
    panel = tk.Canvas(track_win, cursor='cross', width=750, height=h)
    # image is placed from the top left origin and placed on the northwest/topleft section of the 'panel' canvas
    panel.create_image(0, 0, anchor='nw', image=intersection)
    panel.place(relx=0.75, rely=0.5, anchor=CENTER)
    # on each button press, call on_button_press
    panel.bind("<ButtonPress-1>", on_button_press)
    panel.rect = panel.start_x = panel.start_y = None
    # Clearing selected track_win
    clear = tk.Button(track_win, text='Clear Selection', command=lambda: empty())
    clear.place(relx=0.15, rely=0.9, anchor=CENTER)
    # Save region and save template
    save_btn = tk.Button(track_win, text='Save Tracking Region', bg='light green', command=lambda: saveRegion())
    save_btn.place(relx=0.2, rely=0.9, anchor='w')

    track_win.mainloop()


# Draw the lines to create a tracking region based on 4 button clicks and save to entry values
def on_button_press(event):
    # save mouse drag start position
    x = event.x
    y = event.y
    global coordinates
    global x_0, x_1, y_0, y_1, x_2, y_2, x_3, y_3
    # Fill in x and y coordinates in both the array and the entry widget values
    if coordinates[0] == 0:
        coordinates[0] = x
        coordinates[4] = y
        x_0.set(coordinates[0])
        y_0.set(coordinates[4])
    elif coordinates[1] == 0:
        coordinates[1] = x
        coordinates[5] = y
        # create a line between the first clicks
        panel.create_line(coordinates[0], coordinates[4], coordinates[1], coordinates[5], fill="green", tags='line1')
        x_1.set(coordinates[1])
        y_1.set(coordinates[5])
    elif coordinates[2] == 0:
        coordinates[2] = x
        coordinates[6] = y
        x_2.set(coordinates[2])
        y_2.set(coordinates[6])
        panel.create_line(coordinates[1], coordinates[5], coordinates[2], coordinates[6], fill="black", tags='line3')
    elif coordinates[3] == 0:
        coordinates[3] = x
        coordinates[7] = y
        panel.create_line(coordinates[2], coordinates[6], coordinates[3], coordinates[7], fill="green", tags='line2')
        panel.create_line(coordinates[0], coordinates[4], coordinates[3], coordinates[7], fill="black", tags='line4')
        x_3.set(coordinates[3])
        y_3.set(coordinates[7])
    else:
        print("Box is Full")


# Use the filled in entries to draw a box with black lines and replace the coordinates
def draw_box(xcoor0, ycoor0, xcoor1, ycoor1, xcoor2, ycoor2, xcoor3, ycoor3):
    line1 = panel.create_line(xcoor0, ycoor0, xcoor1, ycoor1, fill="green", tags='line1')
    line2 = panel.create_line(xcoor2, ycoor2, xcoor3, ycoor3, fill="green", tags='line2')
    line3 = panel.create_line(xcoor1, ycoor1, xcoor2, ycoor2, fill="black", tags='line3')
    line4 = panel.create_line(xcoor0, ycoor0, xcoor3, ycoor3, fill="black", tags='line4')

    global coordinates
    coordinates = [0, 0, 0, 0, 0, 0, 0, 0]
    coordinates[0] = xcoor0
    coordinates[1] = xcoor1
    coordinates[2] = xcoor2
    coordinates[3] = xcoor3
    coordinates[4] = ycoor0
    coordinates[5] = ycoor1
    coordinates[6] = ycoor2
    coordinates[7] = ycoor3
    global x_0, x_1, y_0, y_1, x_2, y_2, x_3, y_3
    x_0.set(xcoor0)
    y_0.set(ycoor0)
    x_1.set(xcoor1)
    y_1.set(ycoor1)
    x_2.set(xcoor2)
    y_2.set(ycoor2)
    x_3.set(xcoor3)
    y_3.set(ycoor3)


# clears the contents of the coordinates
def empty():
    global coordinates, panel
    panel.delete(panel.find_withtag('line1'))
    panel.delete(panel.find_withtag('line2'))
    panel.delete(panel.find_withtag('line3'))
    panel.delete(panel.find_withtag('line4'))
    coordinates = [0, 0, 0, 0, 0, 0, 0, 0]
    x_0.set(0)
    y_0.set(0)
    x_1.set(0)
    y_1.set(0)
    x_2.set(0)
    y_2.set(0)
    x_3.set(0)
    y_3.set(0)


# global variables are set, closes tracking window and gives comment
def saveRegion():
    global track_win, window
    track_win.destroy()
    tk.Label(window, text="Region Selected!", fg="green").place(relx=0.7, rely=0.3, anchor=CENTER)


# GLOBAL vars for loading and saving templates
tracking_temps = "tracking_region_templates.csv"
temp_view = ''


# Selecting tracking templates
def open_templates():
    global temp_view
    temp_view = tk.Toplevel()
    temp_view.title('Select Template')
    temp_width = 360
    temp_height = 360
    temp_geometry = str(temp_width) + "x" + str(temp_height)
    temp_view.geometry(temp_geometry)
    temp_view.resizable(False, False)
    # Header
    tk.Label(temp_view, text="Double-Click to select a preset template").place(relx=0.2, rely=0.1, anchor=CENTER)
    tk.Label(temp_view, text="Names of the Templates").place(relx=0.3, rely=0.2, anchor=CENTER)
    # Scrollable list of templates
    listbox = tk.Listbox(temp_view)
    listbox.place(relx=0.5, rely=0.5, anchor=CENTER)
    # Scrollbar
    scrollbar = tk.Scrollbar(temp_view, orient='vertical')
    scrollbar.pack(side=RIGHT, fill=BOTH)
    y = 0.4
    trt = pd.read_csv(tracking_temps)
    for row in trt["Title"]:
        listbox.insert(END, row)
    listbox.bind("<Double-Button-1>", selected)
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)


# Function bound for a selected template, read and pass coordinates to draw the box
def selected(event):
    # grab name of the template from the event doubl click
    idx = int(event.widget.curselection()[0])
    value = event.widget.get(idx)
    # read csv and gather values of the template by searching the name
    trt = pd.read_csv(tracking_temps)
    a = trt.loc[(trt["Title"] == value)].index.values[0]
    draw_box(
        int(trt.iloc[a, 1]), int(trt.iloc[a, 2]), int(trt.iloc[a, 3]), int(trt.iloc[a, 4]),
        int(trt.iloc[a, 5]), int(trt.iloc[a, 6]), int(trt.iloc[a, 7]), int(trt.iloc[a, 8])
    )
    temp_view.destroy()


# Save a new template, check first if the name exists
def save_template():
    global coordinates
    trt = pd.read_csv(tracking_temps)
    for row in trt["Title"]:
        a = trt.loc[(trt["Title"] == row)].index.values
        b = [int(trt.iloc[a, 1]), int(trt.iloc[a, 2]), int(trt.iloc[a, 3]), int(trt.iloc[a, 4]),
             int(trt.iloc[a, 5]), int(trt.iloc[a, 6]), int(trt.iloc[a, 7]), int(trt.iloc[a, 8])]
        if b == coordinates:
            no_save = tk.Toplevel()
            geo = '240x240'
            no_save.title('ERROR')
            no_save.geometry(geo)
            tk.Label(no_save, text="Template already exists!").place(relx=0.5, rely=0.5, anchor=CENTER)
            tk.Label(no_save, text="Use " + str(row)).place(relx=0.5, rely=0.6, anchor=CENTER)
            tk.Button(no_save, text="Close", command=lambda: no_save.destroy()) \
                .place(relx=0.5, rely=0.9, anchor=CENTER)
            break
        else:
            saved = tk.Toplevel()
            geo = '240x120'
            saved.title('New Template')
            saved.geometry(geo)
            tk.Label(saved, text="Name the new template:", fg="green").place(relx=0.5, rely=0.4, anchor=CENTER)
            new_name = tk.StringVar()
            tk.Entry(saved, textvariable=new_name, width=16).place(relx=0.3, rely=0.6, anchor=CENTER)
            tk.Button(saved, text="Save", command=lambda: save(new_name, saved)).place(relx=0.7, rely=0.6,
                                                                                       anchor=CENTER)
            break


# Save new template by writing to csv
def save(new_name, saved):
    new_row = [str(new_name.get())]
    for x in coordinates:
        new_row.append(str(x))
    with open(tracking_temps, 'a', newline='') as result_file:
        wr = csv.writer(result_file)
        wr.writerow(new_row)
    saved.destroy()


if __name__ == '__main__':
    try:
        app.run(main())
    except SystemExit:
        pass
