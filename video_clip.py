import pickle
from collections import defaultdict

import imageio
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import csv
import os
import tkinter as tk
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from tkinter import HORIZONTAL, CENTER, VERTICAL, RIGHT, FALSE, BOTH, END, N
import tkinter.filedialog as fd
from tkinter.ttk import Progressbar
import cv2 as cv
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import pandas as pd
from absl import app

# imageio.plugins.ffmpeg.download()

# GLOBAL Vars
window = ""
video_file = ""
video_file_name = ""
csv_file = ""
csv_output = ""
ped_data = ""
coordinates = []
id_list = []
id_path = defaultdict(list)
id_Frames = defaultdict(list)
output_video = ""
v_0 = ""
v_1 = ""
p_0 = ""
p_1 = ""


# uploads csv for weather report
def select_csv():
    # open directory to select csv files only
    file_path = fd.askopenfile(mode='r', filetypes=[('csv Files', '*csv')])
    # check if it ws selected
    if file_path is not None:
        # label to display the file name
        global csv_file, csv_output, coordinates, output_video, ped_data
        csv_file = file_path.name
        split_dir = file_path.name.split('/')
        # file name to display is the final item of the list
        csv_name = split_dir[len(split_dir) - 1]
        csv_output = pd.read_csv(csv_file, header=None, skiprows=[0, 1, 2])
        a = csv_file.split('.csv')
        output_video = a[0]
        coordinates = [int(csv_output.iloc[0, 1]), int(csv_output.iloc[0, 2]),
                       int(csv_output.iloc[0, 3]), int(csv_output.iloc[0, 4]),
                       int(csv_output.iloc[1, 1]), int(csv_output.iloc[1, 2]),
                       int(csv_output.iloc[1, 3]), int(csv_output.iloc[1, 4])]

        ped_data = pd.read_csv(csv_file, skiprows=[0, 1, 2, 3, 4])

        tk.Label(window, text=csv_name, padx=50).place(relx=0.7, rely=0.25, anchor=CENTER)
        return file_path.name


# Opens user directory to select mp4 file to analyze
def select_video():
    # open directory to select mp4 files only
    file_path = fd.askopenfile(mode='r', filetypes=[('Video Files', '*avi *mp4'), ('MP4 Files', '*mp4')])
    # check if it ws selected
    if file_path is not None:
        # split directory string into it's folder names (each /)
        split_dir = file_path.name.split('/')
        # file name to display is the final item of the list
        global video_file_name
        video_file_name = split_dir[len(split_dir) - 1]
        # set global variable
        global video_file
        video_file = file_path.name
        # label to display the file name
        tk.Label(window, text=video_file_name, padx=50).place(relx=0.3, rely=0.25, anchor=CENTER)
        return file_path.name


def extract_clips():
    # open pickle file and unpackage the byte stream
    with open(output_video + ".pickle", 'rb') as handle:
        tracking = pickle.load(handle)

    print(len(tracking))
    # # # #
    no_persons = []
    for i in range(len(tracking)):
        if int(tracking[i][1]) not in no_persons:
            no_persons.append(int(tracking[i][1]))
        else:
            continue

    print("Number of assigned ids = ", len(no_persons))
    print(no_persons)
    #
    # load frames and path for each id in the frame
    global id_path, id_Frames
    for i in range(len(tracking)):
        for ped_id in no_persons:
            if ped_id == int(tracking[i][1]) and (not int(tracking[i][1]) in list(id_path[ped_id])):
                id_path[ped_id].append((int(tracking[i][2]), int(tracking[i][3])))
                id_Frames[ped_id].append(tracking[i][0])
            else:
                continue

    # get list of ids within the speed specified
    ids = ped_data.loc[(ped_data["Speed (m/s)"] >= float(v_0.get())) & (ped_data["Speed (m/s)"] <= float(v_1.get()))]
    print(ids)

    # get frames from specific
    for i in range(len(ids.index)):
        person_id = int(ids.iloc[i, 0])
        id_list.append(person_id)
        start = int(ids.iloc[i, 2])
        final = int(ids.iloc[i, 3])
        print(str(person_id) + ": " + str(start) + " - " + str(final))
        # generate name of id's clip
        input_video = str(video_file)
        # get file directory but file name
        split_dir = input_video.split('/')
        output = '/'.join(split_dir[0:len(split_dir) - 1])
        sub_clip = output + '/id' + str(person_id) + '_clip.mp4'

        # create subclip between start and final times
        input_cap = cv.VideoCapture(input_video)
        fps = input_cap.get(cv.CAP_PROP_FPS)
        ffmpeg_extract_subclip(input_video, start / fps, final / fps, targetname=sub_clip)
        input_cap.release()

        # Read the video frame, then write the file and display it in the window
        cap = cv.VideoCapture(sub_clip)
        fps = cap.get(cv.CAP_PROP_FPS)
        # print("fps = ", fps)
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        zoomed_in = output + '/track_id' + str(person_id) + '.mp4'
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv.VideoWriter(zoomed_in, fourcc, fps, (width, height))
        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # OLD ZOOMING Functionality
                # crop_frame = frame[int(min(coordinates[4:len(coordinates)])):int(max(coordinates[4:len(coordinates)])),
                #              int(min(coordinates[0:4])):int(max(coordinates[0:4]))]
                # # write the cropped frame
                # resize = cv.resize(crop_frame, (320, 240))

                # Black boxes around the tracking region
                # left side
                cv.rectangle(frame, (0, 0), (int(min(coordinates[0:4])), height), (0, 0, 0), -1)
                # right side
                cv.rectangle(frame, (int(max(coordinates[0:4])), 0), (width, height), (0, 0, 0), -1)
                # Top side
                cv.rectangle(frame, (0, 0), (width, int(min(coordinates[4:len(coordinates)]))), (0, 0, 0), -1)
                # Bottom side
                cv.rectangle(frame, (0, int(max(coordinates[4:len(coordinates)]))), (width, height), (0, 0, 0), -1)
                # draw unique points of the trajectory
                # Center coordinates
                center_coordinates = (id_path[person_id][counter])
                counter += 1
                # Radius of circle
                radius = 12
                # Blue color in BGR
                color = (255, 0, 0)
                # Line thickness of 2 px
                thickness = 1
                # Draw a circle with blue line borders of thickness of 2 px
                cv.circle(frame, center_coordinates, radius, color, thickness)
                out.write(frame)
            else:
                break
        print("Video cropping completed : ", person_id)
        # Release reader wand writer after parsing all frames
        cap.release()
        out.release()
        tk.Button(window, text="Plot data", command=lambda: animate_data()).place(relx=0.5, rely=0.7, anchor=CENTER)
        # cv.destroyAllWindows()


def animate_data():
    for key, value in id_path.items():
        if key in id_list:
            x_pixel = [pt[0] for pt in value]
            y_pixel = [pt[1] for pt in value]

            x = [element * 1 for element in x_pixel]  # put scaling factor instead of 1
            y = [element * 1 for element in y_pixel]  # put scaling factor instead of 1

            ###########################################################
            # create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1)

            # intialize two line objects (one in each axes)
            line1, = ax1.plot([], [], lw=2)
            line2, = ax2.plot([], [], lw=2, color='r')

            ax1.set_ylim(min(x) - 10, max(x) + 10)
            ax1.set_xlim(0, len(x))
            # ax1.grid()

            ax2.set_ylim(min(y) - 10, max(y) + 10)
            ax2.set_xlim(0, len(y))
            # ax2.grid()

            # initialize the data arrays
            xdata, y1data, y2data = [], [], []

            def run(i):
                xdata.append(i)
                y1data.append(x[i])
                y2data.append(y[i])

                # update the data of both line objects
                line1.set_data(xdata, y1data)
                line2.set_data(xdata, y2data)

                return line1, line2,

            ani = animation.FuncAnimation(fig, run, blit=True, interval=20)
            plt.show()


def get_clips():
    global window
    window = tk.Toplevel()
    window.title('Video Highlights')
    win_width = 640
    height = 480
    geometry = str(win_width) + "x" + str(height)
    window.geometry(geometry)
    window.resizable(False, False)
    # First row labels for selecting files
    tk.Label(window, text='Upload video output in mp4 format ',
             font=('Helvetica', 11, 'bold')).place(relx=0.3, rely=0.15, anchor=CENTER)
    select_output = tk.Label(window, text='Upload csv output ',
                             font=('Helvetica', 11, 'bold')).place(relx=0.7, rely=0.15, anchor=CENTER)
    # Second row buttons to open directory
    tk.Button(window, text='Choose video',
              command=lambda: select_video()).place(relx=0.3, rely=0.2, anchor=CENTER)
    tk.Button(window, text='Choose .csv',
              command=lambda: select_csv()).place(relx=0.7, rely=0.2, anchor=CENTER)

    # User inputs values that would be important to examine (clips of specific speeds or percentile groups of persons
    speed_label = tk.Label(window, text="What speeds are important? (in m/s)", font=('Helvetica', 11, 'bold'))
    speed_label.place(relx=0.5, rely=0.4, anchor=CENTER)
    global v_0, v_1
    v_0 = tk.StringVar()
    v_1 = tk.StringVar()
    v0 = tk.Entry(window, width=8, textvariable=v_0)
    v0.place(relx=0.4, rely=0.5, anchor=CENTER)
    tk.Label(window, text="to").place(relx=0.5, rely=0.5, anchor=CENTER)
    v1 = tk.Entry(window, width=8, textvariable=v_1)
    v1.place(relx=0.6, rely=0.5, anchor=CENTER)
    # speed_label = tk.Label(window, text="What percentiles are important?", font=('Helvetica', 11, 'bold'))
    # speed_label.place(relx=0.5, rely=0.55, anchor=CENTER)
    # global p_0, p_1
    # p_0 = tk.StringVar()
    # p_1 = tk.StringVar()
    # p0 = tk.Entry(window, width=8, textvariable=p_0)
    # p0.place(relx=0.4, rely=0.6, anchor=CENTER)
    # tk.Label(window, text="to").place(relx=0.5, rely=0.6, anchor=CENTER)
    # p1 = tk.Entry(window, width=8, textvariable=p_1)
    # p1.place(relx=0.6, rely=0.6, anchor=CENTER)

    tk.Button(window, text='Extract clips',
              command=lambda: extract_clips()).place(relx=0.5, rely=0.6, anchor=CENTER)
