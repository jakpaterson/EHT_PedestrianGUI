import pickle
import tkinter as tk
from collections import defaultdict
from tkinter.ttk import Progressbar

import matplotlib
import matplotlib.animation as animation
from tkinter import CENTER, HORIZONTAL
import tkinter.filedialog as fd
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
# uncomment when creating executable
# os.environ["IMAGEIO_FFMPEG_EXE"] = r"MyFFMPEG_PATH"
import moviepy.video.io.ffmpeg_tools as fm

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
file_dir = ""
v_0 = ""
v_1 = ""
p_0 = ""
p_1 = ""
plot_id = ""
videoBool = False
csvBool = False


# Opens user directory to select mp4 file to analyze
def select_video():
    global videoBool, csvBool
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
        tk.Label(window, text=video_file_name, padx=50).place(relx=0.3, rely=0.3, anchor=CENTER)
        videoBool = True
        if csvBool & videoBool:
            parse_data()
        return file_path.name


def select_csv():
    global videoBool, csvBool
    # open directory to select csv files only
    file_path = fd.askopenfile(mode='r', filetypes=[('csv Files', '*csv')])
    # check if it ws selected
    if file_path is not None:
        # label to display the file name
        global csv_file, csv_output, coordinates, file_dir, ped_data
        csv_file = file_path.name
        split_dir = file_path.name.split('/')
        # file name to display is the final item of the list
        csv_name = split_dir[len(split_dir) - 1]
        csv_output = pd.read_csv(csv_file, header=None, skiprows=[0, 1, 2])
        a = csv_file.split('.csv')
        file_dir = a[0]
        # grab coordinates of the csv file
        coordinates = [int(csv_output.iloc[0, 1]), int(csv_output.iloc[0, 2]),
                       int(csv_output.iloc[0, 3]), int(csv_output.iloc[0, 4]),
                       int(csv_output.iloc[1, 1]), int(csv_output.iloc[1, 2]),
                       int(csv_output.iloc[1, 3]), int(csv_output.iloc[1, 4])]

        ped_data = pd.read_csv(csv_file, skiprows=[0, 1, 2, 3, 4])

        tk.Label(window, text=csv_name, padx=50).place(relx=0.7, rely=0.3, anchor=CENTER)
        csvBool = True
        if csvBool & videoBool:
            parse_data()
        return file_path.name


def parse_data():
    pb1 = Progressbar(
        window,
        orient=HORIZONTAL,
        length=300,
        mode='determinate'
    )
    pb1.place(relx=0.5, rely=0.5, anchor=CENTER)
    # open pickle file and unpackage the byte stream
    with open(file_dir + ".pickle", 'rb') as handle:
        tracking = pickle.load(handle)

    # grab number of persons
    no_persons = []
    for i in range(len(tracking)):
        if int(tracking[i][1]) not in no_persons:
            no_persons.append(int(tracking[i][1]))
        else:
            continue

    print("Number of assigned ids = ", len(no_persons))
    print(no_persons)

    # load frame number and coordinates for each id in the frame
    global id_path, id_Frames
    for i in range(len(tracking)):
        for ped_id in no_persons:
            if ped_id == int(tracking[i][1]) and (not int(tracking[i][1]) in list(id_path[ped_id])):
                id_path[ped_id].append((int(tracking[i][2]), int(tracking[i][3])))
                id_Frames[ped_id].append(tracking[i][0])
                pb1['value'] += 1
                window.update_idletasks()
            else:
                continue

    pb1.destroy()

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
    tk.Button(window, text='Extract clips', command=lambda: extract_clips()).place(relx=0.5, rely=0.6,
                                                                                   anchor=CENTER)
    # plot data
    tk.Label(window, text="Input ID to plot positional data").place(relx=0.3, rely=0.7, anchor=CENTER)
    global plot_id
    plot_id = tk.StringVar()
    # input_id = tk.Entry(window, width=8, textvariable=plot_id)
    plot_id.set("1")  # default value
    print(ped_data.iloc[:, 0])
    print(ped_data)
    list_string = list(map(str, ped_data.iloc[:, 0]))
    # Printing sorted list of integers
    input_id = tk.OptionMenu(window, plot_id, *list_string)
    input_id.place(relx=0.5, rely=0.7, anchor=CENTER)
    tk.Button(window, text="Plot speed", command=lambda: plot_speed()).place(relx=0.7, rely=0.7, anchor=CENTER)


def extract_clips():
    # get list of ids within the speed specified
    ids = ped_data.loc[(ped_data["Speed (m/s)"] >= float(v_0.get())) & (ped_data["Speed (m/s)"] <= float(v_1.get()))]
    print(ids)

    # for each id within the speed selected, clip a video of their walking area
    for i in range(len(ids.index)):
        person_id = int(ids.iloc[i, 0])
        id_list.append(person_id)
        # grab first and last frames the id was in
        start = int(ids.iloc[i, 2])
        final = int(ids.iloc[i, 3])
        print(str(person_id) + ": " + str(start) + " - " + str(final))

        # generate name of id's clip
        print(video_file)
        input_video = str(video_file)
        # get file directory but file name
        split_dir = input_video.split('/')
        input_video = '\\'.join(split_dir)
        print(input_video)
        output = '\\'.join(split_dir[0:len(split_dir) - 1])
        sub_clip = output + '\\id' + str(person_id) + '_split_video.mp4'
        print(sub_clip)
        print(output)

        # create subclip between start and final times
        input_cap = cv.VideoCapture(input_video)
        fps = input_cap.get(cv.CAP_PROP_FPS)
        fm.ffmpeg_extract_subclip(input_video, start / fps, final / fps, targetname=sub_clip)
        input_cap.release()

        # Read the video frame, then write the file and display it in the window
        cap = cv.VideoCapture(sub_clip)
        fps = cap.get(cv.CAP_PROP_FPS)
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        zoomed_in = output + '\\track_id' + str(person_id) + '.mp4'
        # TODO: h264 - cv.VideoWriter_fourcc('H', '2', '6', '4')??
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv.VideoWriter(zoomed_in, fourcc, fps, (width, height))
        counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                overlay = frame.copy()
                # OLD ZOOMING Functionality crop_frame = frame[int(min(coordinates[4:len(coordinates)])):int(max(
                # coordinates[4:len(coordinates)])), int(min(coordinates[0:4])):int(max(coordinates[0:4]))] # write
                # the cropped frame resize = cv.resize(crop_frame, (320, 240)) id
                cv.putText(frame, "Tracking ID: " + str(person_id), (10, 50), 0, 0.5, (0, 255, 0), 2)
                # Black boxes around the tracking region
                # left side
                cv.rectangle(overlay, (0, 0), (int(min(coordinates[0:4])), height), (0, 0, 0), -1)
                # right side
                cv.rectangle(overlay, (int(max(coordinates[0:4])), 0), (width, height), (0, 0, 0), -1)
                # Top side
                cv.rectangle(overlay, (0, 0), (width, int(min(coordinates[4:len(coordinates)]))), (0, 0, 0), -1)
                # Bottom side
                cv.rectangle(overlay, (0, int(max(coordinates[4:len(coordinates)]))), (width, height), (0, 0, 0), -1)
                # apply the overlay
                alpha = 0.5
                cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                # draw unique points of the trajectory
                # Center coordinates
                print(str(counter) + ' - ' + str(len(id_path[person_id])))
                if counter < len(id_path[person_id]):
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
        # cv.destroyAllWindows()


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_speed():
    global plot_id
    data = id_path[int(plot_id.get())]
    x_pixel = [pt[0] for pt in data]
    y_pixel = [pt[1] for pt in data]

    # 0.15 if factor to convert pixels to m/s

    x = [element * 0.15 for element in x_pixel]
    y = [element * 0.15 for element in y_pixel]

    # x squared and y squared
    x2 = [a * b for a, b in zip(x, x)]
    y2 = [a * b for a, b in zip(y, y)]

    # numpy array
    s = np.sqrt([x + y for x, y in zip(x2, y2)])
    t = np.linspace(0, 100, len(s))

    speed = (abs(np.diff(s))) * 12  # 12 : Frame Rate
    MAstep = (len(speed) // 5)
    speed_smooth = moving_average(speed, MAstep)
    t_speed = np.linspace(0, 100, len(speed_smooth))

    """Plot Speed"""
    plt.plot(t_speed, speed_smooth)
    plt.title("Walking Speed of Pedestrian ID: " + str(plot_id.get()))
    plt.ylabel("Speed (m/s)")
    plt.xlabel("Walking Duration (s)")
    plt.axis(ymin=0)
    plt.show()

    ###########################################################
    # create a figure with two subplots for x and y position
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # # initialize two line objects (one in each axes)
    # ax1.plot(list(range(0, len(x))), x, lw=2)
    # ax2.plot(list(range(0, len(y))), y, lw=2, color='r')
    #
    # ax1.set_ylim(min(x) - 10, max(x) + 10)
    # ax1.set_xlim(0, len(x))
    # # ax1.grid()
    # # ax1.ylabel('Time (s)')
    # # ax1.set_xlabel('Position in x direction (m)')
    #
    # ax2.set_ylim(min(y) - 10, max(y) + 10)
    # ax2.set_xlim(0, len(y))
    # # ax2.grid()
    # # ax2.set_xlabel('Position in y direction (m)')

    # initialize the data arrays
    plot_dir = file_dir.split('/')
    output = '/'.join(plot_dir[0:len(plot_dir) - 1])
    plt.savefig(output + 'id_' + str(plot_id.get()) + '_plot.png')


def get_clips():
    global window
    window = tk.Toplevel()
    window.title('Video Analytics')
    win_width = 640
    height = 480
    geometry = str(win_width) + "x" + str(height)
    window.geometry(geometry)
    window.resizable(False, False)
    tk.Label(window, text="Breakdown your analyzed footage! ",
             font=('Helvetica', 11, 'bold')).place(relx=0.5, rely=0.05, anchor=CENTER)
    tk.Label(window, text="You can upload the original or tracked video along with the csv output.",
             font=('Helvetica', 11, 'bold')).place(relx=0.5, rely=0.1, anchor=CENTER)
    tk.Frame(window, width=win_width, height=2, bg="#6D2C9E").place(relx=0.5, rely=0.15,
                                                                    anchor=CENTER)
    # First row labels for selecting files
    tk.Label(window, text='Upload video output in mp4 format ',
             font=('Helvetica', 11, 'bold')).place(relx=0.3, rely=0.2, anchor=CENTER)
    select_output = tk.Label(window, text='Upload csv output ',
                             font=('Helvetica', 11, 'bold')).place(relx=0.7, rely=0.2, anchor=CENTER)
    # Second row buttons to open directory
    tk.Button(window, text='Choose video',
              command=lambda: select_video()).place(relx=0.3, rely=0.25, anchor=CENTER)
    tk.Button(window, text='Choose .csv',
              command=lambda: select_csv()).place(relx=0.7, rely=0.25, anchor=CENTER)


def analysis_section():
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
    tk.Button(window, text='Extract clips', command=lambda: extract_clips()).place(relx=0.5, rely=0.6, anchor=CENTER)
    # plot data
    tk.Label(window, text="Input ID to plot positional data").place(relx=0.3, rely=0.7, anchor=CENTER)
    global plot_id
    plot_id = tk.StringVar()
    input_id = tk.Entry(window, width=8, textvariable=plot_id)
    input_id.place(relx=0.5, rely=0.7, anchor=CENTER)
    tk.Button(window, text="Plot speed", command=lambda: plot_speed()).place(relx=0.7, rely=0.7, anchor=CENTER)
