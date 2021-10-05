import datetime
import pickle
from tkinter import CENTER

import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import time
import tkinter as tk
import csv


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def detect_outlier_IQR(data):
    Q1 = np.quantile(data, 0.25)
    Q3 = np.quantile(data, 0.75)
    IQR = Q3 - Q1
    data_out = data[~(data > (Q3 + 1.5 * IQR))]
    return data_out


def track_data(params, coordinates, ped_up_list):
    # ##### CROSSWALK ###########
    Pedestrian_ID = []
    a = params.split('.avi')
    output = a[0]
    with open(output + ".pickle", 'rb') as handle:
        frame_list = pickle.load(handle)

    print(len(ped_up_list))

    maximum = 0
    for i in range(len(ped_up_list)):
        if ped_up_list[i][0] > maximum:
            maximum = ped_up_list[i][0]
        else:
            continue

    no_persons = maximum
    print("Number of assigned ids = ", no_persons)

    id_path = defaultdict(list)

    id_list = []
    for item in ped_up_list:
        if item[0] not in id_list:
            id_list.append(item[0])
    print(id_list)

    # item[0] --> ID
    # item[1][0] --> X
    # item[1][1] --> Y
    # item[1] -->

    # FOR UP
    x_0 = int(float(coordinates[0]))
    x_1 = int(float(coordinates[1]))
    x_2 = int(float(coordinates[2]))
    x_3 = int(float(coordinates[3]))
    y_0 = int(float(coordinates[4]))
    y_1 = int(float(coordinates[5]))
    y_2 = int(float(coordinates[6]))
    y_3 = int(float(coordinates[7]))

    for item in ped_up_list:
        for id in id_list:
            if id == item[0] and (not item[1] in list(id_path[id])):
                X = item[1][0]
                Y = item[1][1]
                # Limit the amount of computation, by only taking orthogonality of lines within the min and max values
                if (min(x_0, x_1, x_2, x_3) <= item[1][0] <= max(x_0, x_1, x_2, x_3)) \
                        and (min(y_0, y_1, y_2, y_3) <= item[1][1] <= max(y_0, y_1, y_2, y_3)):
                    c = (int(X) - x_1) * (y_2 - y_1)
                    d = (int(Y) - y_1) * (x_2 - x_1)
                    side1 = c - d
                    m = (int(X) - x_0) * (y_3 - y_0)
                    n = (int(Y) - y_0) * (x_3 - x_0)
                    side2 = m - n
                    # if the id's position is between the ped_up_list region,
                    # then there will be one positive and one negative orthogonal statement
                    if (side1 >= 0 and side2 <= 0) or (side2 >= 0 and side1 <= 0):
                        a = (int(X) - x_0) * (y_1 - y_0)
                        b = (int(Y) - y_0) * (x_1 - x_0)
                        enter1 = a - b
                        q = (int(X) - x_3) * (y_2 - y_3)
                        p = (int(Y) - y_3) * (x_2 - x_3)
                        enter2 = q - p
                        if (enter1 >= 0 and enter2 <= 0) or (enter2 >= 0 and enter1 <= 0):
                            id_path[id].append(item[1])
            else:
                continue

    ############## COMMON CODE PARTS ##############

    def get_cmap(n, name='hsv'):
        return plt.cm.get_cmap(name, n)

    n = len(id_path)
    print("Number of detected persons = ", len(id_path))
    # time.sleep(5)

    cmap = get_cmap(n)
    count = 0
    speed_list = []
    time_test = []
    j = 0  # for different color plots
    # key is subject ID
    # value is location
    # items is key and values in the dictionary together
    for key, value in id_path.items():
        j += 1  # for different color plots

        x_pixel = [pt[0] for pt in value]
        y_pixel = [pt[1] for pt in value]

        # 0.15 if factor to convert pixels to m/s

        x = [element * 0.15 for element in x_pixel]
        y = [element * 0.15 for element in y_pixel]

        # x squared and y squared
        x2 = [a * b for a, b in zip(x, x)]
        y2 = [a * b for a, b in zip(y, y)]

        # numpy array
        s = np.sqrt([x + y for x, y in zip(x2, y2)])
        t = np.linspace(0, 100, len(s))

        if len(s) <= 24:
            continue

        print("case ID is : " + str(key) + " and walking duration is : " + str(len(s) / 12))

        Pedestrian_ID.append(str(key))

        speed = (abs(np.diff(s))) * 12  # 12 : Frame Rate
        MAstep = (len(speed) // 5)
        speed_smooth = moving_average(speed, MAstep)
        t_speed = np.linspace(0, 100, len(speed_smooth))

        average_speed = np.median(speed) if np.mean(speed) > np.median(speed) else np.mean(speed)
        print(average_speed)
        print("\n")
        speed_list.append(average_speed)
        time_test.append(len(s))

        # plt.plot(speed_smooth)
        # plt.show()

        # fig, axs = plt.subplots(2, 2)
        # fig.suptitle("Pedestrian ID N0. " + "\"" + str(key) + "\"" + " On Northern Crosswalk")

        # axs[0, 0].axis(ymin=min(x)-5, ymax=max(x)+5)
        # axs[0, 0].plot(t,x, c=cmap(j), lw=2)
        # axs[0, 0].set_title("X Coordinate of Pedestrian Path")
        # axs[0, 0].set_xlabel('Percentage of Walking Duration')
        # axs[0, 0].set_ylabel('Coordinate (m)')

        # axs[1, 0].axis(ymin=min(y)-5, ymax=max(y)+5)
        # axs[1, 0].plot(t, y, c=cmap(j), lw=2)
        # axs[1, 0].set_title("Y Coordinate of Pedestrian Path")
        # axs[1, 0].set_xlabel('Percentage of Walking Duration')
        # axs[1, 0].set_ylabel('Coordinate (m)')

        # axs[0, 1].axis(xmin=min(x)-5, xmax=max(x)+5)
        # axs[0, 1].axis(ymin=min(y)-5, ymax=max(y)+5)
        # axs[0, 1].invert_yaxis()
        # axs[0, 1].scatter(x, y, c=cmap(j), marker='o')
        # axs[0, 1].set_title("Walking Path (X vs. Y)")
        # axs[0, 1].set_xlabel('Coordinate (m)')
        # axs[0, 1].set_ylabel('Coordinate (m)')

        # if max(speed) >= 5:
        #    speed_limit = max(speed)
        # else:
        #    speed_limit = 5
        # axs[1, 1].axis(ymin=0, ymax=speed_limit)
        # axs[1, 1].plot(t_speed, speed_smooth, c=cmap(j), lw=2, label="Mean Speed = " + str(round(average_speed,2)) + " m/s")
        # axs[1, 1].legend()
        # axs[1, 1].set_title("Walking Speed")
        # axs[1, 1].set_xlabel('Percentage of Walking Duration')
        # axs[1, 1].set_ylabel('Speed (m/s)')

        # fig.tight_layout()
        # fig.set_size_inches(10, 10)
        # plt.show()
        # plt.savefig('outputs/Light/Segment3/Screenshots/Light_0001_Sidewalk_{}.png'.format(key))
        # plt.close()

    print(speed_list)

    with open(output + '.csv', 'w', newline='') as result_file:
        wr = csv.writer(result_file)
        # header
        header = ['Engineering Health Team YOLOv4 pedestrian detection']
        wr.writerow(header)
        test = datetime.date.today()
        a = test.strftime("%m-%d-%Y")
        wr.writerow(['Date', a])
        # template used
        template = ['Template used:']
        wr.writerow(template)
        x = coordinates[0:3]
        y = coordinates[4:7]
        wr.writerow(['X', x_0, x_1, x_2, x_3])
        wr.writerow(['Y', y_0, y_1, y_2, y_3])
        # speed list
        # wr.writerow(['List of speeds'])
        # for item in speed_list:
        #     wr.writerow([item])
        # pedestrian id w speed
        wr.writerow(['Pedestrian ID', 'Speed (m/s)', 'Start Frame', 'Last Frame'])
        for i in range(len(speed_list)):
            start, final = get_timestamps(Pedestrian_ID[i], frame_list)
            wr.writerow([Pedestrian_ID[i], speed_list[i], start, final])

    window = tk.Tk()
    window.title('Analysis complete')
    win_width = 640
    height = 480
    geometry = str(win_width) + "x" + str(height)
    window.geometry(geometry)

    done_title = tk.Label(window, text='Pedestrian Detection has finished!', fg="green")
    done_title.config(font=('Arial', 20))
    done_title.place(relx=0.5, rely=0.1, anchor=CENTER)

    tk.Label(window, text='Check ' + output + ' for results').place(relx=0.5, rely=0.5, anchor=CENTER)
    window.mainloop()


def get_timestamps(ped_id, frame_list):
    counter = 0
    start_time = 0
    final_time = 0
    for row in frame_list:
        if int(row[1]) == int(ped_id):
            # first iteration, set final and start times are there would never be a zero
            if counter == 0:
                start_time = int(row[0])
                final_time = int(row[0])
            # if the new row has a smaller value than start time, change it
            elif start_time > int(row[0]):
                start_time = int(row[0])
            # if the new row's frame # is larger, change the final time
            elif final_time < int(row[0]):
                final_time = int(row[0])
            else:
                print("END")
                continue
            counter += 1

    print("csv counter = ", counter)

    return start_time, final_time
