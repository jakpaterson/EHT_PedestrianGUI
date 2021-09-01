"""
WEATHER REPORT ANALYSIS SYSTEM
"""
import csv
import os
import tkinter as tk
import time
import webbrowser
from datetime import datetime, timedelta
from tkinter import HORIZONTAL, CENTER, VERTICAL, Y, RIGHT, FALSE, BOTH, END, N
import tkinter.filedialog as fd

# Initialize GUI window
import matplotlib.pyplot as plt
import pandas as pd

weather_frame = ""
year = ''
month = ''
day = ''


# Weather report
def weather_gui():
    global weather_frame
    global year
    global month
    global day
    year = tk.StringVar()
    month = tk.StringVar()
    day = tk.StringVar()
    weather_frame = tk.Toplevel()
    weather_frame.title('YOLOv4 Weather Reports')
    weather_width = 640
    weather_height = 280
    w_geometry = str(weather_width) + "x" + str(weather_height)
    weather_frame.geometry(w_geometry)
    # weather_frame.config(bg="#6D2C9E")
    weather_frame.resizable(False, False)
    # weather report
    tk.Label(weather_frame, text='Plot Weather Report', font=('Helvetica', 12, 'bold')).place(relx=0.5,
                                                                                              rely=0.1, anchor=CENTER)
    tk.Label(weather_frame, text='Download the csv from the year of this video',
             font=('Helvetica', 11)).place(relx=0.3, rely=0.2, anchor=CENTER)
    link = tk.Label(weather_frame, text="Toronto City Weather Station", fg="blue", cursor="hand2",
                    font='helvetica 11 underline')
    link.place(relx=0.8, rely=0.2, anchor=CENTER)
    link.bind("<Button-1>",
              lambda e: webbrowser.open_new("https://climate.weather.gc.ca/climate_data/daily_data_e.html?"
                                            "hlyRange=2002-06-04%7C2021-06-07&dlyRange=2002-06-04%7C2021-06"
                                            "-07&mlyRange=2003-07-01%7C2006-12-01&StationID=31688&Prov=ON"
                                            "&urlExtension=_e.html&searchType=stnProx&optLimit=yearRange"
                                            "&StartYear=2010&EndYear=2021&selRowPerPage=25&Line=3&txtRadius"
                                            "=25&optProxType=city&selCity=43%7C39%7C79%7C23%7CToronto"
                                            "&selPark=&txtCentralLatDeg=&txtCentralLatMin=0&txtCentralLatSec="
                                            "0&txtCentralLongDeg=&txtCentralLongMin=0&txtCentralLongSec=0"
                                            "&txtLatDecDeg=&txtLongDecDeg=&timeframe=2&Day=7&Year=2011"
                                            "&Month=9#"))

    tk.Label(weather_frame, text="Year").place(relx=0.2, rely=0.4, anchor=CENTER)
    year_input = tk.Entry(weather_frame, width=8, textvariable=year)
    year_input.place(relx=0.3, rely=0.4, anchor=CENTER)
    year_input.bind('<FocusOut>', validateYear)
    tk.Label(weather_frame, text="Month").place(relx=0.4, rely=0.4, anchor=CENTER)
    month_input = tk.Entry(weather_frame, width=6, textvariable=month)
    month_input.place(relx=0.5, rely=0.4, anchor=CENTER)
    month_input.bind('<FocusOut>', validateMonth)
    tk.Label(weather_frame, text="Day").place(relx=0.6, rely=0.4, anchor=CENTER)
    day_input = tk.Entry(weather_frame, width=6, textvariable=day)
    day_input.place(relx=0.7, rely=0.4, anchor=CENTER)
    day_input.bind('<FocusOut>', validateDay)

    tk.Button(weather_frame, text="Upload .csv file",
              command=lambda: select_csv()).place(relx=0.3, rely=0.6, anchor=CENTER)


# uploads csv for weather report
def select_csv():
    # open directory to select csv files only
    file_path = fd.askopenfile(mode='r', filetypes=[('csv Files', '*csv')])
    # check if it ws selected
    if file_path is not None:
        # label to display the file name
        if file_path.name != "":
            tk.Label(weather_frame, text=".csv Selected --> ").place(relx=0.6, rely=0.9, anchor=CENTER)
        tk.Button(weather_frame, text="Fetch 7-day Weather Report",
                  command=lambda: run_weather(file_path)).place(relx=0.8, rely=0.9, anchor=CENTER)
        return file_path.name


# plots weather data for weather report given date inputted
def run_weather(csv_file):
    df = pd.read_csv(csv_file.name)
    # import entry variables and convert to datetime obj str without the time
    global year
    global month
    global day
    if (int(month.get()) == 1 and int(day.get()) <= 7) or (int(month.get()) == 2 and int(day.get()) > 28):
        invalid = tk.Toplevel()
        geo = '240x240'
        invalid.title('ERROR')
        invalid.geometry(geo)
        tk.Label(invalid, text="Invalid Date!").place(relx=0.5, rely=0.3, anchor=CENTER)
        tk.Button(invalid, text="Close", command=lambda: invalid.destroy()) \
            .place(relx=0.5, rely=0.9, anchor=CENTER)
    else:
        input_date = str(year.get()) + '-' + str(month.get()) + '-' + str(day.get())
        date_obj = datetime.strptime(input_date, '%Y-%m-%d')
        day1 = date_obj.strftime("%Y-%m-%d")
        week_later = (date_obj - timedelta(days=6)).strftime("%Y-%m-%d")
        # index of the first and last days
        first_day = df.loc[(df["Date/Time"] == week_later)].index.values
        last_day = df.loc[(df["Date/Time"] == day1)].index.values

        fig, ax1 = plt.subplots()
        plt.title('7 Day Record of Weather and Precipitation before ' + day1)
        color = 'tab:red'
        plt.plot()
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Mean Temp (°C)', color=color)
        # row 4 is the date, row 13 is the mean temperature
        ax1.plot(df.iloc[int(first_day):int(last_day) + 1, 4], df.iloc[int(first_day):int(last_day) + 1, 13],
                 color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Precipitation (mm)', color=color)  # we already handled the x-label with ax1
        precipitation = []
        # precipitation can be a missed value, therefore place 0 where there was an M
        for drop in df.iloc[int(first_day):int(last_day) + 1, 23]:
            if drop == "M":
                precipitation.append(0)
            else:
                precipitation.append(drop)
        ax2.plot(df.iloc[int(first_day):int(last_day) + 1, 4], precipitation, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


def validateYear(event):
    global year
    now = datetime.now()
    if year.get().isdigit():
        year_int = int(year.get())
        if 1966 < year_int <= int(now.year):
            print("valid year")
        else:
            invalid = tk.Toplevel()
            geo = '240x240'
            invalid.title('ERROR')
            invalid.geometry(geo)
            tk.Label(invalid, text="Invalid Year!").place(relx=0.5, rely=0.3, anchor=CENTER)
            tk.Label(invalid, text="Must be between 1966 and " + str(now.year)).place(relx=0.5, rely=0.7, anchor=CENTER)
            tk.Button(invalid, text="Close", command=lambda: invalid.destroy()) \
                .place(relx=0.5, rely=0.9, anchor=CENTER)
    else:
        invalid = tk.Toplevel()
        geo = '240x240'
        invalid.title('ERROR')
        invalid.geometry(geo)
        tk.Label(invalid, text="Invalid Year!").place(relx=0.5, rely=0.3, anchor=CENTER)
        tk.Label(invalid, text="Must be between 1966 and " + str(now.year)).place(relx=0.5, rely=0.7, anchor=CENTER)
        tk.Button(invalid, text="Close", command=lambda: invalid.destroy()) \
            .place(relx=0.5, rely=0.9, anchor=CENTER)


def validateMonth(event):
    global month
    if month.get().isdigit():
        month_int = int(month.get())
        if 1 <= month_int <= 12:
            if len(month.get()) == 2:
                print("with zero")
            else:
                add_zero = "0" + month.get()
                month.set(add_zero)
        else:
            invalid = tk.Toplevel()
            geo = '240x240'
            invalid.title('ERROR')
            invalid.geometry(geo)
            tk.Label(invalid, text="Invalid Month!").place(relx=0.5, rely=0.3, anchor=CENTER)
            tk.Label(invalid, text="Must be between 01 and 12").place(relx=0.5, rely=0.7, anchor=CENTER)
            tk.Button(invalid, text="Close", command=lambda: invalid.destroy()) \
                .place(relx=0.5, rely=0.9, anchor=CENTER)
    else:
        invalid = tk.Toplevel()
        geo = '240x240'
        invalid.title('ERROR')
        invalid.geometry(geo)
        tk.Label(invalid, text="Invalid Month!").place(relx=0.5, rely=0.3, anchor=CENTER)
        tk.Label(invalid, text="Must be between 01 and 12").place(relx=0.5, rely=0.7, anchor=CENTER)
        tk.Button(invalid, text="Close", command=lambda: invalid.destroy()) \
            .place(relx=0.5, rely=0.9, anchor=CENTER)


def validateDay(event):
    global day
    if day.get().isdigit():
        day_int = int(day.get())
        if 1 <= day_int <= 31:
            if len(day.get()) == 2:
                print("with zero")
            else:
                add_zero = "0" + day.get()
                day.set(add_zero)
        else:
            invalid = tk.Toplevel()
            geo = '240x240'
            invalid.title('ERROR')
            invalid.geometry(geo)
            tk.Label(invalid, text="Invalid Day!").place(relx=0.5, rely=0.3, anchor=CENTER)
            tk.Label(invalid, text="Must be between 01 and 31").place(relx=0.5, rely=0.7, anchor=CENTER)
            tk.Button(invalid, text="Close", command=lambda: invalid.destroy()) \
                .place(relx=0.5, rely=0.9, anchor=CENTER)
    else:
        invalid = tk.Toplevel()
        geo = '240x240'
        invalid.title('ERROR')
        invalid.geometry(geo)
        tk.Label(invalid, text="Invalid Day!").place(relx=0.5, rely=0.3, anchor=CENTER)
        tk.Label(invalid, text="Must be between 01 and 31").place(relx=0.5, rely=0.7, anchor=CENTER)
        tk.Button(invalid, text="Close", command=lambda: invalid.destroy()) \
            .place(relx=0.5, rely=0.9, anchor=CENTER)
