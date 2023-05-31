import csv
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from PIL import ImageTk, Image
import tkinter as tk


def pickarea(zp):
    minmax = [0, 0]
    with open("data/EMAG2_V3.csv", 'r') as file:
        csvreader = csv.reader(file)
        with open('results/single_area.csv', 'w', newline='') as file1:
            details = ['Longitude', 'Latitude', 'AnomalyValue']
            writer = csv.writer(file1)
            writer.writerow(details)
            for row in csvreader:
                if int(row[6]) == zp:
                    writer.writerow([row[2], row[3], row[5]])
                    if float(row[5]) < minmax[0]:
                        minmax[0] = float(row[5])
                    if float(row[5]) > minmax[1]:
                        minmax[1] = float(row[5])
    return minmax


def forestmeans():
    df = pd.read_csv('results/single_area.csv')
    model = IsolationForest(n_estimators=50, contamination=float(0.05))
    model.fit(df[['AnomalyValue']].values)
    km = np.nan_to_num(df.values)
    k_means = KMeans(init="k-means++", n_clusters=8, n_init=12)
    k_means.fit(km)
    df['AnomalyScore'] = model.decision_function(df[['AnomalyValue']].values)
    df['AnomalyTF'] = model.predict(df[['AnomalyValue']].values)
    df['Cluster'] = k_means.labels_
    dfc = []
    labels = k_means.labels_
    centers = k_means.cluster_centers_
    nrows = df.shape[0]
    clustercnt = np.zeros(8, dtype=np.longlong)
    for i in range(nrows):
        clustercnt[labels[i]] = clustercnt[labels[i]] + 1
    for i in range(nrows):
        dfc.append(sqrt((df.iat[i, 0]-centers[labels[i], 0])**2+(df.iat[i, 1]-centers[labels[i], 1])**2+(df.iat[i, 2]-centers[labels[i], 2])**2))
    df['Distance_fr_Center'] = dfc
    dfc.sort()
    median = dfc[round(nrows/2)]
    abd = []
    for i in range(nrows):
        abd.append(abs(dfc[i] - median))
    abd.sort()
    mad = abd[round(nrows/2)]
    mzs = []
    for i in range(nrows):
        mzs.append(0.6745*(df.iat[i, 6]-median)/mad)
    df['MZScore'] = mzs
    ans = []
    thresh = nrows / 100
    for i in range(nrows):
        flag = 0
        if (df.iat[i, 7] < -3) or (df.iat[i, 7] > 3) or (clustercnt[df.iat[i, 5]] < thresh):
            flag = 1
        if (df.iat[i, 4] == -1) and (flag == 1):
            ans.append(-1)
        else:
            ans.append(1)
    df['Answer'] = ans
    df.to_csv("results/NLZD_data.csv", encoding='utf-8', index=False)


def nlzdgraphik():
    x_data = []
    y_data = []
    z_data = []
    xa_data = []
    ya_data = []
    za_data = []

    with open("results/NLZD_data.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if row[0] == 'Longitude':
                continue

            if row[8] == '1':
                x_data.append(float(row[0]))
                y_data.append(float(row[1]))
                z_data.append(float(row[2]))

            if row[8] == '-1':
                xa_data.append(float(row[0]))
                ya_data.append(float(row[1]))
                za_data.append(float(row[2]))

        ax = plt.axes(projection="3d")
        ax.scatter(x_data, y_data, z_data, color='blue')
        ax.scatter(xa_data, ya_data, za_data, color='red')
        ax.set_xlabel("Долгота")
        ax.set_ylabel("Широта")
        plt.savefig("results/analyzed_data.png")
        plt.close()
        plt.cla()
        plt.clf()


def rawgraphik(minmaxv):
    dat = np.zeros((2, round(minmaxv[1])), dtype=np.longlong)
    x_data = []
    y_data = []
    z_data = []

    for i in range(round(minmaxv[1])):
        dat[0, i] = i - minmaxv[1] // 2

    with open("results/single_area.csv", 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            if row[0] == 'Longitude':
                continue
            x_data.append(float(row[0]))
            y_data.append(float(row[1]))
            z_data.append(float(row[2]))
            c = round(float(row[2]) + minmaxv[1] // 2)
            if c > (minmaxv[1] - 1):
                c = round(minmaxv[1] - 1)
            if c < 0:
                c = 0
            dat[1, c] = dat[1, c] + 1
    plt.plot(dat[0], dat[1])
    plt.xlabel("Магнитная аномалия, нТл")
    plt.ylabel("Количество вхождений")
    plt.savefig("results/raw_values_distribution.png")
    plt.close()
    plt.cla()
    plt.clf()
    ax = plt.axes(projection="3d")
    sc = ax.scatter(x_data, y_data, z_data, c=z_data, vmin=minmaxv[0], vmax=minmaxv[1], cmap="plasma")
    plt.colorbar(sc)
    ax.set_xlabel("Долгота")
    ax.set_ylabel("Широта")
    plt.savefig("results/raw_data.png")
    plt.close()
    plt.cla()
    plt.clf()


def areabttn():
    zone = int(area_tf.get())
    if (zone < 1) or (zone > 121) or ((zone > 38) and (zone < 101)):
        windowtmp = tk.Toplevel(startwindow)
        windowtmp.title("Ошибка")
        windowtmp.geometry('300x50')
        windowtmp.configure(bg='white')
        err_lb = tk.Label(windowtmp, text="Пожалуйста, введите корректный регион")
        err_lb.grid(row=1, column=1)
        err_lb.configure(bg='white')
        err2_lb = tk.Label(windowtmp, text="!!!")
        err2_lb.grid(row=2, column=1)
        err2_lb.configure(bg='red')
        windowtmp.grab_set()
        windowtmp.update()
    else:
        windowtmp = tk.Toplevel(startwindow)
        windowtmp.title("...")
        windowtmp.geometry('180x50')
        windowtmp.configure(bg='white')
        msg_lb = tk.Label(windowtmp, text="Проводим вычисления")
        msg_lb.grid(row=1, column=1)
        msg_lb.configure(bg='white')
        msg2_lb = tk.Label(windowtmp, text="...")
        msg2_lb.grid(row=2, column=1)
        msg2_lb.configure(bg='green')
        windowtmp.grab_set()
        windowtmp.update()

        minmax = pickarea(zone)
        rawgraphik(minmax)

        windowtmp.destroy()
        windowtmp.update()

        window1 = tk.Toplevel(startwindow)
        window1.title("Исходные данные")
        window1.geometry('1280x720')
        window1.configure(bg='white')
        window1.grab_set()
        image1 = Image.open(r"results/raw_values_distribution.png")
        r_v = ImageTk.PhotoImage(image1)
        label1 = tk.Label(window1, image=r_v)
        label1.image = r_v
        label1.grid(row=1, column=1)
        image2 = Image.open(r"results/raw_data.png")
        r_d = ImageTk.PhotoImage(image2)
        label2 = tk.Label(window1, image=r_d)
        label2.image = r_d
        label2.grid(row=1, column=2)
        label22_lb = tk.Label(window1, text="Магнитная аномалия, нТл:")
        label22_lb.grid(row=1, column=2, sticky=tk.NW, padx=50, pady=50)
        label22_lb.configure(bg='white')
        nlz_lb = tk.Label(window1, text="Провести анализ ансамблем алгоритмов (изолирующий лес+k-means++):")
        nlz_lb.grid(row=2, column=1, pady=50)
        nlz_lb.configure(bg='white')
        nlz_btn = tk.Button(window1, text='Ок', command=nlzbutton)
        nlz_btn.grid(row=2, column=2, sticky=tk.W, pady=50)


def nlzbutton():
    windowtmp = tk.Toplevel(startwindow)
    windowtmp.title("...")
    windowtmp.geometry('180x50')
    windowtmp.configure(bg='white')
    msg_lb = tk.Label(windowtmp, text="Проводим вычисления")
    msg_lb.grid(row=1, column=1)
    msg_lb.configure(bg='white')
    msg2_lb = tk.Label(windowtmp, text="...")
    msg2_lb.grid(row=2, column=1)
    msg2_lb.configure(bg='green')
    windowtmp.grab_set()
    windowtmp.update()

    window2 = tk.Toplevel(startwindow)
    window2.title("Данные после анализа")
    window2.geometry('720x720')
    window2.configure(bg='white')
    window2.grab_set()
    forestmeans()
    nlzdgraphik()
    windowtmp.destroy()
    windowtmp.update()
    image1 = Image.open(r"results/analyzed_data.png")
    a_d = ImageTk.PhotoImage(image1)
    label1 = tk.Label(window2, image=a_d)
    label1.image = a_d
    label1.grid(row=1, column=1)
    nlzd_lb = tk.Label(window2, text="После анализа красным выделены аномальные вхождения данных, синим - нормальные.\nС полной базой данных после анализа можно ознакомиться по адресу results/NLZD_data.csv")
    nlzd_lb.grid(row=2, column=1, pady=50)
    nlzd_lb.configure(bg='white')


startwindow = tk.Tk()
startwindow.title("Поиск аномалий")
startwindow.geometry('720x720')
startwindow.configure(bg='white')

frame = tk.Frame(startwindow, padx=10, pady=10)
frame.pack(expand=True)
frame.configure(bg='white')

area_lb = tk.Label(frame, text="Введите номер региона для анализа (1-38; 101-121):")
area_lb.grid(row=1, column=1)
area_lb.configure(bg='white')
area_tf = tk.Entry(frame, )
area_tf.grid(row=1, column=2)
area_btn = tk.Button(frame, text='Ок', command=areabttn)
area_btn.grid(row=2, column=2, pady=50)

startwindow.mainloop()
