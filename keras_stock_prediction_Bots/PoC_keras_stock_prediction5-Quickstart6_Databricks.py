# Databricks notebook source
# MAGIC %md
# MAGIC # PoC
# MAGIC notwendige Dateien sollten im Verzeichnis liegen
# MAGIC - GBPUSD_M5_ab2019.01.21.csv     - OHLC-Candlestick Währungs-Wechselkurs Daten von USD gegen GBP
# MAGIC - GBPUSD_M5_ab2019.01.21.csv.gz  - gespeicherte Normalisierungswerte der CSV
# MAGIC - GBPUSD_M5_ab2019.01.21.csv.h5  - gespeichertes Neuronales-Netz (Tensorflow), um es nicht mit jedem Durchgang neu berechnen zu müssen
# MAGIC - GBPUSD_M5_ab2019.01.21.csv.png - gespeicherte Plot-Ausgabe des Backtradings zu Demozwecken, Anzeige nur im Markdown
# MAGIC - PoC_ keras_stock_prediction5-Quickstart6.py
# MAGIC
# MAGIC Config: 
# MAGIC - notwendige Lib (siehe unten)
# MAGIC   - Tensorflow 
# MAGIC   - Backtrader 
# MAGIC - Python 3.11.0rc1 ist ok
# MAGIC
# MAGIC GitHub: https://github.com/Howdi9/tf-stockprediction
# MAGIC
# MAGIC # Ablauf
# MAGIC Schritt 0: Config und Einrichtung
# MAGIC
# MAGIC Schritt 1: CSV (historischen 5min-Wechselkurse GBP-USD) trainiert ein Neuronales Netz via Keras-Tensorflow
# MAGIC
# MAGIC Schritt 2: NN wird mit identischer CSV verwendet um eine Tradingstrategie mit den zu testen.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC # Schritt 0: Config und Einrichtung
# MAGIC - %sh funktioniert plötzlich nicht mehr, obwohl es immer funktionierte
# MAGIC

# COMMAND ----------

#Welches OS habe ich? -> lief erfolgreich mit Ubuntu
%sh cat /etc/issue

# COMMAND ----------

#In welchem Verzeichnis bin ich? -> idealerweise ein eigenes Verzeichnis einrichten, dort dbc-file entpacken und dort arbeiten
%sh pwd

# COMMAND ----------

#Wer bin ich? ->root!
%sh id

# COMMAND ----------

#pip updaten: sollte unnötig sein, da alles von databricks uptodate gehalten sein sollte

!pip install --upgrade pip

# COMMAND ----------

# MAGIC %md
# MAGIC Installation von Backtrader
# MAGIC - https://www.backtrader.com/
# MAGIC - eigenes Fork https://github.com/Howdi9/backtrader
# MAGIC
# MAGIC python setup.py install

# COMMAND ----------

!pip install git+https://github.com/Howdi9/backtrader.git


# COMMAND ----------

!pip install tensorflow

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

!python --version

# COMMAND ----------

# MAGIC %md
# MAGIC # Schritt 1: keras_stock_prediction-5.py

# COMMAND ----------

# Folgende Parameter in diesem PoC bitte nicht ändern
# Anzahl der Datensetze (hier 5min Kerzen), die für Prediction verwendet werden sollen
DAYS_BEFORE = 100     

# Anzahl der Tage, die geforcasted werden sollen, 
DAYS_PREDICT = 2  
epochs=100
WorkWithSavedNN=True  # True: NN laden von "./GBPUSD_M5_ab2019.01.21.csv.h5"
                      # False: neues NN trainieren, dauerte 4.7h (mit Standard_DS3_V2, Runtimeversion 15.3)

CSV_PATH = "./"
CSV_FILE  = "GBPUSD_M5_ab2019.01.21.csv"

# COMMAND ----------

print("#######Import ...")
import tensorflow as tf
import numpy as np
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math
from tensorflow.keras.models import load_model, Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
#plt.matplotlib.use('TkAgg') (heute nicht mehr nötig? ist im originalCode nicht auskommentiert!!)
import joblib
print("#######Import activities finished")

# ausgeführt auf Databricks kommt "GPU will not be used.", soweit ausgeführt auf Cluster ohne GPU

# COMMAND ----------

print("---------------------------------")
print("#######CSV einlesen")

# Spalten GBPUSD
#    (0)Symbole,(1)TimeFrame,(2)Time,(3)Open,(4)High,(5)Low,(6)Close,(7)Volume
#    EURUSD,D1,2020.05.22 00:00:00,1.22207,1.22333,1.21611,1.21825,52575.0
#    EURCHF_M1

print("File: ", CSV_PATH+CSV_FILE)
initial_stock_data = np.loadtxt(
    CSV_PATH+CSV_FILE,delimiter=";",
    skiprows=1,
    #max_rows=1000,
    usecols=(6),
    comments="#",
    dtype=float,
    #encoding="UTF-16"
    )
print(initial_stock_data)

print("#######CSV einlesen finished")

# COMMAND ----------

print("---------------------------------")
print("#######CSV reshape(-1,1)")
initial_stock_data = np.array(initial_stock_data,dtype="float").reshape(-1,1)
print("#######CSV reshape finished")

# COMMAND ----------

#TensorBoard-callbacks
stockpred_callback = tf.keras.callbacks.TensorBoard(
    log_dir=".\\CallbackLogs"
#   histogram_freq=1
#   write_graph=True
#   write_images=True
#   update_freq=batch
#   profile_batch=
#   embeddings_freq=
#   embeddings_metadata=
    )

# COMMAND ----------

# Normalisierung der Werte
print("---------------------------------")
print("####### MinMaxScaler fit_transform ")
min_max_scaler = MinMaxScaler(feature_range=(0,1))
stock_data = min_max_scaler.fit_transform(initial_stock_data)
joblib.dump(min_max_scaler, './' + CSV_FILE + '.gz')

print("####### Speicherung der normalisierten np-Werte als Dump: ./" + CSV_FILE + ".gz")
print("####### MinMaxScaler fit_transform finished")
print("---------------------------------")

# COMMAND ----------

# Reorganisiert die Daten
def arrange_data(data, daysH, daysF):
    days_before_values = [] # T- days
    days_values = []  # T
    for i in range(len(data) - (daysH+daysF-1)):
       days_before_values.append(data[i:(i+daysH)])
       days_values.append(data[(i+daysH):(i+daysH+daysF)])
    return np.array(days_before_values),np.array(days_values)

days_before_values, days_values =  arrange_data(stock_data,DAYS_BEFORE, DAYS_PREDICT)


# COMMAND ----------

# Wir nehmen nur ein Teil des Datasets, um das Training durchzuführen
# Der Rest (X_test und Y_test) wird für die "virtuelle" Prognose benutzt 
# Splitting des Datasets
def split_to_percentage(data,percentage):                                                           # 1D
    return  data[0: int(len(data)*percentage)] , data[int(len(data)*percentage):]

X_train, X_test = split_to_percentage(days_before_values,0.8) #  80:20 Eingabedaten
Y_train, Y_test = split_to_percentage(days_values,0.8) # 80:20 Ausgabedaten


# COMMAND ----------

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import MeanSquaredError

if not WorkWithSavedNN: 
    # Definition des Keras Modells

    stock_model = Sequential()

    print("---------------------------------")
    print("#########add first LSTM")
    stock_model.add(LSTM(50,input_shape=(DAYS_BEFORE,1), return_sequences=True))     # stateful=True
    print("#########add LSTM finished")

    print("---------------------------------")
    print("#########add second LSTM")
    stock_model.add(LSTM(20,activation="tanh"))
    #stock_model.add(LSTM(5,activation="relu"))
    print("#########add LSTM finished")

    print("---------------------------------")
    print("#########add Dense")
    stock_model.add(Dense(DAYS_PREDICT))
    print("#########add Dense finished")

    sgd = SGD(learning_rate=0.01)

    #Model compiling passiert ohne Output
    stock_model.compile(loss="mean_squared_error", optimizer=sgd, metrics=[MeanSquaredError()])

    print("---------------------------------")
    print("#######Model fit")
    print("")
    batch_size=None
    stock_model.fit(X_train, Y_train, batch_size, epochs, verbose=2, callbacks=[stockpred_callback])
    print("")
    print("#######Model fit finished")
    print("---------------------------------")
    stock_model.summary()
    print("")

    # Das Modell wird gespeichert, passiert ohne Output
    print("#######Model save")
    stock_model.save('./' + CSV_FILE + ".h5")
    print("#######Model save finished")
else:
    # Das Modell wird geladen, passiert ohne Output
    print("#######Model load")
    stock_model= tf.keras.models.load_model('./' + CSV_FILE + "_Databricks.h5")
    print("#######Model load finished")

# COMMAND ----------

# Evaluation der Testdaten
print("---------------------------------")
print("#######Model evaluate")
print("")
score, _ = stock_model.evaluate(X_test,Y_test, verbose=2, callbacks=[stockpred_callback])
print("")
print("#######Model evaluate finished")
print("---------------------------------")

# COMMAND ----------

rmse = math.sqrt(score)
print("RMSE (RootMeanSquaredError): {}".format(rmse))


# Vorhersage mit den "unbekannten" Test-Dataset
predictions_on_test = stock_model.predict(X_test, verbose=2, callbacks=[stockpred_callback])
print("---------------------------------")
print('predictions_on_test: ', predictions_on_test)

predictions_on_test = min_max_scaler.inverse_transform(predictions_on_test)
print("---------------------------------")
print('predictions_on_test: ', predictions_on_test)

# COMMAND ----------

# ... und mit dem Trainings-Dataset
predictions_on_training = stock_model.predict(X_train, verbose=2, callbacks=[stockpred_callback])
print("---------------------------------")
print('predictions_on_training: ', predictions_on_training)

predictions_on_training = min_max_scaler.inverse_transform(predictions_on_training)
print("---------------------------------")
print('predictions_on_training: ', predictions_on_training)

# COMMAND ----------

Row_DAYS_PREDICT_Train_1=np.array(predictions_on_training[:,0])
Row_DAYS_PREDICT_Test_1=np.array(predictions_on_test[:,0])
Row_DAYS_PREDICT_Train_2=np.array(predictions_on_training[:,1])
Row_DAYS_PREDICT_Test_2=np.array(predictions_on_test[:,1])

# Wir shiften nach rechts, damit das Testergebnis grafisch direkt nach der Trainingskurve startet.
shift_1 = range(len(Row_DAYS_PREDICT_Train_1)-DAYS_PREDICT, len(stock_data) - 1 - (DAYS_BEFORE+DAYS_PREDICT) - 1)
shift_2 = range(len(Row_DAYS_PREDICT_Train_2)-DAYS_PREDICT, len(stock_data) - 1 - (DAYS_BEFORE+DAYS_PREDICT) - 1)

# COMMAND ----------


# Anzeige der Kurven mit matplotlib
print("---------------------------------")
print("pyplot Ausgabe: Fenster schließen zum Beenden des Python-Programms")
plt.plot(initial_stock_data, color="grey",label="Kurs", linewidth=2)
plt.plot(Row_DAYS_PREDICT_Train_1, label="Train Pred1", color="green", linewidth=1)
plt.plot(Row_DAYS_PREDICT_Train_2, label="Train Pred2", color="lightgreen", linewidth=1)

#plt.plot(shift_3, Row_DAYS_PREDICT_Test_1, label="Test Pred1", color="green", dashes=[6, 2])
#plt.plot(shift_2, Row_DAYS_PREDICT_Test_2, label="Test Pred2", color="lightgreen", dashes=[6, 2])

plt.legend(loc='upper left')
#plt.set_xlabel('Zeitachse')
#plt.set_ylabel('Kurs in USD')
#plt.set_title("Kursverlauf")
plt.show()
plt.savefig('.' + CSV_FILE + ".png") 

#(Grafik nur bedingt hilfreich)


# COMMAND ----------

# MAGIC %md
# MAGIC # Schritt 2: QuickstartSample6_Indicator.py

# COMMAND ----------

use_IB = False 
# True: Live-Verbindung zu InteractiveBroker für Backtesting (Laden Historischer Daten via Broker) oder LiveTrading
# False: Verwendung der CSV-Datei zum Datenladen (Nur Backtesting möglich)

Go_live = False
#True: Wenn use_IB=true kann hiermit Live-Trading gestartet werden
#False: Kein LiveTrading, nur Laden historischer Daten.

# CSV-Datei mit historischen Daten laden
CSV_PATH = "./"
CSV_FILE  = "GBPUSD_M5_ab2019.01.21.csv"

# COMMAND ----------

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Import the backtrader platform
import backtrader as bt
import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

print("#######Import ...")
import tensorflow as tf
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#import os
import math
from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, LSTM
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.optimizers import SGD

from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#matplotlib.use('TkAgg')

import joblib
print("#######Import activities finished")

# COMMAND ----------

class TensorflowPrediction(bt.Indicator):
    lines = ('tfpred',)
    params = (('period', 20),)
    
    def __init__(self):
        self.addminperiod(self.params.period)
        plotinfo = dict(subplot=False, plotforce=True)
        
    
    def next(self):
        lastdataclose1 = math.fsum(self.data.get(size=self.params.period))
        self.lines.tfpred[0] = lastdataclose1 / self.p.period                     #<<- funktioniert!

# COMMAND ----------

# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (('maperiod', 15), ('TFperiod', 20),)

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
                
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod)
        self.TF = TensorflowPrediction()

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])       
        
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:
    
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.dataclose[0] < self.sma[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()


    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

# COMMAND ----------

from backtrader.feeds import GenericCSVData

if __name__ == '__main__':

    if use_IB:
        if Go_live:
            print(" ")
            print('Beginn Go_live')
            dataLoad = bt.feeds.IBData(
                dataname='GBP.USD-CASH-IDEALPRO', 
                host='127.0.0.1', 
                port=7496, 
                timeframe=bt.TimeFrame.TFrame("Minutes"),
                rtbar=False,
                compression = 5
                )
            print('Ende Go_live')
        else:         
            print(' ')
            print('Beginn DataLoad historical')
            dataLoad = bt.feeds.IBData(
                dataname='GBP.USD-CASH-IDEALPRO', 
                host='127.0.0.1', 
                port=7496, 
                historical=True,
                timeframe=bt.TimeFrame.TFrame("Minutes"),
                compression = 5
                )
            print('Ende DataLoad historical')
    else: 
        print(" ")
        print('Beginn Datenladen CSV')
         
        # Create a Data Feed
        FILEPATH = CSV_PATH + CSV_FILE
        print ("FILEPATH: ", FILEPATH)
        dataLoad = GenericCSVData(
            dataname = CSV_PATH + CSV_FILE,
            separator = ";",
            dtformat = ('%d.%m.%Y %H:%M'),
            datetime = 2,
            time = -1,
            open = 3,
            high = 4,
            low = 5,
            close = 6,
            volume = 7,
            openinterest = -1,
            nullvalue = 0.0,
            header = True
        )
        print('Ende Datenladen CSV')
    

# COMMAND ----------

print(" ")
print('Beginn: Laden des Models')
stock_model= tf.keras.models.load_model("./GBPUSD_M5_ab2019.01.21.csv.h5")
print('Ende')
    
print(" ")
print('Beginn: Laden min_max_scaler')
min_max_scaler = joblib.load('./GBPUSD_M5_ab2019.01.21.csv.gz')
print("Ende")

        
# Create a cerebro entity
print(" ")
print('Beginn: bt.Cerebro()')
cerebro = bt.Cerebro()
print('Ende: bt.Cerebro()')
     
# Add a strategy
print(" ")
print('Beginn: cerebro.addstrategy(TestStrategy)')
cerebro.addstrategy(TestStrategy)
print('Ende')
    

# Add the Data Feed to Cerebro
print(" ")
print('Beginn: cerebro.adddata(data)')
cerebro.adddata(dataLoad)
print('Ende')

# COMMAND ----------

# Set our desired cash start
cerebro.broker.setcash(10000.0)

# Add a FixedSize sizer according to the stake
cerebro.addsizer(bt.sizers.FixedSize, stake=10)

# Set the commission
cerebro.broker.setcommission(commission=0.0)
#cerebro.setbroker(bt.brokers.IBBroker(**storekwargs))

# Print out the starting conditions
print(" ")
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# COMMAND ----------

# Run over everything
print(" ")
print("Beginn: cerebro.run()")
cerebro.run()
print("Ende run")
    
    
# Print out the final result
print(" ")
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Plot the result
cerebro.plot()

#Plotting funktioniert mit Databricks leider nicht!

# COMMAND ----------

# MAGIC %md
# MAGIC Beispiel-Plot
# MAGIC
# MAGIC ![Beispiel-Plot](./QuickstartSample5.png)
