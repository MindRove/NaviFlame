import math
from enum import Enum
import tensorflow as tf 
import socket
import time
from queue import Empty

class MyMagnWarping(tf.keras.layers.Layer):

    def __init__(self, sigma, divide, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.divide = 0.0
        self.sigma = 0.0

    def call(self, x):
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sigma": self.sigma,
            "divide": self.divide
        })
        return config

class MyScaling(tf.keras.layers.Layer):

    def __init__(self, sigma, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.sigma = 0.0

    def call(self, x):
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sigma": self.sigma
        })
        return config


class FilterTypes(Enum):
    bq_type_lowpass = 0
    bq_type_highpass = 1
    bq_type_bandpass = 2
    bq_type_notch = 3
    bq_type_peak = 4
    bq_type_lowshelf = 5
    bq_type_highshelf = 6

class BiquadMultiChan:
    def __init__(self, N, filter_type, Fc, Q, peakGainDB):
        self.Nchan = N
        self.type = filter_type
        self.Fc = Fc
        self.Q = Q
        self.peakGain = peakGainDB
        self.a0 = self.a1 = self.a2 = self.b1 = self.b2 = 0.0
        self.z1 = [0.0] * N
        self.z2 = [0.0] * N
        self.calc_biquad()

    def set_type(self, filter_type):
        self.type = filter_type
        self.calc_biquad()

    def set_Q(self, Q):
        self.Q = Q
        self.calc_biquad()

    def set_Fc(self, Fc):
        self.Fc = Fc
        self.calc_biquad()

    def set_peak_gain(self, peakGainDB):
        self.peakGain = peakGainDB
        self.calc_biquad()

    def set_biquad(self, filter_type, Fc, Q, peakGainDB):
        self.type = filter_type
        self.Q = Q
        self.Fc = Fc
        self.set_peak_gain(peakGainDB)

    def process(self, input_sample, Ichan):
        output = input_sample * self.a0 + self.z1[Ichan]
        self.z1[Ichan] = input_sample * self.a1 + self.z2[Ichan] - self.b1 * output
        self.z2[Ichan] = input_sample * self.a2 - self.b2 * output
        return output

    def calc_biquad(self):
        V = math.pow(10, abs(self.peakGain) / 20.0)
        K = math.tan(math.pi * self.Fc)
        norm = 0.0

        if self.type == FilterTypes.bq_type_lowpass:
            norm = 1 / (1 + K / self.Q + K * K)
            self.a0 = K * K * norm
            self.a1 = 2 * self.a0
            self.a2 = self.a0
            self.b1 = 2 * (K * K - 1) * norm
            self.b2 = (1 - K / self.Q + K * K) * norm

        elif self.type == FilterTypes.bq_type_highpass:
            norm = 1 / (1 + K / self.Q + K * K)
            self.a0 = 1 * norm
            self.a1 = -2 * self.a0
            self.a2 = self.a0
            self.b1 = 2 * (K * K - 1) * norm
            self.b2 = (1 - K / self.Q + K * K) * norm

        elif self.type == FilterTypes.bq_type_bandpass:
            norm = 1.0 / (1.0 + K / self.Q + K * K)
            self.a0 = (K / self.Q) * norm
            self.a1 = 0.0
            self.a2 = -self.a0
            self.b1 = 2.0 * (K * K - 1.0) * norm
            self.b2 = (1.0 - K / self.Q + K * K) * norm

        elif self.type == FilterTypes.bq_type_notch:
            norm = 1.0 / (1.0 + K / self.Q + K * K)
            self.a0 = (1 + K * K) * norm
            self.a1 = 2.0 * (K * K - 1) * norm
            self.a2 = self.a0
            self.b1 = self.a1
            self.b2 = (1.0 - K / self.Q + K * K) * norm

        elif self.type == FilterTypes.bq_type_peak:
            if self.peakGain >= 0:  # boost
                norm = 1 / (1 + (1 / self.Q) * K + K * K)
                self.a0 = (1 + (V / self.Q) * K + K * K) * norm
                self.a1 = 2 * (K * K - 1) * norm
                self.a2 = (1 - (V / self.Q) * K + K * K) * norm
                self.b1 = self.a1
                self.b2 = (1 - (1 / self.Q) * K + K * K) * norm
            else:  # cut
                norm = 1 / (1 + (V / self.Q) * K + K * K)
                self.a0 = (1 + (1 / self.Q) * K + K * K) * norm
                self.a1 = 2 * (K * K - 1) * norm
                self.a2 = (1 - (1 / self.Q) * K + K * K) * norm
                self.b1 = self.a1
                self.b2 = (1 - (V / self.Q) * K + K * K) * norm

        elif self.type == FilterTypes.bq_type_lowshelf:
            if self.peakGain >= 0:  # boost
                norm = 1 / (1 + math.sqrt(2) * K + K * K)
                self.a0 = (1 + math.sqrt(2 * V) * K + V * K * K) * norm
                self.a1 = 2 * (V * K * K - 1) * norm
                self.a2 = (1 - math.sqrt(2 * V) * K + V * K * K) * norm
                self.b1 = 2 * (K * K - 1) * norm
                self.b2 = (1 - math.sqrt(2) * K + K * K) * norm
            else:  # cut
                norm = 1 / (1 + math.sqrt(2 * V) * K + V * K * K)
                self.a0 = (1 + math.sqrt(2) * K + K * K) * norm
                self.a1 = 2 * (K * K - 1) * norm
                self.a2 = (1 - math.sqrt(2) * K + K * K) * norm
                self.b1 = 2 * (V * K * K - 1) * norm
                self.b2 = (1 - math.sqrt(2 * V) * K + V * K * K) * norm

        elif self.type == FilterTypes.bq_type_highshelf:
            if self.peakGain >= 0:  # boost
                norm = 1 / (1 + math.sqrt(2) * K + K * K)
                self.a0 = (V + math.sqrt(2 * V) * K + K * K) * norm
                self.a1 = 2 * (K * K - V) * norm
                self.a2 = (V - math.sqrt(2 * V) * K + K * K) * norm
                self.b1 = 2 * (K * K - 1) * norm
                self.b2 = (1 - math.sqrt(2) * K + K * K) * norm
            else:  # cut
                norm = 1 / (V + math.sqrt(2 * V) * K + K * K)
                self.a0 = (1 + math.sqrt(2) * K + K * K) * norm
                self.a1 = 2 * (K * K - 1) * norm
                self.a2 = (1 - math.sqrt(2) * K + K * K) * norm
                self.b1 = 2 * (K * K - V) * norm
                self.b2 = (V - math.sqrt(2 * V) * K + K * K) * norm
        else:
            raise ValueError("Invalid filter type")


def send_output_to_socket(stop_event, output_queue):
    while not stop_event.is_set(): 
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)  
                s.connect(('localhost', 8052))
                print("Connected to Visualizer successfully.")

                while not stop_event.is_set():
                    try:
                        output_value = output_queue.get(timeout=1)
                        s.sendall(int(output_value).to_bytes(4, byteorder='little'))
                    except Empty:
                        continue  
                    except BrokenPipeError:
                        print("Connection lost. Reconnecting...")
                        break  

        except ConnectionRefusedError:
            print("Connection refused. Ensure the Visualizer is running. Retrying in 40 seconds...")
            time.sleep(40)
        except KeyboardInterrupt:
            print("\nSocket communication interrupted. Closing connection...")
            break 

    print("Socket thread stopped.")
            