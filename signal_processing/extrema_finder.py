class ExtremaFinder:
    def __init__(self, signal):
        self.signal = signal
        self.maximum = []
        self.minimum = []
        self.find_extrema()

    def find_extrema(self):
        for i in range(1, len(self.signal) - 1):
            if self.signal[i] > self.signal[i - 1] and self.signal[i] > self.signal[i + 1]:
                self.maximum.append([i, self.signal[i], 0])  # 极大值位置，值，初始状态0
            if self.signal[i] < self.signal[i - 1] and self.signal[i] < self.signal[i + 1]:
                self.minimum.append([i, self.signal[i], 0])  # 极小值位置，值，初始状态0
