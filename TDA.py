import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

class TDA:
    def __init__(self, signal):
        self.signal = signal
        self.maximum = []  # 极大值列表
        self.minimum = []  # 极小值列表
        self.find_extrema()
        self.find_start_point()

    def process_signal(self):
        print("Processing signal...")
        print("Signal Length:", len(self.signal))
        print("Max Amplitude:", max(self.signal))
        print("Min Amplitude:", min(self.signal))

    def find_extrema(self):
        i = 1
        while i < len(self.signal) - 1:
            if self.signal[i] > self.signal[i - 1] and self.signal[i] > self.signal[i + 1]:
                self.maximum.append([i, self.signal[i], 0])
            elif self.signal[i] < self.signal[i - 1] and self.signal[i] < self.signal[i + 1]:
                self.minimum.append([i, self.signal[i], 0])
            elif self.signal[i] == self.signal[i - 1] and self.signal[i] == self.signal[i + 1]:
                start = i
                while i < len(self.signal) - 1 and self.signal[i] == self.signal[i + 1]:
                    i += 1
                end = i
                mid = (start + end) // 2
                if (start > 0 and self.signal[start] <= self.signal[start - 1] and
                    end < len(self.signal) - 1 and self.signal[end] <= self.signal[end + 1]):
                    if not self.minimum or self.minimum[-1][0] != mid:  # Prevent duplicates
                        self.minimum.append([mid, self.signal[mid], 0])
                elif (start > 0 and self.signal[start] >= self.signal[start - 1] and
                    end < len(self.signal) - 1 and self.signal[end] >= self.signal[end + 1]):
                    if not self.maximum or self.maximum[-1][0] != mid:  # Prevent duplicates
                        self.maximum.append([mid, self.signal[mid], 0])
                i = end  # Move to the end of the flat region
            i += 1


    def find_start_point(self):
        if len(self.maximum) < 3:
            return False  # 如果极大值点少于3个，则无法比较，直接返回

        new_start_found = False
        for i in range(1, len(self.maximum) - 1):
            # 找到第三维度不为-1的前一个极大值点
            prev_index = self.find_valid_extremum(i, -1)
            # 找到第三维度不为-1的后一个极大值点
            next_index = self.find_valid_extremum(i, 1)

            if prev_index is None or next_index is None:
                continue

            prev = self.maximum[prev_index]
            current = self.maximum[i]
            next = self.maximum[next_index]

            # 确保当前点的第三维度不为-1，并且当前点的幅度大于前后两点的幅度
            if current[2] != -1 and current[1] > prev[1] and current[1] > next[1]:
                current[2] = 1  # 将状态设置为1表示为起始点
                new_start_found = True

        return new_start_found

    def find_valid_extremum(self, index, step):
        """根据步长寻找有效的极大值点，步长为-1时向前查找，为1时向后查找"""
        test_index = index + step
        while 0 <= test_index < len(self.maximum):
            if self.maximum[test_index][2] != -1:
                return test_index
            test_index += step
        return None

    def iterate_algorithm(self):
        for minima in self.minimum:
            if minima[2] == 0:
                next_max = self.find_next_maximum(minima[1])
                if next_max:
                    minima[1] = next_max[0]

        for maxima in self.maximum:
            if maxima[2] == 1:
                nearest_min_left = self.find_nearest_minimum(maxima[0], -1)
                nearest_min_right = self.find_nearest_minimum(maxima[0], 1)

                if nearest_min_left and nearest_min_right:
                    if nearest_min_left[2] == 1 or nearest_min_right[2] == 1:
                        nearest_min_left[2] = nearest_min_right[2] = 1
                    else:
                        d1 = abs(nearest_min_left[1] - nearest_min_left[0])
                        d2 = abs(nearest_min_right[1] - nearest_min_right[0])
                        if d1 < d2:
                            nearest_min_left[2] = 1
                        else:
                            nearest_min_right[2] = 1

                    maxima[2] = -1  # 标记此起始点为已处理

    def find_next_maximum(self, start_index):
        for maxima in self.maximum:
            if maxima[0] > start_index and maxima[2] != -1:
                return maxima
        return None

    def find_nearest_minimum(self, start_index, direction):
        if direction == -1:
            return max((minima for minima in self.minimum if minima[0] < start_index), default=None,
                       key=lambda x: abs(x[0] - start_index))
        elif direction == 1:
            return min((minima for minima in self.minimum if minima[0] > start_index), default=None,
                       key=lambda x: abs(x[0] - start_index))

    def multiple_iterations(self, num_iterations=10000):
        for i in range(num_iterations):
            print(f"--- Iteration {i + 1} ---")
            if not self.find_start_point():  # 如果没有新的起始点，则终止迭代
                print("No new start points found. Ending iterations.")
                break
            self.iterate_algorithm()
            self.print_extrema()

    def print_extrema(self):
        print("Maximum points:", self.maximum)
        print("Minimum points:", self.minimum)
        print()

    def plot_signal(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.signal, label='Signal Waveform')
        for maxima in self.maximum:
            plt.plot(maxima[0], maxima[1], 'ro' if maxima[2] == 1 else 'rx')
        plt.title('Signal Waveform with Start Points')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_minimum(self):
        # 初始化图形
        plt.figure(figsize=(10, 4))

        # 如果需要避免重叠，可以设定一个起始 y 值并稍微增加
        base_y = 1.0  # 基础 y 值
        delta_y = 0.5  # 用于增加的小量以避免重叠

        # 绘制每个极小值的线段
        for idx, minima in enumerate(self.minimum):
            # 计算 y 坐标以避免重叠
            y = base_y + idx * delta_y
            # 绘制从开始点到结束点的线段
            plt.plot([minima[0], minima[1]], [y, y], 'g-', label='Minima range' if idx == 0 else "")
            # 可以添加点来明确显示起始和结束位置
            plt.plot(minima[0], y, 'go')  # 开始点
            plt.plot(minima[1], y, 'ro')  # 结束点

        plt.title('Plot of Minima Ranges')
        plt.xlabel('Index')
        plt.ylabel('Adjusted Position')
        plt.legend()
        plt.grid(True)
        plt.show()

def read_wav_to_signal(wav_path, num_frames=None):
    sample_rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data / np.max(np.abs(data))
    if num_frames is not None:
        data = data[:num_frames]
    return data

def main():
    # wav_path = '/Users/yuxuanliu/Downloads/classification-of-heart-sound-recordings-the-physionet-computing-in' \
    #           '-cardiology-challenge-2016-1.0.0/training-d/d0055.wav'
    wav_path = '/Users/yuxuanliu/Downloads/classification-of-heart-sound-recordings-the-physionet-computing-in' \
               '-cardiology-challenge-2016-1.0.0/training-d/d0051.wav'
    signal = read_wav_to_signal(wav_path, num_frames=200)  # Adjust this path and frame count as needed

    # signal = [4,5,6, 5, 4, 4, 4, 4, 4, 5, 6, 7, 8, 9, 7, 6, 5, 6, 7, 8, 9, 8, 7, 6, 5, 7, 8, 0, 5,7,8,10,12,13,12,12,11,10,11,12,13,12,7,1]
    tda = TDA(signal)
    tda.process_signal()
    tda.multiple_iterations(100)  # Adjust the number of iterations as needed
    tda.plot_signal()  # Display the waveform plot
    tda.print_extrema()  # Output extrema information
    tda.plot_minimum()

if __name__ == '__main__':
    main()
