from signal_processing.extrema_finder import ExtremaFinder


class DTHF:
    def __init__(self, signal):
        self.signal = signal
        self.extrema_finder = ExtremaFinder(signal)
        self.maximum = self.extrema_finder.maximum
        self.minimum = self.extrema_finder.minimum
        self.find_start_point()

    def find_start_point(self):
        if len(self.maximum) < 3:
            return False
        new_start_found = False
        for i in range(1, len(self.maximum) - 1):
            prev_index = self.find_valid_extremum(i, -1)
            next_index = self.find_valid_extremum(i, 1)
            if prev_index is None or next_index is None:
                continue
            prev = self.maximum[prev_index]
            current = self.maximum[i]
            next = self.maximum[next_index]
            if current[2] != -1 and current[1] > prev[1] and current[1] > next[1]:
                current[2] = 1
                new_start_found = True
        return new_start_found

    def find_valid_extremum(self, index, step):
        test_index = index + step
        while 0 <= test_index < len(self.maximum):
            if self.maximum[test_index][2] != -1:
                return test_index
            test_index += step
        return None

    def iterate_algorithm(self):
        for minima in self.minimum:
            if minima[2] == 0:
                next_max = self.find_next_maximum(minima[0])
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
                    maxima[2] = -1

    def find_next_maximum(self, start_index):
        for maxima in self.maximum:
            if maxima[0] > start_index and maxima[2] != -1:
                return maxima
        return None

    def find_nearest_minimum(self, start_index, direction):
        if direction == -1:
            return max(
                (minima for minima in self.minimum if minima[0] < start_index),
                default=None,
                key=lambda x: abs(x[0] - start_index)
            )
        elif direction == 1:
            return min(
                (minima for minima in self.minimum if minima[0] > start_index),
                default=None,
                key=lambda x: abs(x[0] - start_index)
            )

    def multiple_iterations(self, num_iterations=10000):
        for _ in range(num_iterations):
            if not self.find_start_point():
                break
            self.iterate_algorithm()
