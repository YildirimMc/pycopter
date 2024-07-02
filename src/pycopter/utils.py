import numpy as np

def find_nearest_idx(array, val):
    arr = np.asarray(array)
    return (np.abs(arr - val)).argmin()

def find_interval_idx(array, val):
    """
    """
    i = 0
    for i in range(len(array-1)):
        if array[i][0] <= val < array[i+1][0]:
            break
    return i

def read_txt(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        data = np.empty(shape=(len(lines), 2))
        for i, line in enumerate(lines):
            data[i] = line.split(" ")
        return data
    
def probe_txt(filepath, x):
    data = read_txt(filepath)
    idx = find_interval_idx(data, x)
    slope = (data[idx+1][1] - data[idx][1]) / (data[idx+1][0] - data[idx][0])
    probe = (x - data[idx][0]) * slope + data[idx][1] 
    return probe

def request_xfoil_probe(alfa, reynolds, mach):
    pass

if __name__ == "__main__":
    filepath = "data/figures/MeanLiftCoef.vs.MeanDragCoef_(Fig3-1.AMCP706-201).txt"
    print(probe_txt(filepath, 0.2))

