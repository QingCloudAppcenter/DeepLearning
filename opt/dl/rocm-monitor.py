import sys
import subprocess
import re
import json

def readIndex(input):
    try:
        return int(input)
    except ValueError:
        return -1

pattern = re.compile(r"(-?[1-9]\d*\.\d*|-?0\.\d*[1-9]\d*|0?\.0+|0|-?[1-9]\d*)")
def readValue(input):
    global pattern
    number = pattern.findall(input)

    if len(number) != 1:
        return int(0)

    try:
        return int(number[0])
    except ValueError:
        return int(float(number[0]))

def readROCMInfo(str_smi, result):
    for line in str_smi.splitlines():
        items = filter(lambda x: x, line.split(' '))
        if len(items) == 10 and readIndex(items[0]) >= 0:
            index = readIndex(items[0])
            result['GPUUtil_' + str(index+1)] = readValue(items[9])
            result['MemoryUtil_' + str(index+1)] = readValue(items[8])
            result['PowerUsage_' + str(index+1)] = readValue(items[2])
            result['Temperature_' + str(index+1)] = readValue(items[1])
    return result

if __name__ == '__main__':
    result = {'GPUUtil_1': 0, 'MemoryUtil_1': 0, 'PowerUsage_1': 0, 'Temperature_1': 0,
              'GPUUtil_2': 0, 'MemoryUtil_2': 0, 'PowerUsage_2': 0, 'Temperature_2': 0,
              'GPUUtil_3': 0, 'MemoryUtil_3': 0, 'PowerUsage_3': 0, 'Temperature_3': 0,
              'GPUUtil_4': 0, 'MemoryUtil_4': 0, 'PowerUsage_4': 0, 'Temperature_4': 0,
              'GPUUtil_5': 0, 'MemoryUtil_5': 0, 'PowerUsage_5': 0, 'Temperature_5': 0,
              'GPUUtil_6': 0, 'MemoryUtil_6': 0, 'PowerUsage_6': 0, 'Temperature_6': 0,
              'GPUUtil_7': 0, 'MemoryUtil_7': 0, 'PowerUsage_7': 0, 'Temperature_7': 0,
              'GPUUtil_8': 0, 'MemoryUtil_8': 0, 'PowerUsage_8': 0, 'Temperature_8': 0,}

    try:
        sp = subprocess.Popen(['/opt/rocm/bin/rocm-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        str_smi = sp.communicate()
        result = readROCMInfo(str_smi[0], result)
    except:
        print json.dumps(result)
        sys.exit()

    print json.dumps(result)