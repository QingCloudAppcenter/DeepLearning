import sys
import subprocess
import xmltodict
import collections
import json

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def readValue(input):
    str_value = input.split(' ')[0]
    try:
        return int(str_value)
    except ValueError:
        return int(float(str_value))

if __name__ == '__main__':

    result = {'GPUUtil_1': 0, 'MemoryUtil_1': 0, 'PowerUsage_1': 0, 'Temperature_1': 0, 'MemoryTotal_1': 0, 'MemoryUsed_1': 0, 'MemoryFree_1': 0,
              'GPUUtil_2': 0, 'MemoryUtil_2': 0, 'PowerUsage_2': 0, 'Temperature_2': 0, 'MemoryTotal_2': 0, 'MemoryUsed_2': 0, 'MemoryFree_2': 0,}

    dict_smi = {}

    try:
        sp = subprocess.Popen(['/usr/bin/nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        str_smi = sp.communicate()
        dict_smi = xmltodict.parse(str_smi[0])
    except:
        print json.dumps(result)
        sys.exit()

    if(dict_smi.get('nvidia_smi_log') is None):
        print json.dumps(result)
        sys.exit()

    retrieve = {'GPUUtil': 'utilization.gpu_util', 'PowerUsage': 'power_readings.power_draw', 'Temperature': 'temperature.gpu_temp',
                'MemoryTotal': 'fb_memory_usage.total', 'MemoryUsed': 'fb_memory_usage.used', 'MemoryFree': 'fb_memory_usage.free'}

    if '1' == dict_smi['nvidia_smi_log']['attached_gpus']:
        index = 0
        dict_gpu = flatten(dict_smi['nvidia_smi_log']['gpu'], sep='.')
        for k, v in retrieve.items():
            result[k + '_' + str(index+1)] = readValue(dict_gpu.get(v))
        result['MemoryUtil_' + str(index+1)] = readValue(dict_gpu.get('fb_memory_usage.used')) * 100 / readValue(dict_gpu.get('fb_memory_usage.total'))
        print json.dumps(result)
        sys.exit()
    elif '2' == dict_smi['nvidia_smi_log']['attached_gpus']:
        for index in range(2):
            dict_gpu = flatten(dict_smi['nvidia_smi_log']['gpu'][index], sep='.')
            for k, v in retrieve.items():
                result[k + '_' + str(index+1)] = readValue(dict_gpu.get(v))
            result['MemoryUtil_' + str(index+1)] = readValue(dict_gpu.get('fb_memory_usage.used')) * 100 / readValue(dict_gpu.get('fb_memory_usage.total'))
        print json.dumps(result)
        sys.exit()

    print json.dumps(result)
