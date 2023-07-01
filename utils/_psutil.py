import psutil
import time
from utils._rich import new_progress



def display_usage(cpu_usage, mem_usage, bars=50):
    cpu_percent = cpu_usage / 100.0
    cpu_bar = '━' * int(cpu_percent * bars) + '-' * (bars - int(cpu_percent * bars))

    mem_percent = mem_usage / 100.0
    mem_bar = '━' * int(mem_percent * bars) + '-' * (bars - int(mem_percent * bars))

    text = 'Usage: CPU {} % |{}|, Memory {} % |{}|'.format(cpu_usage, cpu_bar, mem_usage, mem_bar)
    print(text, end='\r')


def print_memory_info():
    memory = psutil.virtual_memory()

    print('CPU usage: {} %'.format(psutil.cpu_percent()))
    print('Memory usage: {:.2f} / {:.2f} GB, {} %'.format(
        memory.used / (1024.0 ** 3), 
        memory.total / (1024.0 ** 3), 
        memory.percent
    ))


while True:
    cpu_usage = psutil.cpu_percent()
    mem_usage = psutil.virtual_memory().percent
    display_usage(cpu_usage, mem_usage)
    time.sleep(1)
