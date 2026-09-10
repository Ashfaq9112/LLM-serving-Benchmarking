import threading  
import subprocess
import time

class PowerLogger:
    def __init__(self, gpu_index: int = 0, interval_s: float = 1.0):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.stop_event = threading.Event()#always sets as false default
        self.samples = []
        self.thread = None
    def poll(self):
        while not self.stop_event.is_set():
            results = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", str(self.gpu_index)], capture_output=True, text=True)
            watts = float(results.stdout.strip())
            self.samples.append(watts)
            time.sleep(self.interval_s)

    def start(self):
        self.thread = threading.Thread(target=self.poll)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        avg_power = sum(self.samples) / len(self.samples)
        return avg_power



