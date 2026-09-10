import subprocess
import threading
import time

class VRAMPeakLogger:
    def __init__(self, gpu_index: int = 0, interval_s: float = 1.0):
            self.gpu_index = gpu_index
            self.interval_s = interval_s
            self.stop_event = threading.Event()#always sets as false default
            self.samples = []
            self.thread = None
    def poll(self):
        while not self.stop_event.is_set():
            results = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(self.gpu_index)], capture_output=True, text=True)
            vrammem = float(results.stdout.strip())
            self.samples.append(vrammem)
            time.sleep(self.interval_s)

    def start(self):
        self.thread = threading.Thread(target=self.poll)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        maxvram = max(self.samples)
        return maxvram