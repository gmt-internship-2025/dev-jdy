# tools/perf_monitor.py
import subprocess, threading, re, time, csv, os
from datetime import datetime

TEGRA_CMD = ["tegrastats", "--interval", "1000"]  # 1s
GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
CPU_RE  = re.compile(r"CPU\s+\[(.+?)\]")
RAM_RE  = re.compile(r"RAM\s+(\d+)/(\d+)")
EMC_RE  = re.compile(r"EMC_FREQ\s+(\d+)%")

class TegraStatsMonitor:
    def __init__(self, csv_path="perf_tegrastats.csv"):
        self.csv_path = csv_path
        self.proc = None
        self.thread = None
        self._stop = False
        self._f = None
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    def start(self):
        self._f = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self._f)
        self.writer.writerow(["ts", "gpu_pct", "emc_pct", "ram_used_mb", "ram_total_mb", "cpu_raw"])
        self.proc = subprocess.Popen(TEGRA_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        for line in self.proc.stdout:
            if self._stop: break
            ts = datetime.now().isoformat(timespec="seconds")
            gpu = GR3D_RE.search(line)
            emc = EMC_RE.search(line)
            cpu = CPU_RE.search(line)
            ram = RAM_RE.search(line)
            self.writer.writerow([
                ts,
                int(gpu.group(1)) if gpu else "",
                int(emc.group(1)) if emc else "",
                int(ram.group(1)) if ram else "",
                int(ram.group(2)) if ram else "",
                cpu.group(1) if cpu else "",
            ])
            self._f.flush()

    def stop(self):
        self._stop = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        if self.thread:
            self.thread.join(timeout=2)
        if self._f:
            self._f.close()

if __name__ == "__main__":
    mon = TegraStatsMonitor()
    try:
        mon.start()
        while True: time.sleep(1)
    except KeyboardInterrupt:
        mon.stop()

