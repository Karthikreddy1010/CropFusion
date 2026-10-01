"""Background free-space logger (diagnostic only): appends C:/G: free GB every 30 s."""
import shutil, time, datetime as dt
from md_common import LOGS
p = LOGS / "disk_monitor.csv"
if not p.exists():
    p.write_text("time,c_free_gb,g_free_gb\n")
while True:
    c = shutil.disk_usage("C:\\").free / 1e9
    try:
        g = shutil.disk_usage("G:\\").free / 1e9
    except OSError:
        g = float("nan")
    with open(p, "a") as fh:
        fh.write(f"{dt.datetime.now().isoformat(timespec='seconds')},{c:.2f},{g:.2f}\n")
    time.sleep(30)
