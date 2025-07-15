import os 
import socket
import csv

UR_IP        = "192.168.1.100"
SCRIPT_PORT  = 30003
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
WAYPOINTS_CSV = os.path.join(BASE_DIR, "movej_positions.csv") 

def load_waypoints(path: str):
    waypoints = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        try:
            nums = [float(x) for x in header]
            if len(nums) == 6:
                waypoints.append(nums)
        except:
            pass

        for row in reader:
            if len(row) != 6:
                continue
            try:
                waypoints.append([float(x) for x in row])
            except ValueError:
                continue

    return waypoints


def build_urscript_joint_arc(waypoints,
    acc=3.1416,
    vel=0.04,
    blend=0.01,  
    first_zone="Zone000",
    other_zone="Zone001",
    z0=0.0,
    z1=0.01,
):
    lines = [
        "def Program():",
        "  Clay_extruderTcp = p[0, 0.1765, 0.058, -1.5708, -0, 0]",
        "  Clay_extruderWeight = 1.78",
        "  Clay_extruderCog = [0, 0.1765, 0.058]",
        "  Speed000 = 0.006",
        f"  {first_zone} = {z0}",
        f"  {other_zone} = {z1}",
        "",
        "  set_tcp(Clay_extruderTcp)",
        "  set_payload(Clay_extruderWeight, Clay_extruderCog)",
    ]
    lines.append(f"  movej({waypoints[0]}, a={acc}, v={vel}, r=0)")
    for wp in waypoints[1:]:
        lines.append(f"  movej({wp}, a={acc}, v={vel}, r={blend})")
    lines.append("end")
    return "\n".join(lines)


import time

def send_script(script: str, chunk_size=2084, delay=0.01):
    data = script + "\n"
    total = len(data)
    sent = 0
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((UR_IP, SCRIPT_PORT))
        while sent < total:
            end = min(sent + chunk_size, total)
            block = data[sent:end].encode('utf8')
            s.sendall(block)
            sent = end
            time.sleep(delay)
    print("✅The URScript is uploaded and executed.")

if __name__ == "__main__":
    print("🔍 Trying to open:", WAYPOINTS_CSV)
    wps = load_waypoints(WAYPOINTS_CSV)
    if not wps:
        print(f"⚠️ No waypoints found, check if`{WAYPOINTS_CSV}` exists and is in the correct format!")
        exit(1)

    urs = build_urscript_joint_arc(
        wps,
        acc=3.1416,
        vel=0.04,
        first_zone="Zone000",
        other_zone="Zone001"
    )
    print("[generated URScript]:\n", urs)
    send_script(urs)
    
