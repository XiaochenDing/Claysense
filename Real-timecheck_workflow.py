# ================================
# Real-time 3D Printing Calibration Loop
# Author: Xiaochen Ding
# Purpose: Closed-loop ML-based UR5 speed control for 3D clay printing
# ================================

import os
import time
import socket
import threading
import torch
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
from scipy import stats
from glob import glob
import paramiko
from scp import SCPClient
from model.network_module_DINOv2 import DINO2ResAttClassifier
from data.data_module_wholeworkflow import ParametersDataModule
from train_config import preprocess
from datetime import datetime
import matplotlib.pyplot as plt
import shutil

# UR Configuration
UR_IP = "192.168.1.100"
SCRIPT_PORT = 30003
DASH_PORT = 30002
INITIAL_SCALE = 0.5

# Paths
DATA_DIR = r"Claysense\test_print_photo"
INPUT_FOLDER = os.path.join(DATA_DIR, "Image_detection")
OUTPUT_FOLDER = os.path.join(DATA_DIR, "Image_for_preprocess")
PREDICTION_FOLDER = os.path.join(DATA_DIR, "Image_for_prediction")
SAVE_FOLDER1 = os.path.join(DATA_DIR, "Image_for_save_raw")
SAVE_FOLDER2 = os.path.join(DATA_DIR, "Image_for_save_prediction")
DATA_CSV = os.path.join(DATA_DIR, "test_print.csv")
CHECKPOINT_PATH = r"Claysense\checkpoints\23042025\1234\DINO2ResAtt-model6.3_balanced_DINOv2-23042025-epoch=38-val_loss=0.35-val_acc=0.00.ckpt"
WAYPOINTS_CSV = r"Claysense\UR5\movej_positions.csv"
DATASET_NAME = "closeloop_test_v1"

# Raspberry Pi Configuration
PI_IP = "192.168.1.185"  
PI_USER = "user"
PI_PASS = "toi'sLAMA"
PI_IMAGE_DIR = "/home/user/Image_detection"
PI_TIMELAPSE_SCRIPT = "/home/user/camera_project/take_timelapse_xc.py"

# Constants
DATASET_MEAN = [0.2915257, 0.27048784, 0.14393276]
DATASET_STD = [0.066747, 0.06885352, 0.07679665]
BATCH_SIZE = 18
MONITOR_PAUSE = 20

# SSH and SCP clients
ssh = None
scp = None

# Load model
model = DINO2ResAttClassifier.load_from_checkpoint(
    checkpoint_path=CHECKPOINT_PATH,
    num_classes=3,
    gpus=1,
)
model.eval()

def connect_pi():
    global ssh, scp
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_IP, username=PI_USER, password=PI_PASS)
    scp = SCPClient(ssh.get_transport())

def start_timelapse():
    print("📷 Starting timelapse on Raspberry Pi...")
    ssh.exec_command(f"python3 {PI_TIMELAPSE_SCRIPT} &")

def stop_timelapse():
    print("🛑 Stopping timelapse on Raspberry Pi...")
    ssh.exec_command("pkill -f take_timelapse.py")

def sync_images_from_pi():
    print("⬇️ Transferring images from Raspberry Pi...")
    scp.get(PI_IMAGE_DIR, DATA_DIR, recursive=True)
    print("✅ Images transferred to PC.")

def clear_remote_folder(remote_path):
    delete_cmd = f"rm -rf {remote_path}/*"
    stdin, stdout, stderr = ssh.exec_command(delete_cmd)
    exit_status = stdout.channel.recv_exit_status()  
    if exit_status == 0:
        print(f"🧹 Cleared all contents from remote folder: {remote_path}")
    else:
        err = stderr.read().decode().strip()
        print(f"❌ Error deleting remote directory:{err}")

def backup_images_to_timestamped_folder(input_folder, save_root_folder):
    if not os.path.isdir(input_folder):
        raise ValueError(f"The input folder does not exist or is not a directory:{input_folder}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination_folder = os.path.join(save_root_folder, timestamp)
    os.makedirs(destination_folder, exist_ok=True)

    copied_count = 0
    for filename in os.listdir(input_folder):
        name_lower = filename.lower()
        _, ext = os.path.splitext(name_lower)

        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(destination_folder, filename)
        shutil.copy2(src_path, dst_path)  
        copied_count += 1
        
    print(f"📁 Subfolders{timestamp} created under" '{save_root_folder}')
    print(f"✅ Successfully copied {copied_count} Pictures to:{destination_folder}")
    return destination_folder 

def clear_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"⚠️ The path does not exist or is not a directory:{folder_path}")
        return
    
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        try:
            if os.path.isfile(entry_path) or os.path.islink(entry_path):
                os.remove(entry_path)                    
        except Exception as e:
            print(f"Error while deleting:{entry_path} -> {e}")

    print(f"🧹 Cleared all contents from: {folder_path}")


def set_speed_override(scale: float):
    msg = f"set speed {scale:.3f}\n".encode("ascii")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((UR_IP, DASH_PORT))
        s.sendall(msg)

def load_waypoints(path):
    waypoints = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = pd.read_csv(f)
        for row in reader.itertuples(index=False):
            waypoints.append(list(row))
    return waypoints

def build_urscript_joint_arc(waypoints, acc=3.1416, vel=0.07, blend=0.01):
    lines = [
        "def Program():",
        "  Clay_extruderTcp  = p[0, 0.1765, 0.058, -1.5708, 0, 0]",
        "  Clay_extruderWeight = 1.78",
        "  Clay_extruderCog= [0, 0.1765, 0.058]",
        "  set_tcp(Clay_extruderTcp)",
        "  set_payload(Clay_extruderWeight, Clay_extruderCog)",
        f"  movej({waypoints[0]}, a={acc}, v={vel}, r=0)"
    ]
    for wp in waypoints[1:]:
        lines.append(f"  movej({wp}, a={acc}, v={vel}, r={blend})")
    lines.append("end")
    return "\n".join(lines)


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
    print("✅ URScript uploaded and executed.")

def suggest_speed_change(extrusion, overhang):
    if extrusion == 0:
        return -0.1 if overhang <= 1 else -0.2
    elif extrusion == 1:
        return +0.1 if overhang == 0 else (0 if overhang == 1 else -0.1)
    elif extrusion == 2:
        return +0.2 if overhang == 0 else +0.1
    return 0

def preprocess_images(input_folder, output_folder):
    param = np.load(r"Claysense\dataset\cali_images\calibration_parameters0.npz")
    H_up = param['H_up']
    H_down = param['H_down']
    crop_up = param['crop_pts_up']
    crop_down = param['crop_pts_down']
    scale = param['scale_factor']
    warp_size_up = tuple(param['warp_size_up'])
    warp_size_down = tuple(param['warp_size_down'])
    output_size = tuple(param['output_size'])

    def crop_img(img, crop_pts, output_size):
        pts_dst = np.array([[0, 0], [output_size[0]-1, 0], [output_size[0]-1, output_size[1]-1], [0, output_size[1]-1]], dtype=np.float32)
        H_crop = cv.getPerspectiveTransform(crop_pts, pts_dst)
        return cv.warpPerspective(img, H_crop, output_size)

    def crop_to_center(img, crop_size):
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        left = max(center_x - crop_size[0] // 2, 0)
        top = max(center_y - crop_size[1] // 2, 0)
        right = min(center_x + crop_size[0] // 2, w)
        bottom = min(center_y + crop_size[1] // 2, h)
        cropped = img[top:bottom, left:right]
        return cv.resize(cropped, crop_size, interpolation=cv.INTER_LINEAR)

    def add_black_border(img, border_size):
        return cv.copyMakeBorder(img, border_size[1], border_size[1], border_size[0], border_size[0], cv.BORDER_CONSTANT, value=(0, 0, 0))

    os.makedirs(output_folder, exist_ok=True)
    img_list = glob(os.path.join(input_folder, "*.jpg"))
    cam0, cam1 = {}, {}
    for path in img_list:
        name = os.path.basename(path)
        if name.startswith("cam0_"):
            cam0[name[5:-4]] = path
        elif name.startswith("cam1_"):
            cam1[name[5:-4]] = path

    timestamps = sorted(set(cam0.keys()) & set(cam1.keys()))
    for ts in timestamps:
        img_up = cv.imread(cam0[ts])
        img_down = cv.imread(cam1[ts])
        if img_up is None or img_down is None:
            continue

        warp_up = cv.warpPerspective(img_up, H_up, warp_size_up)
        warp_down = cv.warpPerspective(img_down, H_down, warp_size_down)
        crop_u = crop_img(warp_up, crop_up, output_size)
        crop_d = crop_img(warp_down, crop_down, output_size)
        crop_u = cv.rotate(crop_u, cv.ROTATE_90_COUNTERCLOCKWISE)
        crop_d = cv.rotate(crop_d, cv.ROTATE_90_COUNTERCLOCKWISE)
        new_w = int(crop_u.shape[1] * scale)
        new_h = int(crop_u.shape[0] * scale)
        crop_u_scaled = cv.resize(crop_u, (new_w, new_h))
        x_up_center = new_w // 2
        x_down_center = crop_d.shape[1] // 2
        offset_x = x_down_center - x_up_center
        left = max(-offset_x, 0)
        x_up = left + max(offset_x, 0)
        x_down = left + max(-offset_x, 0)
        canvas_w = max(x_up + new_w, x_down + crop_d.shape[1])
        canvas_h = new_h + crop_d.shape[0]
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[0:new_h, x_up:x_up+new_w] = crop_u_scaled
        canvas[new_h:new_h+crop_d.shape[0], x_down:x_down+crop_d.shape[1]] = crop_d
        orig_h, orig_w = canvas.shape[:2]
        bordered = add_black_border(canvas, (1500, 1500))
        resized = cv.resize(bordered, (orig_w, orig_h))
        cropped_merge_img = crop_to_center(resized, (224, 224))
        output_path = os.path.join(output_folder, f"cropped_{ts}.jpg")
        cv.imwrite(output_path, cropped_merge_img)
        print(f"✅ Saved processed image to: {output_path}")

def update_csv_paths():
    df = pd.read_csv(DATA_CSV)
    cropped_files = sorted([
        fname for fname in os.listdir(OUTPUT_FOLDER)
        if fname.startswith("cropped_") and fname.endswith(".jpg")
    ])
    new_paths = [os.path.join(OUTPUT_FOLDER, fname) for fname in cropped_files]

    if len(new_paths) != len(df):
        print(f"⚠️ Warning: The number of cropped images ({len(new_paths)}) does not match the number of CSV rows ({len(df)}), which may cause errors.")
        
    min_len = min(len(new_paths), len(df))
    df.loc[:min_len-1, "img_path"] = new_paths[:min_len]
    df.to_csv(DATA_CSV, index=False)
    print(f"✅ The 'img_path' in the CSV file '{DATA_CSV}' has been updated to the cropped image paths.")



def make_dirs(path):
    try:
        os.makedirs(path)
    except:
        pass

def visualize_batch(batch, df, save_dir, dataset_std, dataset_mean):
    images, labels = batch
    batch_size = len(images)

    image_filenames = df['img_path'].values[:len(images)]

    for i, (img, label) in enumerate(zip(images, labels)):
        print(f"Processing image {i+1}/{batch_size}")
        img = img.permute(1, 2, 0)  
        img = img * torch.tensor(dataset_std) + torch.tensor(dataset_mean) 
        img = img.clamp(0, 1)
        img_np = img.numpy()

        output_filename = os.path.basename(image_filenames[i])
        output_path = os.path.join(save_dir, f"{output_filename}")

        plt.imsave(output_path, img_np)
        print(f"Saved image: {output_path}")

def Label_predict():
    data_module = ParametersDataModule(
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        csv_file=DATA_CSV,
        dataset_name=DATASET_NAME,
        mean=DATASET_MEAN,
        std=DATASET_STD,
        load_saved=False,
        transform=True
    )
  
    df = pd.read_csv(DATA_CSV)

    data_module.setup(stage="test", save=False, test_all=True)

    test_dataloader = data_module.test_dataloader()

    for batch_idx, batch in enumerate(test_dataloader):
        print(f"Processing batch {batch_idx + 1}")
        visualize_batch(batch, df, PREDICTION_FOLDER, DATASET_STD, DATASET_MEAN)

    print("All images processed.")

    img_paths = [
        os.path.join(PREDICTION_FOLDER, img)
        for img in os.listdir(PREDICTION_FOLDER)
        if os.path.splitext(img)[1] == ".jpg"
    ]

    # Preprocess and predict labels
    print("********* Claysense sample predictions *********")
    print("Extrusion | Overhang Success")
    print("*********************************************")
    layer_height_preds = []
    extrusion_preds = []
    

    for img_path in img_paths:
        pil_img = Image.open(img_path)
        x = preprocess(pil_img).unsqueeze(0)
        y_hats = model(x)
        y_hat0, y_hat1 = y_hats

        _, preds0 = torch.max(y_hat0, 1)
        _, preds1 = torch.max(y_hat1, 1)
        preds = torch.stack((preds0, preds1)).squeeze()


        preds_str = str(preds.numpy())
        img_basename = os.path.basename(img_path)
        print("Input:", img_basename, "->", "Prediction:", preds_str)
        # Collect predictions
        layer_height_preds.extend(preds0.numpy())
        extrusion_preds.extend(preds1.numpy())

    mode_result0 = stats.mode(layer_height_preds)   
    mode_result1 = stats.mode(extrusion_preds)

    final_layer_height_label = mode_result0.mode.item()
    final_extrusion_label     = mode_result1.mode.item()

    print(f"Layer Height: {final_layer_height_label}, Extrusion: {final_extrusion_label}")
    return final_layer_height_label, final_extrusion_label

def monitor_loop():
    scale = INITIAL_SCALE
    print(f"Initial speed: {scale}")
    connect_pi()
    clear_remote_folder(PI_IMAGE_DIR)

    while True:
        print("📷 Starting timelapse on Raspberry Pi...")
        start_timelapse()
        print("🕒 Waiting for 10 pairs of images on Raspberry Pi...")

        time.sleep(2)
        stdin, stdout, stderr = ssh.exec_command(f"ls -t {PI_IMAGE_DIR}")

        while True:
            stdin, stdout, stderr = ssh.exec_command(f"ls {PI_IMAGE_DIR} | grep cam0_ | wc -l")
            count_cam0 = int(stdout.read().decode().strip())
            stdin, stdout, stderr = ssh.exec_command(f"ls {PI_IMAGE_DIR} | grep cam1_ | wc -l")
            count_cam1 = int(stdout.read().decode().strip())
            if min(count_cam0, count_cam1) >= 10:
                break
            time.sleep(2)

        stop_timelapse()
        sync_images_from_pi()
        clear_remote_folder(PI_IMAGE_DIR)
        preprocess_images(INPUT_FOLDER, OUTPUT_FOLDER)
        update_csv_paths()
        final_extrusion_label, final_extrusion_label = Label_predict()

        if final_extrusion_label == 1 and final_extrusion_label == 1:
            break

        delta = suggest_speed_change(final_extrusion_label, final_extrusion_label)
        new_scale = min(1.0, max(0.1, scale + delta))
        set_speed_override(new_scale)
        scale = new_scale
        print(f"Adjusted new speed: {scale}")
        print(f"⏸️ Pausing {MONITOR_PAUSE}s...")
        time.sleep(MONITOR_PAUSE)
        backup_images_to_timestamped_folder(INPUT_FOLDER, SAVE_FOLDER1)
        clear_folder(INPUT_FOLDER)
        clear_folder(OUTPUT_FOLDER)
        backup_images_to_timestamped_folder(PREDICTION_FOLDER, SAVE_FOLDER2)
        clear_folder(PREDICTION_FOLDER)
        

    stop_timelapse()
    scp.close()
    ssh.close()
    set_speed_override(0.5)

def main():
    print("🚀 Uploading URScript and starting printing...")
    waypoints = load_waypoints(WAYPOINTS_CSV)
    script = build_urscript_joint_arc(waypoints)
    send_script(script)
    clear_folder(INPUT_FOLDER)
    clear_folder(OUTPUT_FOLDER)
    clear_folder(PREDICTION_FOLDER)
    print("⏳ Waiting 10s before monitoring...")
    time.sleep(10)
    monitor_loop()
    print("🎉 Print complete. Workflow finished.")

if __name__ == "__main__":
    main()
