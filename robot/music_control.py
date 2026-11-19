# -*- coding: utf-8 -*-
"""
树莓派5 TonyPi 实时识别 + 舞蹈控制版本

基于原 infer_pc_lockvote.py，听力音乐识别占主导，异步调用动作脚本，音乐停止时输入“1”停止脚本，然后主程序归中，并添加姿态检测
"""

import time
import argparse
import collections
import os
import sys
import subprocess
import numpy as np
import sounddevice as sd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import hiwonder.ros_robot_controller_sdk as rrc

# ========= 导入 TonyPi 控制模块 =========
try:
    import hiwonder.ActionGroupControl as AGC

    print("✅ TonyPi ActionGroupControl 模块加载成功")
except Exception as e:
    print(f"❌ 加载 TonyPi 控制模块失败: {e}")
    print("⚠️  请确保在 TonyPi 机器人上运行此程序")
    sys.exit(1)

# 全局变量
current_process = None
current_song = None
board = None
count1 = 0
count2 = 0

# ========= 歌曲到动作脚本的映射 =========
song_scripts = {
    "renshi": "renshi_actions",
    "dream": "Huawei_actions",  # dream 对应 Huawei
    "aoteman": "aoteman_actions",
    "amigo": "amigo_actions",
    "wuren": "wuren_actions"
}

# 动作时长字典（单位：秒）
action_durations = {
    "0_Stand": 1.0,
    "ChangE_BenYue": 5.1,
    "Chiken_stand": 7.6,
    "ChuDi_left_right": 17.05,
    "DaPengFlyFly": 10.7,
    "DiJia": 3.3,
    "DiTou_HuiShou": 15.975,
    "DiTou_HuiShou2": 15.975,
    "Dimian_1": 29.4,
    "Hand_Both_Hui_n": 28.9,
    "HuiShou_both": 8.9,
    "JiXie_1": 19.21,
    "JiXie_2.5": 7.15,
    "JiXie_2": 15.0,
    "JiXie_3": 9.9,
    "JiaYou_hug": 22.0,
    "Jun_2": 8.8,
    "Jun_strat": 10.6,
    "KaiHua": 10.8,
    "KaiHua2": 10.8,
    "Kick_Right": 6.0,
    "KunpengFlyFly": 11.4,
    "Left_Right": 15.6,
    "Left_Right2": 15.6,
    "Left_Right_NaoYY": 10.1,
    "Left_Right_XiaoKick": 8.5,
    "Left_Right_XiaoKick2": 8.5,
    "Monkey_Moon": 7.3,
    "Mutong_zhilu": 6.8,
    "Nidawosa": 14.4,
    "Niu_2": 6.3,
    "Niu_3": 5.35,
    "PiCha": 14.0,
    "Push-up": 9.2,
    "Rao": 10.0,
    "Roll_back": 7.3,
    "Roll_forward": 6.0,
    "SheJian_both": 19.8,
    "ShuQing": 13.4,
    "Side_kick_left": 8.4,
    "Side_kick_right": 8.4,
    "Slide_left": 5.8,
    "Slide_right": 8.4,
    "Turn_Left4": 10.8,
    "Turn_Right_8": 8.4,
    "Upercut_right": 1.2,
    "XianNv_Fly": 6.4,
    "XiaoNiaoYiRen": 6.9,
    "YongChun": 2.6,
    "bow": 3.8,
    "qisheng": 5.0,
    "stand_up_back": 5.2,
    "stand_up_front": 5.5
}

# ========= 姿态检测函数（从 Fall_and_Stand.py 复制并优化） =========
def stand_up():
    global count1, count2

    try:
        accel_data = board.get_imu()
        angle_y = int(math.degrees(math.atan2(accel_data[0], accel_data[2])))

        # 面朝地（前倾，angle_y 接近 0°，|angle_y| < 10°）
        if abs(angle_y) < 10:
            count2 += 1
        else:
            count2 = 0

        # 往后倒（后倾，angle_y 接近 ±180°，|angle_y| > 160°）
        if abs(angle_y) > 160:
            count1 += 1
        else:
            count1 = 0

        if count2 >= 5:  # 连续0.5秒检测到前倾
            count2 = 0
            print("stand up front！(面朝地)")
            AGC.runActionGroup('stand_up_front')
            return 'front'

        if count1 >= 5:  # 连续0.5秒检测到后倾
            count1 = 0
            print("stand up back！(往后倒)")
            AGC.runActionGroup('stand_up_back')
            return 'back'

        return None

    except BaseException as e:
        print(f"姿态检测错误: {e}")
        return None

# ========= 检查姿态并归中 =========
def check_posture():
    global count1, count2
    count1 = 0
    count2 = 0
    result = None
    for _ in range(5):  # 检测0.5秒（5次，每次0.1秒）
        result = stand_up()
        if result is not None:
            break
        time.sleep(0.1)

    if result is None:
        print("姿态正常，执行归中: 0_Stand")
        AGC.runActionGroup("0_Stand")
        time.sleep(action_durations.get("0_Stand", 1.0) + 0.5)
    else:
        print(f"已执行站起动作 ({result})，等待完成")
        time.sleep(action_durations.get(f"stand_up_{result}", 5.0) + 0.5)

# ========= 启动舞蹈 =========
def start_dance(song_name, script_path):
    global current_process, current_song
    current_song = song_name
    script_name = song_scripts.get(song_name)

    if not script_name:
        print(f"\n❌ 歌曲 {song_name} 没有对应的动作脚本")
        current_song = None
        return False

    full_script_path = os.path.join(script_path, f"{script_name}.py")

    if not os.path.exists(full_script_path):
        print(f"\n❌ 动作脚本不存在: {full_script_path}")
        current_song = None
        return False

    print(f"📂 启动动作脚本: {script_name}.py")

    try:
        # 异步启动子进程运行脚本
        current_process = subprocess.Popen(
            [sys.executable, full_script_path],
            stdin=subprocess.PIPE,
            text=True
        )
        return True
    except Exception as e:
        print(f"\n❌ 启动动作脚本失败: {e}")
        current_song = None
        return False

# ========= 停止舞蹈并执行归中 =========
def stop_dance_and_reset():
    """音乐停止时，直接执行归中动作，并杀死当前正在运行的动作脚本"""
    global current_process, current_song
    try:
        if current_process and current_process.poll() is None:
            print("\n⏹️  【终止当前动作脚本】")
            current_process.stdin.write("1\n")
            current_process.stdin.flush()
            time.sleep(0.1)  # 保持0.1s，加快响应
            if current_process.poll() is None:
                current_process.terminate()
                try:
                    current_process.wait(timeout=0.5)  # 保持0.5s
                except subprocess.TimeoutExpired:
                    current_process.kill()
                    current_process.wait()

        print("\n🎬 执行归中动作: 0_Stand")
        try:
            AGC.runActionGroup("0_Stand")  # 直接调用 TonyPi 动作组
            time.sleep(action_durations.get("0_Stand", 1.0) + 0.0)  # 保持1.0s，无缓冲
        except Exception as e:
            print(f"\n❌ 归中动作执行失败: {e}, 尝试备用方案...")
            subprocess.run(
                ["python3", "-c", "import hiwonder.ActionGroupControl as AGC; AGC.runActionGroup('0_Stand')"],
                check=True,
                timeout=3
            )
            time.sleep(action_durations.get("0_Stand", 1.0) + 0.0)

        current_song = None
        return True
    except Exception as e:
        print(f"\n❌ 执行归中动作时出错: {e}")
        current_song = None
        return False

# ========= 预处理 =========
def read_wav_array(y, sr, target_sr=22050):
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)

    if sr != target_sr:
        x_old = np.linspace(0, 1, num=len(y), endpoint=False)
        new_len = int(round(len(y) * target_sr / sr))
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)

    m = np.max(np.abs(y)) + 1e-9
    if m > 1.0:
        y = y / m
    return y, target_sr

def melspec(y, sr=22050, n_fft=1024, hop_length=256, n_mels=128, fmin=20.0, fmax=None, eps=1e-10):
    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (len(y) - n_fft) // hop_length if len(y) >= n_fft else 0

    if frames <= 0:
        y = np.pad(y, (0, n_fft - len(y) + 1))
        frames = 1 + (len(y) - n_fft) // hop_length

    spec = []
    for i in range(frames):
        s = i * hop_length
        frame = y[s:s + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        frame = frame * win
        spec.append(np.abs(np.fft.rfft(frame, n=n_fft)))

    spec = np.stack(spec, axis=1)

    if fmax is None:
        fmax = sr / 2

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    m_min, m_max = hz_to_mel(fmin), hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = mel_to_hz(m_pts)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    bins = np.searchsorted(fft_freqs, f_pts)
    bins = np.clip(bins, 0, spec.shape[0] - 1)

    fb = np.zeros((n_mels, spec.shape[0]), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        if l == c or c == r:
            continue
        fb[m - 1, l:c] = (np.arange(l, c) - l) / (c - l + 1e-12)
        fb[m - 1, c:r] = (r - np.arange(c, r)) / (r - c + 1e-12)

    mel = fb @ (spec ** 2)
    return np.log(mel + eps).astype(np.float32)

def crop_or_pad_time(spec, target_T):
    n_mels, T = spec.shape
    if T == target_T:
        return spec

    if T > target_T:
        start = (T - target_T) // 2
        return spec[:, start:start + target_T]

    pad = target_T - T
    l = pad // 2
    r = pad - l
    return np.pad(spec, ((0, 0), (l, r)), mode="constant", constant_values=spec.min())

# ========= 模型 =========
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, pool=True, p_drop=0.25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, k, 1, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return self.drop(x)

class SmallResNet(nn.Module):
    def __init__(self, n_classes=6, p_drop=0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ConvBlock(32, 64, pool=True, p_drop=p_drop)
        self.layer2 = ConvBlock(64, 128, pool=True, p_drop=p_drop)
        self.layer3 = ConvBlock(128, 192, pool=True, p_drop=p_drop)
        self.layer4 = ConvBlock(192, 256, pool=True, p_drop=p_drop)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

# ========= 主逻辑 =========
def main():
    global board, current_process, current_song
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--buffer_sec", type=float, default=6.0)
    ap.add_argument("--step_sec", type=float, default=0.05, help="20Hz 判定，提升响应")
    ap.add_argument("--ema_beta", type=float, default=0.2, help="logits EMA（越小越跟手）")
    ap.add_argument("--smooth_k", type=int, default=2, help="概率展示小窗")
    ap.add_argument("--conf_thresh", type=float, default=0.5, help="置信阈值，增至0.5")
    ap.add_argument("--margin_start", type=float, default=0.06, help="开始：top-second 最小间隔，增至0.06")
    ap.add_argument("--consec_start", type=int, default=2, help="开始：连续命中帧数，增至2")
    ap.add_argument("--lock_sec", type=float, default=45.0)
    ap.add_argument("--stop_consec", type=int, default=4, help="停止：连续非当前帧数（含 noises），增至4")
    ap.add_argument("--margin_switch", type=float, default=0.05, help="停止：对手领先当前的最小间隔，增至0.05")
    ap.add_argument("--consec_switch", type=int, default=1, help="停止：对手连续压制帧数")
    ap.add_argument("--decision_sec", type=float, default=1.5, help="开始/停止 最长决策时间")
    ap.add_argument("--scripts-dir", required=True, help="队友提供的编舞Python脚本目录；不是.d6a动作目录")

    args = ap.parse_args()
    missing = [name + ".py" for name in song_scripts.values()
               if not os.path.isfile(os.path.join(args.scripts_dir, name + ".py"))]
    if missing:
        ap.error("编舞脚本缺失: " + ", ".join(missing))

    # 初始化机器人底层驱动
    board = rrc.Board()
    board.enable_reception()  # 使能数据接收
    time.sleep(1)

    # 程序启动时执行站立归中动作
    print("\n🎬 程序启动：执行归中动作 0_Stand")
    try:
        subprocess.run(
            ["python3", "-c", "import hiwonder.ActionGroupControl as AGC; AGC.runActionGroup('0_Stand')"],
            check=True,
            timeout=4
        )
        time.sleep(action_durations.get("0_Stand", 1.0) + 0.2)
    except subprocess.SubprocessError as e:
        print(f"\n❌ 启动归中动作失败: {e}")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt.get("classes", ["renshi", "dream", "aoteman", "amigo", "wuren", "noises"])
    a = ckpt.get("args", {})
    frames = int(a.get("frames", 256))
    mel = int(a.get("mel", 128))
    nfft = int(a.get("nfft", 1024))
    hop = int(a.get("hop", 256))
    dropout = float(a.get("dropout", 0.25))

    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallResNet(n_classes=len(classes), p_drop=dropout).to(device_t)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sd.default.samplerate = args.sr
    if args.device is not None:
        sd.default.device = args.device

    chan, block = 1, 1024
    ring_len = int(args.buffer_sec * args.sr)
    ring = np.zeros(ring_len, dtype=np.float32)
    write_pos = 0

    def callback(indata, frames_cb, time_info, status):
        nonlocal ring, write_pos
        if status:
            print(status)
        audio = indata[:, 0].copy()
        L = len(audio)
        if L > ring_len:
            audio = audio[-ring_len:]
            L = ring_len

        end = write_pos + L
        if end <= ring_len:
            ring[write_pos:end] = audio
        else:
            first = ring_len - write_pos
            ring[write_pos:] = audio[:first]
            ring[:L - first] = audio[first:]
        write_pos = (write_pos + L) % ring_len

    stream = sd.InputStream(channels=chan, callback=callback, blocksize=block)
    stream.start()

    prob_hist = collections.deque(maxlen=max(4, args.smooth_k))
    ema_logits = None
    window_labels = collections.deque(maxlen=128)
    song_start_t = 0.0
    start_win_t0 = None
    stop_win_t0 = None
    consec_start_cnt = 0
    stop_notcur_cnt = 0
    lead_consec = 0
    lead_opponent = None
    last_step = 0
    step_samples = int(args.step_sec * args.sr)

    def forward_logits(buf):
        y, _ = read_wav_array(buf, args.sr, target_sr=22050)
        S = melspec(y, 22050, n_fft=nfft, hop_length=hop, n_mels=mel)
        S = crop_or_pad_time(S, frames)
        mu, sigma = S.mean(), S.std() + 1e-6
        x = torch.from_numpy((S - mu) / sigma).unsqueeze(0).unsqueeze(0).to(device_t)
        with torch.no_grad():
            return model(x).cpu().numpy()[0]

    tnow = time.time

    print("\n" + "=" * 60)
    print("🤖 TonyPi 舞蹈机器人 - 实时音乐识别系统")
    print("=" * 60)
    print(f"📁 动作文件路径: {args.scripts_dir}")
    print(f"🎵 支持的歌曲: {', '.join([c for c in classes if c != 'noises'])}")
    print("=" * 60)
    print("[INFO] 开始监听（Ctrl+C 停止）\n")

    try:
        while True:
            cur_write = write_pos
            do_step = False

            if last_step + step_samples <= cur_write:
                do_step = True
            elif cur_write < last_step and cur_write + ring_len - last_step >= step_samples:
                do_step = True

            if not do_step:
                time.sleep(0.005)
                continue

            idx = (cur_write - ring_len) % ring_len
            buf = ring[idx:] if idx == 0 else np.concatenate([ring[idx:], ring[:idx]], axis=0)
            last_step = cur_write

            logits = forward_logits(buf)

            if ema_logits is None:
                ema_logits = logits
            else:
                ema_logits = args.ema_beta * ema_logits + (1 - args.ema_beta) * logits

            prob = torch.softmax(torch.tensor(ema_logits), dim=0).numpy()
            prob_hist.append(prob)
            avg_prob = np.mean(prob_hist, axis=0)

            order = np.argsort(-avg_prob)
            top_i, sec_i = int(order[0]), int(order[1])
            top_p, sec_p = float(avg_prob[top_i]), float(avg_prob[sec_i])
            top_name, sec_name = classes[top_i], classes[sec_i]
            margin = top_p - sec_p

            lock_left = max(0, args.lock_sec - (tnow() - song_start_t)) if current_song else 0
            lock_txt = f" | lock {lock_left:4.1f}s" if current_song and lock_left > 0 else ""
            bar = "=" * int(top_p * 20) + "-" * int((1 - top_p) * 20)

            print(
                f"\r[pred={top_name:8s}] p={top_p:0.2f} Δ={margin:0.2f} |{bar}|  cur={str(current_song):8s}{lock_txt}",
                end=""
            )

            if current_song is None:
                # 开始检测逻辑，添加0.5s延迟
                if start_win_t0 is None:
                    if (top_name != "noises" and top_p >= args.conf_thresh) or (
                            top_name != "noises" and margin >= args.margin_start):
                        start_win_t0 = tnow()
                if start_win_t0 is not None and tnow() - start_win_t0 >= 0.5:  # 0.5s后开始计帧
                    window_labels.append(top_name if top_name != "noises" and top_p >= args.conf_thresh else "low")

                    if top_name != "noises" and margin >= args.margin_start:
                        consec_start_cnt += 1
                    else:
                        consec_start_cnt = 0

                    if consec_start_cnt >= args.consec_start:
                        current_song = top_name
                        song_start_t = tnow()
                        start_dance(current_song, args.scripts_dir)
                        start_win_t0 = None
                        window_labels.clear()
                        stop_win_t0 = None
                        stop_notcur_cnt = 0
                        lead_consec = 0
                        lead_opponent = None
                        prob_hist.clear()
                        continue

                    if tnow() - start_win_t0 >= args.decision_sec:
                        start_win_t0 = None
                        window_labels.clear()

            else:
                # 听力音乐识别占主导的停止逻辑
                if tnow() - song_start_t >= args.lock_sec:
                    if top_name != current_song and top_p >= args.conf_thresh:
                        print(f"\n🎵 检测到音乐变化或停止: 从 {current_song} 切换至 {top_name}")
                        stop_dance_and_reset()
                        if top_name != "noises":
                            current_song = top_name
                            song_start_t = tnow()
                            start_dance(current_song, args.scripts_dir)
                        continue

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n\n⚠️  收到中断信号...")
        if current_song:
            stop_dance_and_reset()
    finally:
        stream.stop()
        stream.close()
        print("=" * 60)
        print("👋 程序已退出")
        print("=" * 60)

if __name__ == "__main__":
    main()