# -*- coding: utf-8 -*-
"""
PC端实时识别（终端打印；决策窗参数可配置，实际延迟需测量；无任何按类特判）

要点：
- step_sec=0.2（5Hz），更快
- 进入“开始/停止”判定窗后，最迟 decision_sec=3.0 秒强制决策
- 开始窗：出现非 noises 且 ≥阈值，或 margin≥起跳阈值 即刻开窗
- 停止窗（解锁后）：只要 top≠当前 就开窗（不等置信度高）
- 触发条件更敏捷：consec_start=2，stop_consec=2，consec_switch=2；margin 更低

"""

import time, argparse, collections, os
import numpy as np
import sounddevice as sd
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf

# ========= 预处理 =========
def read_wav_array(y, sr, target_sr=22050):
    if y.ndim > 1: y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != target_sr:
        x_old = np.linspace(0, 1, num=len(y), endpoint=False)
        new_len = int(round(len(y) * target_sr / sr))
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
    m = np.max(np.abs(y)) + 1e-9
    if m > 1.0: y = y / m
    return y, target_sr

def melspec(y, sr=22050, n_fft=1024, hop_length=256, n_mels=128, fmin=20.0, fmax=None, eps=1e-10):
    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (len(y) - n_fft) // hop_length if len(y) >= n_fft else 0
    if frames <= 0:
        y = np.pad(y, (0, n_fft - len(y) + 1)); frames = 1 + (len(y) - n_fft) // hop_length
    spec = []
    for i in range(frames):
        s = i * hop_length
        frame = y[s:s+n_fft]
        if len(frame) < n_fft: frame = np.pad(frame, (0, n_fft - len(frame)))
        frame = frame * win
        spec.append(np.abs(np.fft.rfft(frame, n=n_fft)))
    spec = np.stack(spec, axis=1)
    if fmax is None: fmax = sr/2
    def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz/700.0)
    def mel_to_hz(m): return 700.0 * (10**(m/2595.0) - 1.0)
    m_min, m_max = hz_to_mel(fmin), hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels+2)
    f_pts = mel_to_hz(m_pts)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
    bins = np.searchsorted(fft_freqs, f_pts); bins = np.clip(bins, 0, spec.shape[0]-1)
    fb = np.zeros((n_mels, spec.shape[0]), dtype=np.float32)
    for m in range(1, n_mels+1):
        l, c, r = bins[m-1], bins[m], bins[m+1]
        if l==c or c==r: continue
        fb[m-1, l:c] = (np.arange(l, c) - l) / (c - l + 1e-12)
        fb[m-1, c:r] = (r - np.arange(c, r)) / (r - c + 1e-12)
    mel = fb @ (spec**2)
    return np.log(mel + eps).astype(np.float32)

def crop_or_pad_time(spec, target_T):
    n_mels, T = spec.shape
    if T == target_T: return spec
    if T > target_T:
        start = (T - target_T)//2
        return spec[:, start:start+target_T]
    pad = target_T - T
    l = pad//2; r = pad - l
    return np.pad(spec, ((0,0),(l,r)), mode="constant", constant_values=spec.min())

# ========= 模型 =========
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, pool=True, p_drop=0.25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, k, 1, p, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        x = self.conv(x); x = self.pool(x); return self.drop(x)

class SmallResNet(nn.Module):
    def __init__(self, n_classes=6, p_drop=0.25):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 32, 5, 2, 2, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = ConvBlock(32, 64,  pool=True, p_drop=p_drop)
        self.layer2 = ConvBlock(64, 128, pool=True, p_drop=p_drop)
        self.layer3 = ConvBlock(128, 192, pool=True, p_drop=p_drop)
        self.layer4 = ConvBlock(192, 256, pool=True, p_drop=p_drop)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.Linear(256,256), nn.ReLU(inplace=True),
            nn.Dropout(p_drop), nn.Linear(256, n_classes)
        )
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x); return self.head(x)

# ========= 主逻辑（可配置决策窗） =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--buffer_sec", type=float, default=6.0)
    ap.add_argument("--step_sec", type=float, default=0.2, help="5Hz 判定，提升响应")
    ap.add_argument("--ema_beta", type=float, default=0.5, help="logits EMA（越小越跟手）")
    ap.add_argument("--smooth_k", type=int, default=5, help="概率展示小窗")
    ap.add_argument("--conf_thresh", type=float, default=0.55)
    # 开始判据（更快）
    ap.add_argument("--margin_start", type=float, default=0.08, help="开始：top-second 最小间隔")
    ap.add_argument("--consec_start", type=int, default=2, help="开始：连续命中帧数")
    # 锁定与停止（更快）
    ap.add_argument("--lock_sec", type=float, default=45.0)
    ap.add_argument("--stop_consec", type=int, default=2, help="停止：连续非当前帧数（含 noises）")
    ap.add_argument("--margin_switch", type=float, default=0.06, help="停止：对手领先当前的最小间隔")
    ap.add_argument("--consec_switch", type=int, default=2, help="停止：对手连续压制帧数")
    # 3秒强制决策窗
    ap.add_argument("--decision_sec", type=float, default=3.0, help="开始/停止 最长决策时间")
    ap.add_argument("--cooldown", type=float, default=0.6)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt.get("classes", ["renshi","dream","aoteman","amigo","wuren","noises"])
    a = ckpt.get("args", {})
    frames = int(a.get("frames",256)); mel=int(a.get("mel",128))
    nfft=int(a.get("nfft",1024)); hop=int(a.get("hop",256)); dropout=float(a.get("dropout",0.25))

    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallResNet(n_classes=len(classes), p_drop=dropout).to(device_t)
    model.load_state_dict(ckpt["model"]); model.eval()

    # 音频流
    sd.default.samplerate = args.sr
    if args.device is not None: sd.default.device = args.device
    chan, block = 1, 1024
    ring_len = int(args.buffer_sec * args.sr)
    ring = np.zeros(ring_len, dtype=np.float32); write_pos = 0

    def callback(indata, frames_cb, time_info, status):
        nonlocal ring, write_pos
        if status: print(status)
        audio = indata[:,0].copy()
        L = len(audio)
        if L > ring_len: audio = audio[-ring_len:]; L = ring_len
        end = write_pos + L
        if end <= ring_len:
            ring[write_pos:end] = audio
        else:
            first = ring_len - write_pos
            ring[write_pos:] = audio[:first]
            ring[:L-first] = audio[first:]
        write_pos = (write_pos + L) % ring_len

    stream = sd.InputStream(channels=chan, callback=callback, blocksize=block); stream.start()

    # 状态缓存
    prob_hist = collections.deque(maxlen=max(4, args.smooth_k))
    ema_logits = None
    window_labels = collections.deque(maxlen=128)  # 记录当前判定窗的标签
    current_song = None
    song_start_t = 0.0

    # 判定窗时间戳
    start_win_t0 = None
    stop_win_t0  = None

    # 连续计数
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
        mu, sigma = S.mean(), S.std()+1e-6
        x = torch.from_numpy((S-mu)/sigma).unsqueeze(0).unsqueeze(0).to(device_t)
        with torch.no_grad():
            return model(x).cpu().numpy()[0]

    tnow = time.time
    print("[INFO] 开始监听（Ctrl+C 停止）")
    try:
        while True:
            cur_write = write_pos
            do_step = False
            if last_step + step_samples <= cur_write: do_step = True
            elif cur_write < last_step and cur_write + ring_len - last_step >= step_samples: do_step = True
            if not do_step:
                time.sleep(0.005); continue

            idx = (cur_write - ring_len) % ring_len
            buf = ring[idx:] if idx==0 else np.concatenate([ring[idx:], ring[:idx]], axis=0)
            last_step = cur_write

            # 前向 + EMA
            logits = forward_logits(buf)
            if ema_logits is None: ema_logits = logits
            else: ema_logits = args.ema_beta * ema_logits + (1 - args.ema_beta) * logits

            prob = torch.softmax(torch.tensor(ema_logits), dim=0).numpy()
            prob_hist.append(prob)
            avg_prob = np.mean(prob_hist, axis=0)
            order = np.argsort(-avg_prob)
            top_i, sec_i = int(order[0]), int(order[1])
            top_p, sec_p = float(avg_prob[top_i]), float(avg_prob[sec_i])
            top_name, sec_name = classes[top_i], classes[sec_i]
            margin = top_p - sec_p

            # 状态条
            lock_left = max(0, args.lock_sec - (tnow()-song_start_t)) if current_song else 0
            lock_txt = f" | lock {lock_left:4.1f}s" if current_song and lock_left>0 else ""
            bar = "="*int(top_p*20) + "-"*int((1-top_p)*20)
            print(f"\r[pred={top_name:8s}] p={top_p:0.2f} Δ={margin:0.2f} |{bar}|  cur={str(current_song):8s}{lock_txt}", end="")

            # ========== 未锁定：开始判定窗 ==========
            if current_song is None:
                # 开窗条件：非noises且 ≥阈值，或 margin≥起跳阈值
                if start_win_t0 is None:
                    if (top_name != "noises" and top_p >= args.conf_thresh) or (top_name != "noises" and margin >= args.margin_start):
                        start_win_t0 = tnow(); window_labels.clear(); consec_start_cnt = 0
                if start_win_t0 is not None:
                    # 记录窗帧
                    window_labels.append(top_name if top_name!="noises" and top_p>=args.conf_thresh else "low")

                    # 快速触发 A：margin 连胜
                    if top_name!="noises" and margin >= args.margin_start:
                        consec_start_cnt = consec_start_cnt + 1
                    else:
                        consec_start_cnt = 0
                    if consec_start_cnt >= args.consec_start:
                        current_song = top_name; song_start_t = tnow()
                        print(f"\n开始跳舞：{current_song}（margin连胜）")
                        # 重置所有窗/计数
                        start_win_t0=None; window_labels.clear()
                        stop_win_t0=None; stop_notcur_cnt=0; lead_consec=0; lead_opponent=None
                        prob_hist.clear()
                        continue

                    # 快速触发 B：连续过阈值
                    if top_name!="noises" and top_p >= args.conf_thresh:
                        recent = list(window_labels)[-args.consec_start:]
                        if len(recent)==args.consec_start and all(v==top_name for v in recent):
                            current_song = top_name; song_start_t = tnow()
                            print(f"\n开始跳舞：{current_song}（阈值连中）")
                            start_win_t0=None; window_labels.clear()
                            stop_win_t0=None; stop_notcur_cnt=0; lead_consec=0; lead_opponent=None
                            prob_hist.clear()
                            continue

                    # 3 秒到期：多数 ≥60% 强制开始
                    if tnow() - start_win_t0 >= args.decision_sec:
                        wins = [v for v in window_labels if v!="low" and v!="noises"]
                        if wins:
                            cand = max(set(wins), key=wins.count)
                            if wins.count(cand)/len(wins) >= 0.60:
                                current_song = cand; song_start_t = tnow()
                                print(f"\n开始跳舞：{current_song}（3s多数 {wins.count(cand)}/{len(wins)}）")
                        start_win_t0=None; window_labels.clear()
                continue

            # ========== 锁定期：不切歌/不停止 ==========
            if (tnow() - song_start_t) < args.lock_sec:
                stop_win_t0=None; window_labels.clear()
                continue

            # ========== 解锁：停止判定窗 ==========
            # 开窗条件：只要 top≠当前（不再等置信度高）
            if stop_win_t0 is None and top_name != current_song:
                stop_win_t0 = tnow(); window_labels.clear()
                stop_notcur_cnt = 0; lead_consec = 0; lead_opponent = None

            if stop_win_t0 is not None:
                # 记录窗帧（是否非当前）
                window_labels.append("other" if top_name != current_song else "cur")

                # 快速停 1：连续“非当前”（含 noises）
                if top_name != current_song: stop_notcur_cnt += 1
                else: stop_notcur_cnt = 0
                if stop_notcur_cnt >= args.stop_consec:
                    print(f"\n停止跳舞（连续{args.stop_consec}帧非当前，含noises）")
                    current_song=None; song_start_t=0.0
                    stop_win_t0=None; window_labels.clear()
                    prob_hist.clear(); time.sleep(args.cooldown); continue

                # 快速停 2：同一对手 margin 连续压制
                curr_i = classes.index(current_song); curr_p = float(avg_prob[curr_i])
                if top_name != current_song and (top_p - curr_p) >= args.margin_switch:
                    if lead_opponent == top_name: lead_consec += 1
                    else: lead_opponent = top_name; lead_consec = 1
                else:
                    lead_opponent = None; lead_consec = 0
                if lead_consec >= args.consec_switch:
                    print(f"\n停止跳舞（对手{lead_opponent} 连续{args.consec_switch}帧领先 ≥ {args.margin_switch:.2f}）")
                    current_song=None; song_start_t=0.0
                    stop_win_t0=None; window_labels.clear()
                    prob_hist.clear(); time.sleep(args.cooldown); continue

                # 3 秒到期：“非当前”占比 ≥60% 强制停止
                if tnow() - stop_win_t0 >= args.decision_sec:
                    total = len(window_labels); other_cnt = sum(1 for v in window_labels if v=="other")
                    if total>0 and other_cnt/total >= 0.60:
                        print(f"\n停止跳舞（3s多数非当前 {other_cnt}/{total}）")
                        current_song=None; song_start_t=0.0
                        prob_hist.clear()
                    stop_win_t0=None; window_labels.clear()
                    time.sleep(args.cooldown); continue

            time.sleep(0.005)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop(); stream.close()
        print("\n退出。")

if __name__ == "__main__":
    main()
