# -*- coding: utf-8 -*-
"""
六分类（renshi/dream/aoteman/amigo/wuren/noises）训练脚本
从 slices_multi/index.json 读取你刚才生成的 **全部切片**，并按“源文件”分组做 train/val/test 划分。
新增增强：时间拉伸、背景噪声混入、Mixup；保留增益/时移/高斯噪声/SpecAugment/LabelSmoothing/Dropout/加权采样等。
"""

import os, json, math, argparse, random, time
from collections import defaultdict, Counter

import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from tqdm import tqdm

# --------------------
# 全局
# --------------------
CLASSES = ["renshi", "dream", "aoteman", "amigo", "wuren", "noises"]
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASSES)}

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# --------------------
# I/O 与基础音频处理
# --------------------
def read_wav(path, target_sr=22050, mono=True):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1 and mono:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != target_sr:
        # 简易线性重采样
        x_old = np.linspace(0, 1, num=len(y), endpoint=False)
        new_len = int(round(len(y) * target_sr / sr))
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        sr = target_sr
    # 归一
    maxv = np.max(np.abs(y)) + 1e-9
    if maxv > 1.0: y = y / maxv
    return y, sr

def time_stretch(y, rate_min=0.95, rate_max=1.05):
    """不改变音高的高保真拉伸需要更重依赖；这里用近似法：整体重采样实现时域拉伸，再拉回原长度。"""
    rate = random.uniform(rate_min, rate_max)
    L = len(y)
    x_old = np.linspace(0, 1, num=L, endpoint=False)
    new_len = max(1, int(round(L / rate)))
    x_new = np.linspace(0, 1, num=new_len, endpoint=False)
    y_st = np.interp(x_new, x_old, y).astype(np.float32)
    # 拉回原长度
    x_old2 = np.linspace(0, 1, num=len(y_st), endpoint=False)
    x_new2 = np.linspace(0, 1, num=L, endpoint=False)
    y_ret = np.interp(x_new2, x_old2, y_st).astype(np.float32)
    return y_ret

def bg_mix(y, ybg, snr_db):
    """把背景 ybg 混入 y（两者长度对齐/循环），按目标 SNR。"""
    L = len(y)
    if len(ybg) < L:
        reps = (L + len(ybg) - 1) // len(ybg)
        ybg = np.tile(ybg, reps)
    ybg = ybg[:L]
    sig_pow = np.mean(y**2) + 1e-12
    noise_pow = np.mean(ybg**2) + 1e-12
    if noise_pow < 1e-12:
        return y
    scale = math.sqrt(sig_pow / (noise_pow * (10**(snr_db/10.0))))
    ym = y + ybg * scale
    return np.clip(ym, -1.0, 1.0)

def wav_augment(
    y, sr,
    gain_db_range=(-6, 6),
    shift_max_frac=0.1,
    noise_snr_db=(15, 35),
    stretch_prob=0.5,
    stretch_range=(0.95, 1.05),
):
    """波形域增强：增益、时移、加性高斯噪声、随机时间拉伸"""
    # 增益
    g = 10 ** (random.uniform(*gain_db_range) / 20.0)
    y = np.clip(y * g, -1.0, 1.0)

    # 时移
    if shift_max_frac > 0:
        max_shift = int(len(y) * shift_max_frac)
        if max_shift > 0:
            shift = random.randint(-max_shift, max_shift)
            y = np.roll(y, shift)

    # 随机时间拉伸
    if random.random() < stretch_prob:
        y = time_stretch(y, rate_min=stretch_range[0], rate_max=stretch_range[1])

    # 加性高斯噪声（指定 SNR）
    if noise_snr_db is not None:
        snr_db = random.uniform(*noise_snr_db)
        sig_pow = np.mean(y**2) + 1e-12
        noise_pow = sig_pow / (10 ** (snr_db / 10.0))
        n = np.random.randn(len(y)).astype(np.float32) * math.sqrt(noise_pow)
        y = np.clip(y + n, -1.0, 1.0)

    return y

def melspec(y, sr=22050, n_fft=1024, hop_length=256, n_mels=128, fmin=20.0, fmax=None, eps=1e-10):
    # STFT
    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (len(y) - n_fft) // hop_length if len(y) >= n_fft else 0
    if frames <= 0:
        pad_len = n_fft - len(y) + 1
        y = np.pad(y, (0, pad_len))
        frames = 1 + (len(y) - n_fft) // hop_length
    spec = []
    for i in range(frames):
        s = i * hop_length
        frame = y[s:s+n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        frame = frame * win
        mag = np.abs(np.fft.rfft(frame, n=n_fft))
        spec.append(mag)
    spec = np.stack(spec, axis=1)  # [freq, time]
    if fmax is None: fmax = sr/2

    def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz/700.0)
    def mel_to_hz(m): return 700.0 * (10**(m/2595.0) - 1.0)

    m_min, m_max = hz_to_mel(fmin), hz_to_mel(fmax)
    m_pts = np.linspace(m_min, m_max, n_mels+2)
    f_pts = mel_to_hz(m_pts)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)
    bins = np.searchsorted(fft_freqs, f_pts)
    bins = np.clip(bins, 0, spec.shape[0]-1)

    fb = np.zeros((n_mels, spec.shape[0]), dtype=np.float32)
    for m in range(1, n_mels+1):
        l, c, r = bins[m-1], bins[m], bins[m+1]
        if l==c or c==r: continue
        for k in range(l, c): fb[m-1, k] = (k-l) / (c-l + 1e-12)
        for k in range(c, r): fb[m-1, k] = (r-k) / (r-c + 1e-12)

    mel = fb @ (spec**2)
    mel = np.log(mel + eps).astype(np.float32)  # [n_mels, time]
    return mel

def spec_augment(spec, max_mask_t=24, max_mask_f=12, num_t=2, num_f=2):
    S = spec.copy()
    n_mels, T = S.shape
    mean = S.mean()
    for _ in range(num_t):
        w = random.randint(0, max_mask_t)
        if w>0 and T>w:
            t0 = random.randint(0, T-w)
            S[:, t0:t0+w] = mean
    for _ in range(num_f):
        w = random.randint(0, max_mask_f)
        if w>0 and n_mels>w:
            f0 = random.randint(0, n_mels-w)
            S[f0:f0+w, :] = mean
    return S

def crop_or_pad_time(spec, target_T, train=True):
    n_mels, T = spec.shape
    if T == target_T: return spec
    if T > target_T:
        start = random.randint(0, T-target_T) if train else (T-target_T)//2
        return spec[:, start:start+target_T]
    pad = target_T - T
    l = pad//2; r = pad - l
    return np.pad(spec, ((0,0),(l,r)), mode="constant", constant_values=spec.min())

# --------------------
# 索引与划分
# --------------------
def load_index(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    rows = [r for r in rows if r.get("class") in CLASS_TO_IDX and os.path.exists(r.get("out",""))]
    return rows

def training_noise_rows(train_items):
    return [row for row in train_items if row["class"] == "noises"]


def stratified_split_by_src(rows, test_size=0.1, val_size=0.1, seed=42):
    by_class_src = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_class_src[r["class"]][r["src"]].append(r)

    train, val, test = [], [], []
    rng = random.Random(seed)
    for cls, d in by_class_src.items():
        src_list = list(d.keys()); rng.shuffle(src_list)
        n = len(src_list)
        n_test = max(1, int(round(n * test_size))) if n >= 5 else max(1, n//5 or 1)
        n_val  = max(1, int(round(n * val_size)))  if n >= 5 else max(1, (n - n_test)//4 or 1)
        n_train = max(1, n - n_test - n_val)
        src_train = set(src_list[:n_train])
        src_val   = set(src_list[n_train:n_train+n_val])
        src_test  = set(src_list[n_train+n_val:])
        for s in src_train: train.extend(d[s])
        for s in src_val:   val.extend(d[s])
        for s in src_test:  test.extend(d[s])
    return train, val, test

# --------------------
# 数据集（含背景噪声库）
# --------------------
class MelDataset(Dataset):
    def __init__(self, items, all_noises_rows=None, target_sr=22050,
                 n_fft=1024, hop_length=256, n_mels=128,
                 target_frames=256, train=True, augment=True,
                 label_smoothing=0.0, bgmix_p=0.5, bgmix_snr=(5, 20),
                 stretch_range=(0.95, 1.05)):
        self.items = items
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop = hop_length
        self.n_mels = n_mels
        self.target_frames = target_frames
        self.train = train
        self.augment = augment and train
        self.ls = label_smoothing
        self.bgmix_p = bgmix_p
        self.bgmix_snr = bgmix_snr
        self.stretch_range = stretch_range

        # 准备噪声素材库（只存路径，按需读取）
        self.noise_paths = []
        if all_noises_rows is not None:
            self.noise_paths = [r["out"] for r in all_noises_rows if os.path.exists(r["out"])]

    def __len__(self): return len(self.items)

    def _rand_noise(self):
        if not self.noise_paths: return None
        return random.choice(self.noise_paths)

    def __getitem__(self, idx):
        r = self.items[idx]
        path = r["out"]; cls = r["class"]
        y, sr = read_wav(path, self.target_sr)

        if self.augment:
            # 先做基础波形增强
            y = wav_augment(
                y, sr,
                gain_db_range=(-6, 6),
                shift_max_frac=0.1,
                noise_snr_db=(18, 35),
                stretch_prob=0.6,
                stretch_range=self.stretch_range
            )
            # 再概率性加入“noises 类背景”（仅非 noises 类时效果更明显，但这里统一处理）
            if random.random() < self.bgmix_p and len(self.noise_paths) > 0:
                npth = self._rand_noise()
                try:
                    yb, _ = read_wav(npth, self.target_sr)
                    snr = random.uniform(*self.bgmix_snr)
                    y = bg_mix(y, yb, snr_db=snr)
                except Exception:
                    pass

        S = melspec(y, sr, n_fft=self.n_fft, hop_length=self.hop, n_mels=self.n_mels)
        S = crop_or_pad_time(S, self.target_frames, train=self.train)
        if self.augment:
            S = spec_augment(S, max_mask_t=32, max_mask_f=16, num_t=2, num_f=2)

        # 标准化
        mu, sigma = S.mean(), S.std() + 1e-6
        S = (S - mu) / sigma

        x = torch.from_numpy(S).unsqueeze(0)
        y_idx = CLASS_TO_IDX[cls]

        if self.ls > 0 and self.train:
            t = torch.full((len(CLASSES),), self.ls / (len(CLASSES)-1), dtype=torch.float32)
            t[y_idx] = 1.0 - self.ls
            return x, t
        else:
            return x, torch.tensor(y_idx, dtype=torch.long)

# --------------------
# 模型
# --------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, pool=True, p_drop=0.1):
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
        x = self.conv(x); x = self.pool(x); x = self.drop(x); return x

class SmallResNet(nn.Module):
    def __init__(self, n_classes=6, p_drop=0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ConvBlock(32, 64,  pool=True, p_drop=p_drop)
        self.layer2 = ConvBlock(64, 128, pool=True, p_drop=p_drop)
        self.layer3 = ConvBlock(128, 192, pool=True, p_drop=p_drop)
        self.layer4 = ConvBlock(192, 256, pool=True, p_drop=p_drop)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x); x = self.head(x); return x

# --------------------
# Mixup
# --------------------
def do_mixup(x, y, alpha=0.2):
    if alpha <= 0: return x, y
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    # y 可能是 hard labels（LongTensor）或 soft labels（float向量）
    if y.ndim == 1:
        y_onehot = torch.zeros((y.size(0), len(CLASSES)), device=x.device)
        y_onehot.scatter_(1, y.view(-1,1), 1.0)
        y_mix = lam * y_onehot + (1 - lam) * y_onehot[perm]
    else:
        y_mix = lam * y + (1 - lam) * y[perm]
    return x_mix, y_mix

def loss_from_logits(logits, target):
    if target.ndim == 2:  # soft labels
        logp = F.log_softmax(logits, dim=1)
        return -(target * logp).sum(dim=1).mean()
    return F.cross_entropy(logits, target)

def accuracy(logits, target):
    if target.ndim == 2:
        pred = logits.argmax(dim=1)
        hard = target.argmax(dim=1)
        return (pred == hard).float().mean().item()
    else:
        pred = logits.argmax(dim=1)
        return (pred == target).float().mean().item()

# --------------------
# 训练与评估
# --------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, mixup_alpha=0.0):
    model.train()
    loss_m, acc_m = 0.0, 0.0
    for x, y in tqdm(loader, ncols=80, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        if mixup_alpha > 0:
            x, y = do_mixup(x, y, alpha=mixup_alpha)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_from_logits(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        if scheduler is not None: scheduler.step()

        loss_m += loss.item() * x.size(0)
        acc_m += accuracy(logits.detach(), y) * x.size(0)
    n = len(loader.dataset)
    return loss_m/n, acc_m/n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_m, acc_m = 0.0, 0.0
    for x, y in tqdm(loader, ncols=80, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_from_logits(logits, y)
        loss_m += loss.item() * x.size(0)
        acc_m += accuracy(logits, y) * x.size(0)
    n = len(loader.dataset)
    return loss_m/n, acc_m/n

# --------------------
# 主函数
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--mel", type=int, default=128)
    ap.add_argument("--nfft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--scheduler", type=str, default="onecycle", choices=["onecycle","cosine","none"])
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--outdir", type=str, default="models")
    # 新增增强控制
    ap.add_argument("--bgmix_p", type=float, default=0.5, help="背景噪声混入概率")
    ap.add_argument("--bgmix_snr_min", type=float, default=5.0)
    ap.add_argument("--bgmix_snr_max", type=float, default=20.0)
    ap.add_argument("--mixup_alpha", type=float, default=0.0, help=">0 启用 mixup，推荐 0.2~0.4")
    ap.add_argument("--stretch_min", type=float, default=0.95)
    ap.add_argument("--stretch_max", type=float, default=1.05)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # 读取索引 & 划分
    rows = load_index(args.index)
    train_items, val_items, test_items = stratified_split_by_src(rows, test_size=0.1, val_size=0.1, seed=args.seed)

    if not train_items or not val_items or not test_items:
        raise ValueError("训练/验证/测试集均需非空；请为各类补充独立源录音，而非复制切片。")

    # 噪声增强只使用训练集，避免验证/测试音频进入训练增强。
    noises_rows = training_noise_rows(train_items)

    # Dataset/DataLoader
    ds_train = MelDataset(
        train_items, all_noises_rows=noises_rows,
        n_mels=args.mel, n_fft=args.nfft, hop_length=args.hop, target_frames=args.frames,
        train=True, augment=True, label_smoothing=args.label_smoothing,
        bgmix_p=args.bgmix_p, bgmix_snr=(args.bgmix_snr_min, args.bgmix_snr_max),
        stretch_range=(args.stretch_min, args.stretch_max),
    )
    ds_val = MelDataset(
        val_items, all_noises_rows=noises_rows,
        n_mels=args.mel, n_fft=args.nfft, hop_length=args.hop, target_frames=args.frames,
        train=False, augment=False, label_smoothing=0.0,
    )
    ds_test = MelDataset(
        test_items, all_noises_rows=noises_rows,
        n_mels=args.mel, n_fft=args.nfft, hop_length=args.hop, target_frames=args.frames,
        train=False, augment=False, label_smoothing=0.0,
    )

    # 类别不均衡 -> 加权采样
    cls_counts = Counter([CLASS_TO_IDX[it["class"]] for it in train_items])
    cls_weight = {c: len(train_items) / (len(CLASSES) * max(1, cls_counts[c])) for c in cls_counts}
    sample_weights = [cls_weight[CLASS_TO_IDX[it["class"]]] for it in train_items]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    dl_train = DataLoader(ds_train, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False,   num_workers=args.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch, shuffle=False,   num_workers=args.num_workers, pin_memory=True)

    # 模型/优化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallResNet(n_classes=len(CLASSES), p_drop=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(dl_train))
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    else:
        scheduler = None

    # 训练
    best_val_acc, bad_epochs = 0.0, 0
    best_path = os.path.join(args.outdir, f"best_model_{int(time.time())}.pt")
    patience = 8

    for epoch in range(1, args.epochs+1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, scheduler, device, mixup_alpha=args.mixup_alpha)
        val_loss, val_acc = evaluate(model, dl_val, device)
        print(f"train: loss {tr_loss:.4f} | acc {tr_acc:.4f}")
        print(f"valid: loss {val_loss:.4f} | acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": CLASSES, "args": vars(args)}, best_path)
            print(f"[BEST] 保存最佳模型 -> {best_path}")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"验证集连续 {patience} 次未提升，提前停止。")
                break

    # 测试
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print("已加载最佳模型进行测试评估。")
    test_loss, test_acc = evaluate(model, dl_test, device)
    print(f"\n[TEST] loss {test_loss:.4f} | acc {test_acc:.4f}")
    print("完成。")

if __name__ == "__main__":
    main()
