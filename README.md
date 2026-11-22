# 机器人音乐识别与动作控制

## On-Device Music Recognition for a Dancing Robot

This competition project uses a microphone and a Raspberry Pi 5 to recognize
five songs and trigger the corresponding dance routine on a TonyPi robot.
A sixth class represents background noise.

### My contribution

- **Model training:** I prepared the song and noise recordings, split the audio
  into clips, and trained a PyTorch CNN on log-Mel spectrograms on my PC.
- **Streaming inference:** I deployed the classifier to a Raspberry Pi 5 and
  smoothed predictions across audio windows before triggering a routine.
- **Robot integration:** I connected the recognition output to the dance
  scripts and handled routine interruption and the robot's return to its
  neutral pose. My teammates created the choreography, and the robot SDK
  was supplied by the manufacturer.

### What you can try

With a microphone and PortAudio configured, `inference/listen.py` runs the
classifier and prints its predictions. It can be used independently of the
robot. The hardware-independent tests are in `tests/`.

This repository was assembled from the competition code. Selected regression
tests have been run; end-to-end testing of this release on the robot is
still pending.

基于树莓派5与TonyPi平台的比赛项目，通过自行接入的麦克风采集音频，完成音乐类别识别与机器人动作调用。**模型在个人电脑上训练，树莓派运行推理与控制程序。** 本仓库由比赛材料整理，保留可解释的核心实现，并标明硬件与队友模块的边界。

## 我的工作

- 模型训练：组织歌曲与噪声数据、音频切片、特征处理、卷积分类网络训练与调参。
- 音乐识别：流式音频采集、对数Mel频谱、推理结果平滑、歌曲判决和动作触发。
- 机器人调试：模型部署、音频设备配置、SDK调用、动作进程中断与归中、启动环境排查。

舞蹈编排由队友完成。本仓库展示识别到编舞脚本的调用接口，不包含队友的编舞脚本及动作库。机器人底层SDK来自厂商，未声称独立实现底层驱动。

## 系统流程

```mermaid
flowchart LR
    A[个人电脑：录音与切片] --> B[Mel特征与CNN训练]
    B --> C[PyTorch模型权重]
    C --> D[树莓派：实时音乐识别]
    D --> E[判决与动作进程控制]
    E --> F[外部编舞脚本与机器人SDK]
    G[自行接入的麦克风] --> D
```

音乐分类包括五首目标曲目 `renshi / dream / aoteman / amigo / wuren` 和 `noises` 噪声类。模型输入为对数Mel频谱，网络由多级卷积块与分类头组成；源码沿用历史类名SmallResNet，但不将其称为标准残差网络。

## 目录

| 路径 | 用途 |
| --- | --- |
| `training/prepare_data.py` | 在个人电脑生成多窗口切片与索引 |
| `training/train.py` | 在个人电脑训练六分类模型 |
| `inference/listen.py` | 麦克风实时识别，只打印结果，不控制机器人 |
| `robot/music_control.py` | 树莓派音乐识别及外部编舞脚本调用 |
| `models/music_classifier.pt` | 比赛材料中已有的模型，未因本次整理而重新训练 |
| `docs/deployment.md` | 数据、运行命令和机器人外部依赖 |
| `docs/engineering-notes.md` | 调试经验、整理修改及验证边界 |
| `THIRD_PARTY.md` | 模块来源与贡献说明 |

个人电脑上的原始训练工程为 `music_dance` 和 `music-dance` 两个目录。前者含本仓库采用的PyTorch训练主线；后者保留早期特征处理与训练实验，作为迭代历史，没有混入最终默认依赖。

## 开始使用

先在个人电脑创建独立Python环境，再安装依赖。`requirements.txt` 是按导入整理的依赖清单，不是已验证的跨平台环境锁定文件。

```bash
python -m pip install -r requirements.txt
python inference/listen.py --ckpt models/music_classifier.pt
```

实时识别需要可用麦克风及PortAudio支持。没有音频设备时，可以先阅读训练代码和运行不依赖硬件的回归检查：

```bash
python -m unittest discover -s tests -v
```

准备数据、重新训练、麦克风配置与树莓派部署见 [部署说明](docs/deployment.md)。

## 当前验证状态

此版本是经过整理的比赛代码展示版，完成Python语法检查及针对数据划分、噪声增强与编舞进程接口的局部回归测试，尚未在机器人上完成整理后的端到端复测。

未附公开训练集、队友动作库、厂商SDK或演示录像。没有公布未经复核的准确率、延迟或噪声鲁棒性指标。`decision_sec`是判决参数，不是实测响应时间；默认45秒歌曲锁定会影响停止与切歌响应，详见工程说明。

**训练整理修正：** 原版背景噪声增强从全部样本取材，展示版改为只使用训练集噪声。随附权重仍是修正前已有模型，因此不能据此宣称该权重的评估已消除相关数据泄漏风险。
