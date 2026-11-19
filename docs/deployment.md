# 个人电脑训练与树莓派部署

以下命令从仓库根目录执行。示例需要个人数据、可用设备和正确的Python环境；本次整理没有重新安装整套环境或执行硬件联调。

## 个人电脑：数据与训练

原始项目分别位于个人电脑的music_dance和music-dance目录。当前主线采用music_dance中的PyTorch训练代码，而非树莓派训练。建立以下数据结构，每个类别应有足够的独立源录音，不能将同一原始音频的切片伪装成多个独立源：

```text
data/
  songs/
    renshi/*.wav
    dream/*.wav
    aoteman/*.wav
    amigo/*.wav
    wuren/*.wav
  noises/*.wav
```

```bash
python training/prepare_data.py --root data
python training/train.py --index data/slices_multi/index.json --outdir outputs/models
```

切片覆盖1秒、3秒、首10秒、5秒步长2秒、50秒步长5秒方案，索引记录源音频与切片路径。索引在本机生成后含绝对路径，迁移数据位置应重新生成。源文件路径分组能隔离同一文件的切片，但不能自动识别不同文件中重复录制或复制的相同内容；实验前仍需人工核对独立录音。

训练使用源文件分组划分、加权采样及噪声等增强；Mixup为可配置选项，默认值不启用。训练脚本会拒绝整体为空的训练/验证/测试集，但仍需检查每个类别的来源数量和分布。历史权重不对应本次修正后重新训练的结果。

## 个人电脑：识别预览

```bash
python inference/listen.py --ckpt models/music_classifier.pt
```

`--device`指定麦克风设备ID；`--sr`为设备采集采样率，输入随后重采样到模型使用的22050 Hz。优先使用设备支持的采样率。此程序只输出识别状态。

## 树莓派：音乐到动作

机器人运行环境必须已提供可导入的`hiwonder`包及其舵机/动作配置。使用自行接入的麦克风，并安装匹配树莓派平台的PyTorch和音频依赖，不要从PC复制虚拟环境。

队友需要另行提供以下编舞脚本及其引用的动作组：

| 识别类别 | 外部脚本 |
| --- | --- |
| renshi | renshi_actions.py |
| dream | Huawei_actions.py |
| aoteman | aoteman_actions.py |
| amigo | amigo_actions.py |
| wuren | wuren_actions.py |

```bash
python robot/music_control.py --ckpt models/music_classifier.pt --scripts-dir /home/pi/deploy_pi5_v1/jiaoben
```

`--scripts-dir`仅指定Python编舞脚本目录；`.d6a`须放入机器人SDK实际使用的动作目录。编舞脚本接受标准输入`1`停止，或SIGTERM终止；主程序负责随后归中。必须能调用`0_Stand`，姿态辅助还依赖IMU及站起动作。外部编舞不在仓库内，完整动作执行不能独立复现。

历史外部amigo脚本中KunPengFlyFly与文件KunpengFlyFly大小写不一致，须在团队部署副本中统一。仓库没有修改队友原件。

默认`--lock_sec 45`沿用比赛代码，锁定期内不按音乐类别变化停舞。不要把默认配置理解为“音乐停止3秒内必停”；参数需要在实际硬件与音频上测量后调整。

## 自启动

自启动应在手动启动验证后再配置，服务须使用安装依赖的同一解释器及正确工作目录；比赛记录中的用户服务命令不作为通用安装脚本直接执行。
