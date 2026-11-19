# 来源与贡献边界

作者负责个人电脑端模型训练、音乐识别、机器人调试与麦克风音频采集接入。模型在个人电脑训练后部署到树莓派，未将树莓派表述为训练设备。

- 音乐训练主线来自原music_dance/src/train_audio_classifier.py，数据切片来自同目录slice_audio_dataset.py。
- PC实时推理来自比赛部署包code/infer_pc_lockvote.py，机器人音乐控制来自最终材料中的dance.py。
- 姿态辅助代码源于TonyPi的Fall_and_Stand示例并经过项目适配；底层动作、舵机接口由Hiwonder提供。
- 编舞为队友贡献，仅列出脚本接口，不包含其实现或.d6a动作库。
- PyTorch、NumPy、SoundDevice、SoundFile、tqdm作为外部依赖使用，版权与许可属于各自项目。

本目录不为第三方SDK、厂商二进制库或队友资源另行授予许可。未统一添加MIT等许可，以免错误覆盖来源不同的材料。
