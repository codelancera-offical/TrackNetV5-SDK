from pathlib import Path

# ------------------- 1. 模型定义 (Model) -------------------
model = dict(
    type='WASB',
    backbone=dict(
        type='WASBHRNetBackbone',
        frames_in=3, # 对应 3 帧输入 [cite: 110]
        MODEL=dict(
            EXTRA=dict(
                STEM=dict(
                    INPLANES=64,
                    STRIDES=[1, 1]  # Modified Stem: 保持高分辨率 [cite: 104, 108]
                ),
                STAGE1=dict(
                    NUM_MODULES=1,
                    NUM_BRANCHES=1,
                    BLOCK='BOTTLENECK', # Stage 1 暴力语义提取 [cite: 106]
                    NUM_BLOCKS=[4],
                    NUM_CHANNELS=[64],
                    FUSE_METHOD='SUM'
                ),
                STAGE2=dict(
                    NUM_MODULES=1,
                    NUM_BRANCHES=2,
                    BLOCK='BASIC',
                    NUM_BLOCKS=[4, 4],
                    NUM_CHANNELS=[32, 64],
                    FUSE_METHOD='SUM'
                ),
                STAGE3=dict(
                    NUM_MODULES=4,
                    NUM_BRANCHES=3,
                    BLOCK='BASIC',
                    NUM_BLOCKS=[4, 4, 4],
                    NUM_CHANNELS=[32, 64, 128],
                    FUSE_METHOD='SUM'
                ),
                STAGE4=dict(
                    NUM_MODULES=3,
                    NUM_BRANCHES=4,
                    BLOCK='BASIC',
                    NUM_BLOCKS=[4, 4, 4, 4],
                    NUM_CHANNELS=[32, 64, 128, 256],
                    FUSE_METHOD='SUM' # 暴力特征交换 [cite: 81]
                )
            )
        )
    ),
    neck=None, # WASB 逻辑不需要独立 Neck
    head=dict( 
        type='TrackNetV2Head', # 沿用你 V2/V5 的 Head
        in_channels=32,       # 对应 Stage 4 高分辨率分支的通道数
        out_channels=3        # MIMO 输出 3 张热力图 [cite: 110]
    )
)

# ------------------- 2. 数据定义 (沿用 V5) -------------------
input_size = (288, 512)
original_size = (1080, 1920)
data_root = './data/loveall_tennis_gauss_heatmap'

pipeline = [
    dict(type='LoadMultiImagesFromPaths', to_rgb=True),
    dict(type='Resize', keys=['path_prev', 'path', 'path_next'], size=input_size),
    dict(type='ConcatChannels', keys=['path_prev', 'path', 'path_next'], output_key='image'),
    dict(type='LoadAndFormatMultiTargets', 
         keys=['gt_path_prev', 'gt_path', 'gt_path_next'], 
         output_key='target'),
    dict(type='Finalize',
         image_key='image',
         final_keys=['image', 'target', 'coords', 'visibility', 'original_info'])
]

data = dict(
    samples_per_gpu=4, # WASB 显存开销大，如果爆显存请调至 2
    workers_per_gpu=4,
    train=dict(
        type='TennisDataset',
        data_dir=data_root,
        csv_path=f'{data_root}/labels_context_train.csv',
        input_height=input_size[0],
        input_width=input_size[1],
        pipeline=pipeline
    ),
    val=dict(
        type='TennisDataset',
        data_dir=data_root,
        csv_path=f'{data_root}/labels_context_val.csv',
        input_height=input_size[0],
        input_width=input_size[1],
        pipeline=pipeline
    )
)

# ------------------- 3. 损失函数 (沿用 V5, 不使用 QFL) -------------------
loss = dict(
    type='TrackNetV2Loss' # 按照你的要求，直接用原有的 Loss
)

# ------------------- 4. 优化策略 (沿用 V5) -------------------
optimizer = dict(type='AdamW', lr=1e-4)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
lr_config = dict(
    policy='Step',
    step=[20, 25],
    gamma=0.1
)

# ------------------- 5. 评估与运行时 (沿用 V5) -------------------
evaluation = dict(
    interval=1,
    metric=dict(type='TrackNetV2Metric', min_dist=4, original_size=original_size)
)
total_epochs = 30
work_dir = f'./workdirs/{Path(__file__).stem}'
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')]
)
custom_hooks = [
    dict(type='ValidationVisualizerV2Hook', num_samples_to_save=100, original_size=original_size)
]
seed = 42
resume_from = None