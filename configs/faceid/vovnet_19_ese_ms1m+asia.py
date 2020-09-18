# model settings
model = dict(
    type='FaceID',
    backbone=dict(
        type='VoVNet',
        spec='V-19-eSE',
        num_stages=3,
        use_dcn = [False, False, False, False, False]
        ),
    neck=dict(
        type='GlobalConvolutionPooling',
        input_channel=768,
        last_channel=512,
        emb_size=128
        ),
    head=dict(
        type='LinearArcFaceHead',
        num_classes=86876,
        in_channels=128,
        margin=0.5,
        scale=64.0,
        t=0.8,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)
    ))

img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    to_rgb=True)

test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

test_dataset_type = 'FaceVerification'
test_root = '/home/acer/nfs-share/faceid_benchmark_112_112'
gpu_ids = [0,1,2,3]
distributed = True
data = dict(
    samples_per_gpu=512,
    workers_per_gpu=1,
    test=[
        dict(type=test_dataset_type, root=test_root, name='lfw', pipeline=test_pipeline),
        dict(type=test_dataset_type, root=test_root, name='cplfw', pipeline=test_pipeline),
        dict(type=test_dataset_type, root=test_root, name='agedb', pipeline=test_pipeline),
        dict(type=test_dataset_type, root=test_root, name='calfw', pipeline=test_pipeline),
        dict(type=test_dataset_type, root=test_root, name='cplfw_bad_landmark', pipeline=test_pipeline)
        ])

tensorrt = dict(
        input_size = (112,112),
        max_batch_size = 20,
        fp16_mode = False,
        save_name = f'vovnet_19_ese_ms1m+asia.trt.pth'
        )

work_dir = './work_dirs/vovnet_19_ese_ms1m+asia/'
