import numpy as np
import torch
import time

configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        spshot=20000,
        nclass=2,
        sshow=10,
    ),
    'stage2_cfg': dict(
        NUM_BRANCHES = 2,
        NUM_CHANNELS = [32, 64],
        NUM_BLOCKS = [4, 4],
    ),
    'stage3_cfg': dict(
        NUM_BRANCHES = 3,
        NUM_CHANNELS=[256, 512, 1024],
        NUM_BLOCKS=[4, 4, 4],
    ),
    'stage4_cfg': dict(
        NUM_BRANCHES = 4,
        NUM_BLOCKS = [4, 4, 4, 4],
        NUM_CHANNELS = [256, 512, 1024, 256],
    )
}
CFG = configurations


def computeTime(model, device='cuda'):
    inputs = []
    for i in range(4):
        input = torch.randn(1, 3, 448, 448)
        inputs.append(input.cuda())
    # inputs = torch.randn(1, 3, 448, 448)
    # inputs = inputs.cuda()
    if device == 'cuda':
        model = model.cuda()
        # inputs = inputs.cuda()

    model.eval()

    time_spent = []
    for idx in range(100):
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Avg execution time (ms): %.4f, FPS:%d' % (np.mean(time_spent), 1*1//np.mean(time_spent) * 4))
    return 1*1//np.mean(time_spent)


if __name__=="__main__":

    torch.backends.cudnn.benchmark = True

    from libs.networks import VideoModel, ImageModel
    model = VideoModel(output_stride=16, pretrained=True, cfg=CFG)

    computeTime(model)
