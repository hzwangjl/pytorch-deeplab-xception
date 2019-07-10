import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from PIL import Image
# from utils.metrics import IoUMetric

from modeling.deeplab import *

def data_process(x):
    x = np.array(x, dtype='float32') / 255
    x1 = np.expand_dims(x, 0)
    x = np.transpose(x1, (0, 3, 1, 2))
    x = torch.from_numpy(x)

    mean = [0.517446, 0.360147, 0.310427]
    std = [0.061526, 0.049087, 0.041330]
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    return x.sub_(mean[:, None, None]).div_(std[:, None, None])


if __name__ == '__main__':
    Vis = True
    SaveTxt = False
    nb_classes = 3

    # model = torch.load('./models/model_best.pth.tar')
    
    model = DeepLab(num_classes=3, backbone="mobilenet")
    model.cuda()
    resume_model = torch.load('./models/model_best.pth.tar')
    model.load_state_dict(resume_model["state_dict"])

    # cudnn.benchmark = True

    model.eval()

    #img_path = 'training_data/val/imgs/'
    img_path = '/home/wangjialei/teeth_dataset/output_save/'
    mask_path = 'training_data/val/masks/'


    img_folder = os.listdir(img_path)
    Ltime = []
    tTime_start = time.time()

    mode = "test"
    if mode == "test":
        for iter, img_file in enumerate(img_folder):
            if iter % 500 == 0:
                print("### proc ",iter)
            img_name = img_path + img_file
            image = Image.open(img_name)

            x = data_process(image)     
            start_time = time.time()
            y = model(x.cuda().float())
            y = torch.nn.Softmax2d()(y)
            numpy_y = y.cpu().detach().numpy()[0, ...]
            end_time = time.time()
            Ltime.append((end_time - start_time) * 1000)
            y1 = np.argmax(numpy_y, axis=0)
            if SaveTxt:
                os.makedirs("save_txt", exist_ok=True)
                np.savetxt((os.path.join("save_txt", img_file.split('.')[0] + '.txt')), y1, fmt='%d')
            if Vis:
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.subplot(1, 2, 2)
                plt.imshow(y1)
                save_file = "save_fig_test"
                os.makedirs(save_file, exist_ok=True)
                plt.savefig(os.path.join(save_file, img_file.split('.')[0] + '.png'))
                # plt.show()
        tTime_end = time.time()
        print("Mean Time:{:.2f} ms".format(sum(Ltime[1:]) / (len(Ltime) - 1)))
        print("Total Mean Time:{:.2f} ms".format((tTime_end - tTime_start) * 1000 / (len(Ltime) - 1)))

    """
    elif mode == "val":
        metrics = IoUMetric(eps=1., activation="softmax2d")
        for iter, img_file in enumerate(img_folder):
            if img_file.endswith(('.jpg','.png')):
                img_name = img_path + img_file
                mask_name = mask_path + img_file.split('.')[0] + '.npy'
                image = Image.open(img_name)
                
                mask = np.load(mask_name)
                mask[mask > nb_classes - 1] = 0
                # mask_img = mask_to_image(mask)
                mask_gt = mask.reshape((1, mask.shape[0], mask.shape[1]))
                mask_gt = torch.from_numpy(mask_gt)

                x = data_process(image)
                start_time = time.time()
                y = model(x.cuda().float())
                end_time = time.time()
                Ltime.append((end_time - start_time) * 1000)

                iou = metrics(y, mask_gt)
                iou = iou.cpu().numpy()
                if SaveTxt:
                    os.makedirs("save_txt", exist_ok=True)
                    np.savetxt((os.path.join("save_txt", img_file.split('.')[0] + '.txt')), y1, fmt='%d')

                if Vis and iou < 0.8:
                    plt.subplot(2, 2, 1)
                    plt.annotate('image', xy=(250, -1))
                    plt.imshow(image)

                    plt.subplot(2, 2, 2)
                    plt.annotate('groundtruth', xy=(250, -1))
                    plt.imshow(mask)

                    plt.subplot(2, 2, 3) 
                    plt.annotate('pred: ' + str(iou), xy=(250, -1))
                    y = torch.nn.Softmax2d()(y)
                    numpy_y = y.cpu().detach().numpy()[0, ...]
                    y1 = np.argmax(numpy_y, axis=0)
                    plt.imshow(y1)

                    save_file = "save_fig_val"
                    os.makedirs(save_file, exist_ok=True)
                    plt.savefig(os.path.join(save_file, img_file.split('.')[0] + '.png'))
                    # plt.show()
            tTime_end = time.time()
            print("Forward Mean Time:{:.2f} ms".format(sum(Ltime[1:]) / (len(Ltime) - 1)))
            print("Total Mean Time:{:.2f} ms".format((tTime_end - tTime_start) * 1000 / (len(Ltime) - 1)))
    """
