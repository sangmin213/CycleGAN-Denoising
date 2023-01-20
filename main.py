import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import itertools

from data import horse2zebraDataset, AAPMDataset
from network import UNetGenerator, Discriminator, ResNetGenerator
from utils import set_requires_grad, weights_init_normal, LambdaLR

from tqdm import tqdm
from torchvision.utils import save_image
from torchmetrics import PeakSignalNoiseRatio as PSNR
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
psnr = PSNR().to(device)

def train():
    # net_G_x = ResNetGenerator(in_channels=1,out_channels=1,n_block=6).to(device)
    # net_G_y = ResNetGenerator(in_channels=1,out_channels=1,n_block=6).to(device)
    net_G_x = UNetGenerator(in_channel=1, out_channel=1).to(device)
    net_G_y = UNetGenerator(in_channel=1, out_channel=1).to(device)
    net_D_x = Discriminator(in_channel=1).to(device)
    net_D_y = Discriminator(in_channel=1).to(device)

    net_G_x.apply(weights_init_normal)
    net_G_y.apply(weights_init_normal)
    net_D_x.apply(weights_init_normal)
    net_D_y.apply(weights_init_normal)

    epochs = 70

    criterionGAN = nn.MSELoss() # more stable than BCE ( LSGAN의 loss에서 착안 ) 
    criterionCycle = nn.L1Loss()

    optimizer_D = optim.Adam(itertools.chain(net_D_x.parameters(), net_D_y.parameters()),lr=0.0002, betas=(0.5,0.999))
    optimizer_G = optim.Adam(itertools.chain(net_G_x.parameters(),net_G_y.parameters()), lr=0.0002, betas=(0.5,0.999))
    # scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D,lr_lambda=LambdaLR(epochs).step)
    # scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G,lr_lambda=LambdaLR(epochs).step)
    # D의 optimizer는 따로 따로 하는 게 맞는 것 같은데
    # G의 optimizer는 Cycle loss에서 두 generator에 통합적으로 적용되어야 하기 때문에,
    # G_x, G_y, G(cycle loss용) 3가지의 optimizer를 사용해버리면 G와 G_x(or G_y)간의 호환이 안되기 때문에
    # 학습에 어려움이 생길 수 있을 것 같다.

    # dataset = horse2zebraDataset(imgSize=256)
    # dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)

    dataset = AAPMDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

    epoch_loss_G_x_list = []
    epoch_loss_D_x_list = []
    psnr_FakeX_ImgX = []
    psnr_FakeY_ImgY = []
    psnr_ImgX_ImgY = []
    for epoch in range(epochs):
        print("-"*30)
        print(f"Epoch: {epoch+1}/{epochs}")

        epoch_loss_D_x = 0.0 
        epoch_loss_D_y = 0.0
        epoch_loss_G_x = 0.0
        epoch_loss_G_y = 0.0
        epoch_loss_cycle = 0.0

        net_G_x.train()
        net_G_y.train()

        for idx, batch in enumerate(tqdm(dataloader)):
            imgX, imgY = batch
            imgX = imgX.to(device)
            imgY = imgY.to(device)

            # < train Discriminator > ####################
            # set_requires_grad([net_D_x, net_D_y], True)
            # net_D_x.train()
            # net_D_y.train()
            # net_G_x.eval() # discriminator 학습 시에는 generator가 fixed 해야함
            # net_G_y.eval()
            
            optimizer_D.zero_grad()
            ''' x -> y '''
            # train Discriminator with real data 
            output = net_D_y(imgY)
            real_label = torch.ones_like(output, device=device)
            lossD_y_real = criterionGAN(output, real_label) # y 판별자의 real 판별 loss
            
            # train Discriminator with fake data 
            fakeY = net_G_y(imgX)
            output = net_D_y(fakeY.detach())
            fake_label = torch.zeros_like(output, device=device)
            lossD_y_fake = criterionGAN(output, fake_label) # y 판별자의 fake 판별 loss

            lossD_y = (lossD_y_real + lossD_y_fake) * 0.5
            lossD_y.backward()

            ''' y -> x '''
            # train Discriminator with real data 
            output = net_D_x(imgX)
            lossD_x_real = criterionGAN(output, real_label)

            # train Discriminator with fake data 
            fakeX = net_G_x(imgY)
            output = net_D_x(fakeX.detach())
            lossD_x_fake = criterionGAN(output, fake_label)

            lossD_x = (lossD_x_real + lossD_x_fake) * 0.5
            lossD_x.backward()

            optimizer_D.step()

            epoch_loss_D_x += lossD_x
            epoch_loss_D_y += lossD_y

            
            # < train Generator > ########################
            for i in range(3):
                # set_requires_grad([net_D_x, net_D_y], False)
                # net_D_x.eval() # generator 학습 시에는 discriminator가 fixed 해야함
                # net_D_y.eval()
                # net_G_x.train()
                # net_G_y.train()

                optimizer_G.zero_grad()

                ''' x -> y '''
                fakeY = net_G_y(imgX)
                output = net_D_y(fakeY)
                idtY = net_G_y(imgY)
                loss_identity_y = criterionCycle(idtY, imgY)
                lossG_y = criterionGAN(output, real_label) # y 생성자의 loss < 여기서 이미 접근한 loss를 재접근해서 문제가 발생한다는 건데 | 동일한 입력에 대한 동일한 신경망의 결과에 대해 loss를 구했음. 이에 대해 EBP를 수행하였음.
                ''' y -> x '''
                fakeX = net_G_x(imgY)
                output = net_D_x(fakeX)
                idtX = net_G_x(imgX)
                loss_identity_x = criterionCycle(idtX,imgX)
                lossG_x = criterionGAN(output, real_label) # x 생성자의 loss
                ''' cycle loss '''
                ## |G_x(G_y(x)) - x|
                lossC_x = criterionCycle(net_G_x(fakeY),imgX) # 생성된 y로 x 생성하기
                ## |G_y(G_x(y)) - y|
                lossC_y = criterionCycle(net_G_y(fakeX),imgY) # 생성된 x로 y 생성하기

                loss_cycle = lossC_x + lossC_y
                loss_identity = loss_identity_x + loss_identity_y

                lambda_c = 10.0 # from 논문 implementation chapter
                lambda_idt = 5.0
                lossG = lossG_x + lossG_y + lambda_c * loss_cycle + lambda_idt * loss_identity

                lossG.backward() # 순수하게 생성자의 loss만 이용했고, loss로 생성자를 학습함
                optimizer_G.step()


            # loss 저장
            epoch_loss_G_x += lossG_x
            epoch_loss_G_y += lossG_y
            epoch_loss_cycle += loss_cycle

            if idx % 300 == 0:
                print(f"[lossG_f: {lossG_x} | lossG_q: {lossG_y} | loss_cycle: {loss_cycle} | lossD_f: {lossD_x} | lossD_q: {lossD_y}]")
                save_image(dataset.train_transform(imgX), f"./result_imgs/aapm/imgF{epoch}_{idx}.png")
                save_image(dataset.train_transform(imgY), f"./result_imgs/aapm/imgQ{epoch}_{idx}.png")
                save_image(dataset.train_transform(fakeX), f"./result_imgs/aapm/fakeF{epoch}_{idx}.png")
                save_image(dataset.train_transform(fakeY), f"./result_imgs/aapm/fakeQ{epoch}_{idx}.png")
                save_image(dataset.train_transform(fakeX-imgY), f"./result_imgs/aapm/noise{epoch}_{idx}.png")

        # scheduler_D.step()
        # scheduler_G.step()

        data_len = len(dataset) // 4 # divide by batch size
        epoch_loss_D_x /= data_len
        epoch_loss_D_y /= data_len
        epoch_loss_G_x /= data_len
        epoch_loss_G_y /= data_len
        epoch_loss_cycle /= data_len

        print(f"[D_f loss: {epoch_loss_D_x} | D_q loss: {epoch_loss_D_y} | G_f loss: {epoch_loss_G_x} | G_q loss: {epoch_loss_G_y} | Cycle loss: {epoch_loss_cycle}]")
        epoch_loss_G_x_list.append(epoch_loss_G_x.item())
        epoch_loss_D_x_list.append(epoch_loss_D_x.item())

        #eval
        net_G_x.eval()
        net_G_y.eval()
        # generate
        tmp = np.random.randint(0,3000)
        imgX, _ = dataset.__getitem__(tmp+500)
        _, imgY = dataset.__getitem__(tmp) # paired imgX & imgY
        imgX = imgX.view(1,1,512,512).to(device)
        imgY = imgY.view(1,1,512,512).to(device)        
        with torch.no_grad():
            net_G_x.eval()
            net_G_y.eval()
            fakeY = dataset.test_transform(net_G_y(imgX))
            fakeX = dataset.test_transform(net_G_x(imgY))
            imgX = dataset.test_transform(imgX)
            imgY = dataset.test_transform(imgY)
        # psnr
        psnr_FakeX_ImgX.append(psnr(fakeY,imgY))
        psnr_FakeY_ImgY.append(psnr(fakeX,imgX))
        psnr_ImgX_ImgY.append(psnr(imgX,imgY))
        print(f"[PSNR FakeX-ImgX: {psnr_FakeX_ImgX[-1]} | PSNR FakeY-ImgY: {psnr_FakeY_ImgY[-1]} | PSNR ImgX-ImgY: {psnr_ImgX_ImgY[-1]}]")
        # show
        save_image(dataset.train_transform(imgX), f"./result_imgs/aapm/imgF{epoch}.png")
        save_image(dataset.train_transform(imgY), f"./result_imgs/aapm/imgQ{epoch}.png")
        save_image(dataset.train_transform(fakeX), f"./result_imgs/aapm/fakeF{epoch}.png")
        save_image(dataset.train_transform(fakeY), f"./result_imgs/aapm/fakeQ{epoch}.png")

        # model save
        torch.save(net_D_x.state_dict(), "./net_D_f.pt")
        torch.save(net_D_y.state_dict(), "./net_D_q.pt")
        torch.save(net_G_x.state_dict(), "./net_G_f.pt")
        torch.save(net_G_y.state_dict(), "./net_G_q.pt")

        # optimizer save
        torch.save(optimizer_D.state_dict(), "./optimizer_D.pt")
        torch.save(optimizer_G.state_dict(), "./optimizer_G.pt")

        # # scheduler save
        # torch.save(scheduler_D.state_dict(), "./scheduler_D.pt")
        # torch.save(scheduler_G.state_dict(), "./scheduler_G.pt")

    # GAN loss graph
    x = np.linspace(1,3839,3839)
    plt.plot(x, epoch_loss_G_x_list, color="blue")
    plt.plot(x, epoch_loss_D_x_list, color="red")
    plt.legend(('G','D'))
    plt.savefig("./loss.png")
    plt.cla()
    # PSNR graph
    plt.plot(x,psnr_FakeX_ImgX,color="blue")
    plt.plot(x,psnr_FakeY_ImgY,color="red")
    plt.plot(x,psnr_ImgX_ImgY,color="green")
    plt.legend(('fX-rX','fY-rY','rX-rY'))
    plt.savefig("./PSNR.png")
    plt.cla()


    return

def eval():
    net_G_x = UNetGenerator(in_channel=1, out_channel=1).to(device)
    net_G_y = UNetGenerator(in_channel=1, out_channel=1).to(device)

    # model load
    checkpoint_x = torch.load('./net_G_f.pt')
    net_G_x.load_state_dict(checkpoint_x)
    net_G_x.eval()

    checkpoint_y = torch.load('./net_G_q.pt')
    net_G_y.load_state_dict(checkpoint_y)
    net_G_y.eval()

    # inference
    dataset = AAPMDataset("test")
    dataloader = DataLoader(dataset, batch_size=1)

    # generate
    for idx in range(10):
        tmp = np.random.randint(0,200)
        imgX, _ = dataset.__getitem__(tmp+50)
        _, imgY = dataset.__getitem__(tmp) # paired imgX & imgY
        imgX = imgX.view(1,1,512,512).to(device)
        imgY = imgY.view(1,1,512,512).to(device)        
        with torch.no_grad():
            net_G_x.eval()
            net_G_y.eval()
            fakeY = dataset.test_transform(net_G_y(imgX))
            fakeX = dataset.test_transform(net_G_x(imgY))
            imgX = dataset.test_transform(imgX)
            imgY = dataset.test_transform(imgY)

        result = torch.cat((imgX,fakeX,imgY,fakeY))
        noise = torch.cat((imgX-imgY,fakeX-imgY,imgX-fakeY))*5
        save_image(result, f"./result{idx}.png")
        save_image(noise, f"./noise{idx}.png")

        print(f"[{idx}-th] PSNR Full-GAN: {psnr(imgX, fakeX)} | PSNR Full-Quarter: {psnr(imgX, imgY)}")

    return


if __name__ == '__main__':
    # train()
    eval()
