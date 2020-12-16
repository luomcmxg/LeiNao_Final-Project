import torch
from torch import nn
import torch.utils.data as D
import torch.nn.functional as F
import numpy as np
from PIL import Image
from keras_preprocessing import image as kim
from PIL import ImageFilter
import matplotlib.pyplot as plt
import os

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True), # change to leaky Relu, save the small pixles, but not relate to the work!!!!!!!!!!!!!
            #nn.LeakyRelU(negative_slope=0.05,inplace=True)
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(ch_in , ch_in, kernel_size=2, stride=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
        nn.BatchNorm2d(ch_out),
      nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class UNET(nn.Module):
  def __init__(self,img_ch=1,output_ch=2):
    super(UNET,self).__init__()
    
    self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

    self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
    self.Conv2 = conv_block(ch_in=64,ch_out=128)
    self.Conv3 = conv_block(ch_in=128,ch_out=256)
    self.Conv4 = conv_block(ch_in=256,ch_out=512)
    self.Conv5 = conv_block(ch_in=512,ch_out=1024)

    self.Up5 = up_conv(ch_in=1024,ch_out=512)
    self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

    self.Up4 = up_conv(ch_in=512,ch_out=256)
    self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
    
    self.Up3 = up_conv(ch_in=256,ch_out=128)
    self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
    
    self.Up2 = up_conv(ch_in=128,ch_out=64)
    self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

    self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
    self.BatchNorm = nn.BatchNorm2d(output_ch)






    ##############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
    
    # def another set of blocks for forward_2 if fully connect concat

  def forward(self,x):
    # encoding path
    x1 = self.Conv1(x)

    x2 = self.Maxpool(x1)
    x2 = self.Conv2(x2)
    
    x3 = self.Maxpool(x2)
    x3 = self.Conv3(x3)

    x4 = self.Maxpool(x3)
    x4 = self.Conv4(x4)

    x5 = self.Maxpool(x4)
    x5 = self.Conv5(x5)
    # decoding + concat path
    d5 = self.Up5(x5)
    d5 = torch.cat((x4,d5),dim=1)
    
    d5 = self.Up_conv5(d5)
    
    d4 = self.Up4(d5)
    d4 = torch.cat((x3,d4),dim=1)
    d4 = self.Up_conv4(d4)

    d3 = self.Up3(d4)
    d3 = torch.cat((x2,d3),dim=1)
    d3 = self.Up_conv3(d3)

    d2 = self.Up2(d3)
    d2 = torch.cat((x1,d2),dim=1)
    d2 = self.Up_conv2(d2)

    d1 = self.Conv_1x1(d2)
    d1 = self.BatchNorm(d1)
    d1 = torch.sigmoid(d1)
    #without concat can work but need more time and L2 regular

    return d1

def main():
  batchsize=1
  trainnum=25
  testnum=5
  image_dir='./segmentation_dataset/new train set/train_img/'
  label_dir='./segmentation_dataset/new train set/train_label/'
  test_image_dir="./segmentation_dataset/new_test_set/test_img/"
  test_label_dir="./segmentation_dataset/new_test_set/test_label/"
  x=[]
  label=[]
  test_x=[]
  test_label=[]
  for i in range(trainnum):
    im_dir_tp=image_dir+str(i)+'.png'
    temp_x=Image.open(im_dir_tp)
    x.append(np.array(temp_x))
  # x=Image.open('test5.png')
  # x=np.array(x)
  
  # print(x)
  # y = np.load("24.png.npy")
  # for i in range(512):
  #   for j in range(512):
  #     if y[i][j]==2:
  #       y[i][j]=255
  #     else:
  #       y[i][j]=0
  # print(y)
  
  #img = Image.fromarray(y.reshape(1,512,512)[0].astype('uint8'))
  #img.show()
  for j in range(trainnum):
    label_dir_tp=label_dir+str(j)+'.png.npy'
    temp_label=np.load(label_dir_tp)
    for i_tp in range(512):
      for j_tp in range(512):
        temp_label[i_tp][j_tp]-=1
    label.append(temp_label)
  # for i in range(512):
  #   for j in range(512):
  #     label[i][j]-=1
  # label=torch.tensor(label).float()
  # print(label)
  x=np.array(x)
  label=np.array(label,dtype='int32')
  x=np.expand_dims(x,1)
  label=np.expand_dims(label,1)

  x = torch.tensor(x).float()
  label=torch.tensor(label).float()
  U_dataset=D.TensorDataset(x,label)
  U_loader=D.DataLoader(dataset=U_dataset,batch_size=batchsize,shuffle=True)

  for i in range(testnum):
    te_im_dir_tp=test_image_dir+str(i)+'.png'
    temp_x=Image.open(te_im_dir_tp)
    test_x.append(np.array(temp_x))
  for i in range(testnum):
    te_label_dir_tp=test_label_dir+str(i)+'.png'
    temp_label = np.array(Image.open(te_label_dir_tp))/255
    test_label.append(temp_label)
  test_x=np.array(test_x)
  test_label=np.array(test_label,dtype='int32')
  test_x=np.expand_dims(test_x,1)
  test_label=np.expand_dims(test_label,1)

  test_x = torch.tensor(test_x).float()
  test_label=torch.tensor(test_label).float()
  U_te_dataset=D.TensorDataset(test_x,test_label)
  U_te_loader=D.DataLoader(dataset=U_te_dataset,batch_size=batchsize,shuffle=True)
  # for epoch in range(1):
  #   for step, (batch_img,batch_label) in enumerate(U_loader):
  #     print(batch_img)
  #     print(batch_label)

  






 #  myNet = nn.Sequential(
 #      nn.Linear(2, 10),
 #      nn.ReLU(),
 #      nn.Linear(10, 1),
 #      nn.Sigmoid()
 # )
  # print(myNet)
  unet=UNET()
  #print(unet)









  #network=UNET()
  # x = np.mat('0 0;'
  #            '0 1;'
  #            '1 0;'
  #            '1 1')
  # x = torch.tensor(x).float()
  # y = np.mat('1;'
  #            '0;'
  #            '0;'
  #            '1')
  # y = torch.tensor(y).float()

  # testdata=D.TensorDataset(x,y)
  # testloader=D.DataLoader(dataset=testdata,shuffle=True,batch_size=2)






  # 设置优化器
  optimzer = torch.optim.SGD(unet.parameters(), lr=0.05,momentum=0.5,weight_decay=0.03)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimzer,  milestones = [150, 1000], gamma = 0.1, last_epoch=-1)
  loss_func_1 = nn.BCELoss()
  loss_func_2 = nn.MSELoss()

  #torch.save(myNet,'./test.pkl')
  #model=torch.load('\test.pkl')
  step_im=np.array(())
  acc_step_im=np.array(())
  loss_im=np.array(())
  acc_im=np.array(())
  te_acc_im=np.array(())
  device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
  unet.to(device)
  

  for epoch in range(1500+1):
    acc=0
    te_acc=0
    for step,(batch_x,batch_label) in enumerate(U_loader):
    #      print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
    #              batch_x.numpy(), '| batch y: ', batch_label.numpy())
    #     out=unet.forward(x)
          batch_x=batch_x.to(device)
          batch_label=batch_label.to(device)
          out=unet(batch_x)
    #     loss = loss_func(out, label)  # 计算误差
          L1_regularization_loss = 0
          if epoch<25:
            L1_lambda=0.001
          else:
            L1_lambda=0.05
          for param in unet.parameters():
            L1_regularization_loss += torch.sum(abs(param))/param.numel()
          loss_1= 0.2*loss_func_1(out[0][0],batch_label)+L1_lambda*L1_regularization_loss
          loss_2= 0.5*loss_func_2(out[0][1],batch_x/255)
          loss=loss_1+loss_2
          if epoch%5==0:
            predict=out>0.5
            acc+=float(torch.sum(predict[0][0]==batch_label[0][0]))/(512*512)
            if epoch%300==0:
              predict=(predict).int()
              os.makedirs('./'+str(epoch)+'trainresult_3', exist_ok=True)
              train_img = Image.fromarray(predict[0][0].detach().cpu().numpy().reshape(1,512,512)[0].astype('uint8')*255)
              train_img.save('./'+str(epoch)+'trainresult_3/'+str(step)+'.png')
          loss_im=np.append(loss_im,loss.detach().cpu().numpy())
          step_im=np.append(step_im,epoch)
          optimzer.zero_grad()  # 清除梯度
          loss_1.backward(retain_graph=True)
          loss_2.backward()
          optimzer.step()
    scheduler.step()
    if epoch%5==0:

      for test_step,(batch_test_x,batch_test_label) in enumerate(U_te_loader):
      #      print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
      #              batch_test_x.numpy(), '| batch y: ', batch_test_label.numpy())
      #     out=unet.forward(x)
            batch_test_x=batch_test_x.to(device)
            batch_test_label=batch_test_label.to(device)
            test_out=unet(batch_test_x)
            if epoch%300==0:
              os.makedirs('./'+str(epoch)+'testresult_3', exist_ok=True)
              test_out_img=(test_out>0.5).int()
              test_img = Image.fromarray(test_out_img[0][0].detach().cpu().numpy().reshape(1,512,512)[0].astype('uint8')*255)
              test_img.save('./'+str(epoch)+'testresult_3/'+str(test_step)+'.png')
      #     loss = loss_func(out, label)  # 计算误差
            
            test_predict=test_out>0.5
            te_acc+=float(torch.sum(test_predict[0][0]==batch_test_label[0][0]))/(512*512)
      acc_im=np.append(acc_im,acc/trainnum)
      te_acc_im=np.append(te_acc_im,te_acc/testnum)  
      acc_step_im=np.append(acc_step_im,epoch)    

    if epoch%5==0:
      print('Epoch: ', epoch, '| Loss: ', loss,  '| Accuracy: ', acc/trainnum, '| Test Accuracy: ', te_acc/testnum)
      torch.save(unet,'./unet_3.pkl')
  #model=torch.load('./unet_3.pkl') 
  np.save('./loss_3.npy',loss_im)
  np.save('./step_3.npy',step_im)
  loss_im=np.load('./loss_3.npy')
  step_im=np.load('./step_3.npy')
  plt.plot(step_im,loss_im,color='b')
  plt.savefig('./result_3.png')
  plt.show()
  plt.close()
  plt.plot(acc_step_im,acc_im,color='r')
  plt.savefig('./accuracy_3.png')
  plt.show()
  plt.plot(acc_step_im,te_acc_im,color='y')
  plt.savefig('./test_accuracy_3.png')
  plt.show()
  # print(myNet(x).data)
if __name__=='__main__':
  main()