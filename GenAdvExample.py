import os
import time
import argparse
import random
from PIL import Image
import torch
from torch.autograd import Variable
from HyperTools import *
from Model_S3ANet import *
from scipy.io import savemat

DataName = {1:'PaviaU',2:'Salinas',3: 'Houston',4:'IndianP'}

def set_seed(seed=42):
    """Fix all random seeds for full reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(42)
    if args.dataID==1:
        num_classes = 9
        num_features = 103
        save_pre_dir = './Data/PaviaU/'
    elif args.dataID==2:       
        num_classes = 16  
        num_features = 204  
        save_pre_dir = './Data/Salinas/'
    elif args.dataID == 3:
        num_classes = 15
        num_features = 144
        save_pre_dir = './Data/Houston/'
    elif args.dataID == 4:
        num_classes = 16
        num_features = 200
        save_pre_dir = './Data/IndianP/'

    X = np.load(save_pre_dir+'X.npy')
    num_features,h,w = X.shape
    Y = np.load(save_pre_dir+'Y.npy')
    num_classes = int(Y.max()) + 1
    
    X_train = np.reshape(X,(1,num_features,h,w))
    train_array = np.load(save_pre_dir+'train_array.npy')
    Y_train = np.ones(Y.shape)*255
    Y_train[train_array] = Y[train_array]
    Y_train = np.reshape(Y_train,(1,h,w)) 

    # define the targeted label in the attack
    Y_tar = np.zeros(Y.shape)
    Y_tar = np.reshape(Y_tar,(1,h,w))
    

    save_path_prefix = args.save_path_prefix+'Exp_adv_3D_'+DataName[args.dataID]+'/'
    
    if os.path.exists(save_path_prefix)==False:
        os.makedirs(save_path_prefix)
    
    num_epochs = 100
    if args.model == 'S3ANet':
        Model = S3ANet(num_features=num_features, num_classes=num_classes, bins=args.bins).cuda()



    Model.train()
    optimizer = torch.optim.Adam(Model.parameters(),lr=args.lr,weight_decay=args.decay)

    images = torch.from_numpy(X_train).float().cuda()
    label = torch.from_numpy(Y_train).long().cuda()
    criterion = CrossEntropy2d().cuda()      

    # train the classification model
    for epoch in range(num_epochs):  
        adjust_learning_rate(optimizer,args.lr,epoch,num_epochs)
        tem_time = time.time()      
        optimizer.zero_grad()
        output = Model(images)  
                
        seg_loss = criterion(output,label)
        seg_loss.backward()

        optimizer.step()
       
        batch_time = time.time()-tem_time
        if (epoch+1) % 1 == 0:            
            print('epoch %d/%d:  time: %.2f cls_loss = %.3f'%(epoch+1, num_epochs,batch_time,seg_loss.item()))
    
    Model.eval()
    output = Model(images)  
    _, predict_labels = torch.max(output, 1)  
    predict_labels = np.squeeze(predict_labels.detach().cpu().numpy()).reshape(-1)

    # adversarial attack
    epsilon = [0.01,0.02,0.04,0.06,0.08,0.1,0.2,0.4,0.6,0.8,1,2,4,6,8,10]
    for i in range(len(epsilon)):
        print('Generate adversarial example with epsilon = %.2f'%(epsilon[i]))
        processed_image = Variable(images)
        processed_image = processed_image.requires_grad_()
        label = torch.from_numpy(Y_tar).long().cuda()
                                                                    
        output = Model(processed_image)
        seg_loss = criterion(output,label)
        seg_loss.backward()
        
        adv_noise = epsilon[i] * processed_image.grad.data / torch.norm(processed_image.grad.data,float("inf"))

        processed_image.data = processed_image.data - adv_noise
       
        X_adv = torch.clamp(processed_image, 0, 1).cpu().data.numpy()[0]
        noise_image = X_adv - images.cpu().data.numpy()[0]        
        noise_image[noise_image > 1] = 1
        noise_image[noise_image < 0] = 0

        savemat(
            save_path_prefix + args.model + '_' + DataName[args.dataID] + '_perturbation' + str(epsilon[i]) + '.mat',
            {'per': noise_image})
        savemat(
            save_path_prefix + args.model + '_' + DataName[args.dataID] + '_advimage' + str(epsilon[i]) + '.mat',
            {'advimage': X_adv})

        def get_bands(b1, b2, b3, max_b):
            return [min(b1, max_b), min(b2, max_b), min(b3, max_b)]
        
        if args.dataID == 1:
            bands = get_bands(102, 56, 31, num_features - 1)
        elif args.dataID == 2:
            bands = get_bands(57, 27, 17, num_features - 1)
        elif args.dataID == 3:
            bands = get_bands(50, 40, 20, num_features - 1)
        elif args.dataID == 4:
            bands = get_bands(102, 56, 31, num_features - 1)
            
        im = Image.fromarray(np.moveaxis((noise_image[bands,:,:]*25500).astype('uint8'),0,-1))
        im.save(save_path_prefix+args.model+'_perturbation'+str(epsilon[i])+'.png','png')
        im = Image.fromarray(np.moveaxis((X_adv[bands,:,:]*255).astype('uint8'),0,-1))
        im.save(save_path_prefix+args.model+'_advimage'+str(epsilon[i])+'.png','png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--model', type=str, default='S3ANet')
    
    # train
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--decay', type=float, default=5e-5)
    parser.add_argument('--bins', nargs='+', type=int, default=[1, 2, 3, 6])

    main(parser.parse_args())
