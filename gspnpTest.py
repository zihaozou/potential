import os
import numpy as np
import hdf5storage
from scipy import ndimage
from argparse import ArgumentParser
from utils.utils_restoration import single2uint,crop_center, matlab_style_gauss2D, imread_uint, imsave, modcrop
from natsort import os_sorted
from scipy.optimize import fminbound
from GS_PnP_restoration import PnP_restoration
from utils.utils_sr import classical_degradation
from collections import OrderedDict
import logging
from datetime import datetime
def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exists!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)
def getInput(dataset_path,dataset_name):
    return os_sorted([os.path.join(dataset_path,dataset_name,p) for p in os.listdir(os.path.join(dataset_path,dataset_name))])

def makeOutputPath(*args):
    outputPath=[]
    for arg in args:
        outputPath.append(arg)
        if not os.path.exists(os.path.join(*outputPath)):
            os.makedirs(os.path.join(*outputPath))
    return os.path.join(*outputPath)
def deblur(hparams):

    # Deblurring specific hyperparameters
    hparams.relative_diff_F_min = 1e-5
    hparams.sigma_denoiser = 1.8 * hparams.noise_level_img
    hparams.degradation_mode = 'deblurring'

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    input_paths = getInput(hparams.dataset_path,hparams.dataset_name)

    # Output images and curves paths
    if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
        exp_out_path=makeOutputPath(hparams.degradation_mode,hparams.denoiser_name,hparams.PnP_algo,hparams.dataset_name,str(hparams.noise_level_img),hparams.kernel_path.split('/')[-1],datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logger_info('log', log_path=os.path.join(exp_out_path, 'deblur'+'.log'))
    logger = logging.getLogger('log')

    psnr_list = []
    F_list = []

    # Load the 8 motion blur kernels
    kernels = hdf5storage.loadmat(hparams.kernel_path)['kernels']

    # Kernels follow the order given in the paper (Table 2). The 8 first kernels are motion blur kernels, the 9th kernel is uniform and the 10th Gaussian.
    if hparams.kernel_path.find('kernels_12.mat') != -1:
        k_list = range(8)
    else:
        k_list = range(10)

    logger.info('\n GS-DRUNET deblurring with image sigma:{:.3f}, model sigma:{:.3f}, lamb:{:.3f} \n'.format(hparams.noise_level_img, hparams.sigma_denoiser, hparams.lamb))

    for k_index in k_list: # For each kernel

        psnr_k_list = []
        psnrY_k_list = []

        if hparams.kernel_path.find('kernels_12.mat') != -1:
            k = kernels[0, k_index]
        else:
            if k_index == 8: # Uniform blur
                k = (1/81)*np.ones((9,9))
                hparams.lamb = 0.075
            elif k_index == 9:  # Gaussian blur
                k = matlab_style_gauss2D(shape=(25,25),sigma=1.6)
                hparams.lamb = 0.075
            else: # Motion blur
                k = kernels[0, k_index]
                hparams.lamb = 0.1

        if hparams.extract_images or hparams.extract_curves :
            kout_path = os.path.join(exp_out_path, 'kernel_'+str(k_index))
            if not os.path.exists(kout_path):
                os.mkdir(kout_path)

        if hparams.extract_curves:
            PnP_module.initialize_curves()

        for i in range(min(len(input_paths),hparams.n_images)): # For each image

            logger.info('__ kernel__',k_index, '__ image__',i,'name:',input_paths[i])

            # load image
            input_im_uint = imread_uint(input_paths[i])
            if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
               input_im_uint = crop_center(input_im_uint, hparams.patch_size,hparams.patch_size)
            input_im = np.float32(input_im_uint / 255.)
            # Degrade image
            blur_im = ndimage.filters.convolve(input_im, np.expand_dims(k, axis=2), mode='wrap')
            np.random.seed(seed=0)
            noise = np.random.normal(0, hparams.noise_level_img/255., blur_im.shape)
            blur_im += noise
            def optimSigFunc(sigFac):
                PnP_module.hparams.sigma_denoiser=sigFac*hparams.noise_level_img
                _,output_psnr,_=PnP_module.restore(blur_im,input_im,k)
                return -output_psnr
            sigFac = fminbound(optimSigFunc, 0.1, 5,disp=2,maxfun=25)
            PnP_module.hparams.sigma_denoiser=sigFac*hparams.noise_level_img
            def optimLambFunc(lamb):
                PnP_module.hparams.lamb=lamb
                _,output_psnr,_=PnP_module.restore(blur_im,input_im,k)
                return -output_psnr
            lamb = fminbound(optimLambFunc, 0.0, 1.0,disp=2,maxfun=50)
            PnP_module.hparams.lamb=lamb
            logger.info(f'__ kernel__{k_index}__ image__{i}__ sigma__{sigFac}__ lamb__{lamb}__')
            # PnP restoration
            if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
                deblur_im, output_psnr, output_psnrY, x_list, z_list, Dx_list, psnr_tab, Ds_list, s_list, F_list = PnP_module.restore(blur_im,input_im,k, extract_results=True)
            else :
                deblur_im, output_psnr,output_psnrY = PnP_module.restore(blur_im,input_im,k)

            logger.info('PSNR: {:.2f}dB'.format(output_psnr))

            psnr_k_list.append(output_psnr)
            psnrY_k_list.append(output_psnrY)
            psnr_list.append(output_psnr)

            if hparams.extract_curves:
                # Create curves
                PnP_module.update_curves(x_list, z_list, Dx_list, psnr_tab, Ds_list, s_list, F_list)

            if hparams.extract_images:
                # Save images
                save_im_path = os.path.join(kout_path, 'images')
                if not os.path.exists(save_im_path):
                    os.mkdir(save_im_path)

                imsave(os.path.join(save_im_path, 'kernel_' + str(k_index) + '.png'), single2uint(deblur_im))
                imsave(os.path.join(save_im_path, 'img_'+str(i)+'_input.png'), input_im_uint)
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'), single2uint(deblur_im))
                imsave(os.path.join(save_im_path, 'img_'+str(i)+'_blur.png'), single2uint(blur_im))
                logger.info('output image saved at '+os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'))

        if hparams.extract_curves:
            # Save curves
            save_curves_path = os.path.join(kout_path,'curves')
            if not os.path.exists(save_curves_path):
                os.mkdir(save_curves_path)
            PnP_module.save_curves(save_curves_path)
            logger.info('output curves saved at '+save_curves_path)

        avg_k_psnr = np.mean(np.array(psnr_k_list))
        logger.info('avg RGB psnr on kernel {}: {:.2f}dB'.format(k_index, avg_k_psnr))
        avg_k_psnrY = np.mean(np.array(psnrY_k_list))
        logger.info('avg Y psnr on kernel {} : {:.2f}dB'.format(k_index, avg_k_psnrY))

def SR(hparams):

    # SR specific hyperparameters
    hparams.degradation_mode = 'SR'
    hparams.classical_degradation = True
    hparams.relative_diff_F_min = 1e-6
    hparams.lamb = 0.065
    hparams.sigma_denoiser = 2*hparams.noise_level_img

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    input_paths = getInput(hparams.dataset_path,hparams.dataset_name)


    # Output images and curves paths
    exp_out_path=makeOutputPath(hparams.degradation_mode,hparams.denoiser_name,hparams.PnP_algo,f'sf={hparams.sf}',hparams.dataset_name,hparams.kernel_path.split('/')[-1],str(hparams.noise_level_img),datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    logger_info('log', log_path=os.path.join(exp_out_path, 'SR'+'.log'))
    logger = logging.getLogger('log')
    psnr_list = []
    psnr_list_sr = []
    F_list = []

    # Load the 8 blur kernels
    kernels = hdf5storage.loadmat(hparams.kernel_path)['kernels']
    # Kernels follow the order given in the paper (Table 3)
    if hparams.kernel_path.find('kernels_12.mat') != -1:
        k_list = range(8)
    else:
        k_list = range(10)
    

    logger.info('\n GS-DRUNET super-resolution with image sigma:{:.3f}, model sigma:{:.3f}, lamb:{:.3f} \n'.format(hparams.noise_level_img, hparams.sigma_denoiser, hparams.lamb))

    for k_index in k_list: # For each kernel

        psnr_k_list = []

        if hparams.kernel_path.find('kernels_12.mat') != -1:
            k = kernels[0, k_index]
        else:
            if k_index == 8: # Uniform blur
                k = (1/81)*np.ones((9,9))
                hparams.lamb = 0.075
            elif k_index == 9:  # Gaussian blur
                k = matlab_style_gauss2D(shape=(25,25),sigma=1.6)
                hparams.lamb = 0.075
            else: # Motion blur
                k = kernels[0, k_index]
                hparams.lamb = 0.1

        if hparams.extract_images or hparams.extract_curves:
            kout_path = os.path.join(exp_out_path, 'kernel_'+str(k_index))
            if not os.path.exists(kout_path):
                os.mkdir(kout_path)

        if hparams.extract_curves:
            PnP_module.initialize_curves()

        for i in range(min(len(input_paths),hparams.n_images)) : # For each image

            logger.info(f'__ kernel__{k_index}__ image__{i}__')

            # load image
            input_im_uint = imread_uint(input_paths[i])
            if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]) and hparams.crop:
               input_im_uint = crop_center(input_im_uint, hparams.patch_size,hparams.patch_size)
            # Degrade image
            input_im_uint = modcrop(input_im_uint, hparams.sf)
            input_im = np.float32(input_im_uint / 255.)
            if classical_degradation:
                blur_im = classical_degradation(input_im, k, hparams.sf)
            else:
                print('not implemented yet')
            np.random.seed(seed=0)
            noise = np.random.normal(0, hparams.noise_level_img/255., blur_im.shape)
            blur_im += noise
            def optimSigFunc(sigFac):
                PnP_module.hparams.sigma_denoiser=sigFac*hparams.noise_level_img
                _,output_psnr,_=PnP_module.restore(blur_im,input_im,k)
                return -output_psnr
            sigFac = fminbound(optimSigFunc, 0.1, 5,disp=2,maxfun=25)
            PnP_module.hparams.sigma_denoiser=sigFac*hparams.noise_level_img
            def optimLambFunc(lamb):
                PnP_module.hparams.lamb=lamb
                _,output_psnr,_=PnP_module.restore(blur_im,input_im,k)
                return -output_psnr
            lamb = fminbound(optimLambFunc, 0.0, 1.0,disp=2,maxfun=50)
            PnP_module.hparams.lamb=lamb
            logger.info(f'__ kernel__{k_index}__ image__{i}__ sigma__{sigFac}__ lamb__{lamb}__')
            # PnP restoration
            if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
                deblur_im, output_psnr, output_psnrY, x_list, z_list, Dx_list, psnr_tab, Ds_list, s_list, F_list = PnP_module.restore(blur_im,input_im,k, extract_results=True)
            else :
                deblur_im, output_psnr, output_psnrY = PnP_module.restore(blur_im,input_im,k)

            logger.info('PSNR: {:.2f}dB'.format(output_psnr))

            psnr_k_list.append(output_psnr)
            psnr_list.append(output_psnr)

            if hparams.extract_curves:
                # Create curves
                PnP_module.update_curves(x_list, z_list, Dx_list, psnr_tab, Ds_list, s_list, F_list)

            if hparams.extract_images:
                # Save images
                save_im_path = os.path.join(kout_path, 'images')
                if not os.path.exists(save_im_path):
                    os.mkdir(save_im_path)

                imsave(os.path.join(save_im_path, 'kernel_' + str(k_index) + '.png'), single2uint(deblur_im))
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_input.png'), input_im_uint)
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_HR.png'), single2uint(deblur_im))
                imsave(os.path.join(save_im_path, 'img_' + str(i) + '_GSPnP.png'), single2uint(blur_im))
                #print('output image saved at ', os.path.join(save_im_path, 'img_' + str(i) + '_GSPnP.png'))

        if hparams.extract_curves:
            # Save curves
            save_curves_path = os.path.join(kout_path, 'curves')
            if not os.path.exists(save_curves_path):
                os.mkdir(save_curves_path)
            PnP_module.save_curves(save_curves_path)
            print('output curves saved at ', save_curves_path)
            
        avg_k_psnr = np.mean(np.array(psnr_k_list))
        logger.info('avg RGB psnr on kernel {}: {:.2f}dB'.format(k_index, avg_k_psnr))

        psnr_list_sr.append(avg_k_psnr)
        if hparams.kernel_path.find('kernels_12.mat') != -1:
            if k_index == 3:
                logger.info('------ avg RGB psnr on isotropic kernels : {:.2f}dB ------'.format(np.mean(np.array(psnr_list_sr))))
                psnr_list_sr = []
            if k_index == 7:
                logger.info('------ avg RGB psnr on anisotropic kernel : {:.2f}dB ------'.format(np.mean(np.array(psnr_list_sr))))
                psnr_list_sr = []

def inpaint(hparams):

    # Inpainting specific hyperparameters
    hparams.degradation_mode = 'inpainting'
    hparams.sigma_denoiser = 10
    hparams.noise_level_img = 0
    hparams.n_init = 10
    hparams.maxitr = 30
    hparams.use_backtracking = False
    hparams.inpainting_init = True

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    input_paths = getInput(hparams.dataset_path,hparams.dataset_name)

    # Output images and curves paths
    kout_path=makeOutputPath(hparams.degradation_mode,hparams.denoiser_name,hparams.PnP_algo,hparams.dataset_name,str(hparams.noise_level_img),'prop_' + str(hparams.prop_mask))
    logger_info('log', log_path=os.path.join(kout_path, 'SR'+'.log'))
    logger = logging.getLogger('log')
    test_results = OrderedDict()
    test_results['psnr'] = []

    if hparams.extract_curves:
        PnP_module.initialize_curves()

    psnr_list = []
    psnrY_list = []
    F_list = []

    for i in range(min(len(input_paths), hparams.n_images)): # For each image

        logger.info('__ image__'%i)

        # load image
        input_im_uint = imread_uint(input_paths[i])
        if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
            input_im_uint = crop_center(input_im_uint, hparams.patch_size, hparams.patch_size)
        input_im = np.float32(input_im_uint / 255.)
        # Degrade image
        mask = np.random.binomial(n=1, p=hparams.prop_mask, size=(input_im.shape[0],input_im.shape[1]))
        mask = np.expand_dims(mask,axis=2)
        mask_im = input_im*mask + (0.5)*(1-mask)

        np.random.seed(seed=0)
        mask_im += np.random.normal(0, hparams.noise_level_img/255., mask_im.shape)
        def optimSigFunc(sigFac):
            PnP_module.hparams.sigma_denoiser=sigFac
            _,output_psnr,_=PnP_module.restore(mask_im, input_im, mask)
            return -output_psnr
        sigFac = fminbound(optimSigFunc, 1, 15,disp=2,maxfun=25)
        PnP_module.hparams.sigma_denoiser=sigFac
        def optimLambFunc(lamb):
            PnP_module.hparams.lamb=lamb
            _,output_psnr,_=PnP_module.restore(mask_im, input_im, mask)
            return -output_psnr
        lamb = fminbound(optimLambFunc, 0.0, 1.0,disp=2,maxfun=50)
        PnP_module.hparams.lamb=lamb
        # PnP restoration
        if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
            inpainted_im, output_psnr, output_psnrY, x_list, z_list, Dx_list, psnr_tab, Ds_list, s_list, F_list = PnP_module.restore(mask_im, input_im, mask, extract_results=True)
        else:
            inpainted_im, output_psnr, output_psnrY = PnP_module.restore(mask_im, input_im, mask)

        logger.info('PSNR: {:.2f}dB'.format(output_psnr))
        psnr_list.append(output_psnr)
        psnrY_list.append(output_psnrY)

        if hparams.extract_curves:
            # Create curves
            PnP_module.update_curves(x_list, z_list, Dx_list, psnr_tab, Ds_list, s_list, F_list)

        if hparams.extract_images:
            # Save images
            save_im_path = os.path.join(kout_path, 'images')
            if not os.path.exists(save_im_path):
                os.mkdir(save_im_path)

            imsave(os.path.join(save_im_path, 'img_' + str(i) + '_input.png'), input_im_uint)
            imsave(os.path.join(save_im_path, 'img_' + str(i) + '_inpainted.png'), single2uint(inpainted_im))
            imsave(os.path.join(save_im_path, 'img_' + str(i) + '_masked.png'), single2uint(mask_im*mask))

            logger.info('output images saved at '+save_im_path)

    if hparams.extract_curves:
        # Save curves
        save_curves_path = os.path.join(kout_path, 'curves')
        if not os.path.exists(save_curves_path):
            os.mkdir(save_curves_path)
        PnP_module.save_curves(save_curves_path)
        logger.info('output curves saved at '+save_curves_path)

    avg_k_psnr = np.mean(np.array(psnr_list))
    logger.info('avg RGB psnr : {:.2f}dB'.format(avg_k_psnr))
    avg_k_psnrY = np.mean(np.array(psnrY_list))
    logger.info('avg Y psnr : {:.2f}dB'.format(avg_k_psnrY))
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('task', type=str, choices=['SR', 'inpainting','deblur'])
    parser.add_argument('--prop_mask', type=float, default=0.5)
    parser.add_argument('--sf', type=int, default=3)
    parser.add_argument('--kernel_path', type=str, default=os.path.join('miscs','kernels_12.mat'))
    parser.add_argument('--no_crop', dest='crop', action='store_false')
    parser.set_defaults(crop=True)
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()
    if hparams.task == 'SR':
        SR(hparams)
    elif hparams.task == 'inpainting':
        inpaint(hparams)
    elif hparams.task == 'deblur':
        deblur(hparams)
    
