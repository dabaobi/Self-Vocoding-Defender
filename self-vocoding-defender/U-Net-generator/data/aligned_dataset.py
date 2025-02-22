import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from data.frequency import fixed_STFT
import torchaudio
import torch.nn.functional as F


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory

        if opt.phase == 'train':
            self.AB_paths = []
            for fname in os.listdir(os.path.join(opt.dataroot, 'self_vocoded')):
                path_A = os.path.join(opt.dataroot, 'self_vocoded', fname)
                path_B = os.path.join(opt.dataroot, 'GT', fname)
                self.AB_paths.append([path_A, path_B])
        elif opt.phase == 'test':
            # hifigan
            distorted_pth = opt.dataroot
            GT_path = opt.GT_path
            self.AB_paths = []
            for fname in os.listdir(distorted_pth):
                path_A = os.path.join(distorted_pth, fname)
                path_B = os.path.join(GT_path, fname)
                self.AB_paths.append([path_A, path_B])

        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.stft = fixed_STFT(1024, 256, 1024)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # # read a image given a random integer index
        # AB_path = self.AB_paths[index]
        # AB = Image.open(AB_path).convert('RGB')
        # # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # A = A_transform(A)
        # B = B_transform(B)
        # print('A', A.shape) # [1, 256, 256]

        # return {'A': A, 'B': A, 'A_paths': AB_path, 'B_paths': AB_path}


        path_A = self.AB_paths[index][0]
        path_B = self.AB_paths[index][1]
        wav_A, sr = torchaudio.load(path_A)
        wav_B, sr = torchaudio.load(path_B)
        # if wav_A.shape[1] > wav_B.shape[1]:
        #     wav_A = wav_A[:, :wav_B.shape[1]]
        # elif wav_A.shape[1] < wav_B.shape[1]:
        #     wav_B = wav_B[:, :wav_A.shape[1]]
        spect_A, phase_A = self.stft.transform(wav_A.unsqueeze(0))
        spect_B, phase_B = self.stft.transform(wav_B.unsqueeze(0))
        # spect_A = spect_A.permute(0, 2, 1)
        # spect_B = spect_B.permute(0, 2, 1)
        spect_A = spect_A[:,:512,:] # frequency 513 --> 512
        spect_B = spect_B[:,:512,:]
        if spect_A.shape[2] > 512:  # time > 512 frame --> cut, time < 512 --> pad 0
            spect_A = spect_A[:,:,:512]
        else:
            spect_A = F.pad(spect_A, (0,512-spect_A.shape[2]), "constant", 0)
        if spect_B.shape[2] > 512:
            spect_B = spect_B[:,:,:512]
        else:
            spect_B = F.pad(spect_B, (0,512-spect_B.shape[2]), "constant", 0)
        # print('path A', path_A)
        # print('wav_A.shape', wav_A.shape, sr)
        # print('spect_A.shape', spect_A)
        # print('path B', path_B)
        # print('wav_B.shape', wav_B.shape, sr)
        # print('spect_B.shape', spect_B.shape)

        return {'A': spect_A, 'B': spect_B, 'A_paths': path_A, 'B_paths': path_B}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
