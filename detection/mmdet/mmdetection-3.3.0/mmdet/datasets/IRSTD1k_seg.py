# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .ade20k import ADE20KSegDataset

# class IRSTD1kDataset(Data.Dataset):
#     '''
#     Return: Single channel
#     '''

#     def __init__(self, base_dir=r'D:/WFY/datasets/IRSTD-1k',
#                  mode='train', base_size=256):
#         assert mode in ['train', 'test']


#         if mode == 'train':
#             self.data_dir = osp.join(base_dir, 'trainval')
#         elif mode == 'test':
#             self.data_dir = osp.join(base_dir, 'test')
#         else:
#             raise NotImplementedError
#         self.base_size = base_size

#         self.names = []
#         for filename in os.listdir(osp.join(self.data_dir, 'images')):
#             if filename.endswith('png'):
#                 self.names.append(filename)

#         # self.tranform = augumentation()

#         # self.transform = transforms.Compose([
#         #     transforms.ToTensor(),
#         #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
#         # ])

#     def __getitem__(self, i):
#         name = self.names[i]
#         img_path = osp.join(self.data_dir, 'images', name)
#         label_path = osp.join(self.data_dir, 'masks', name)

#         img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
#         # img, mask = self.tranform(img, mask)
#         img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR)
#         mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
#         img = img.reshape(1, self.base_size, self.base_size) / 255.
#         if np.max(mask) > 0:
#             mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
#         else:
#             mask = mask.reshape(1, self.base_size, self.base_size)

#         img = torch.from_numpy(img).type(torch.FloatTensor)
#         mask = torch.from_numpy(mask).type(torch.FloatTensor)

#         return img, mask

#     def __len__(self):
#         return len(self.names)


@DATASETS.register_module()
class IRSTDDataset(ADE20KSegDataset):
    """COCO dataset.

    In segmentation map annotation for COCO. The ``img_suffix`` is fixed to
    '.jpg',  and ``seg_map_suffix`` is fixed to '.png'.
    """

    METAINFO = dict(
        classes=['target'],
        palette=[(255,255,255)]
    )
