import os
import torch
from torch.utils.data.dataset import Dataset
import h5py

from pathlib import Path
import random
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans


class WSIDataset(Dataset):
    """Basic WSI Dataset, which can obtain the features of each patch of WSIs."""

    def __init__(self, fea_dir, label_csv, preload: bool = False):
        super(WSIDataset, self).__init__()
        self.fea_dir = fea_dir
        self.csv = pd.read_csv(label_csv)
        self.slide_ids = [slide_id.split('.kfb')[0] for slide_id in self.csv['slide_id'].tolist()]
        self.labels = self.csv['label'].tolist()
        self.preload = preload

        if self.preload:
            self.patch_features = self.load_patch_features()

    def load_patch_features(self):
        """Load the all the patch features of all WSIs. """
        patch_features = []
        for slide_id in self.slide_ids:
            # slide_name, ext = os.path.splitext(slide_id)
            f = h5py.File(os.path.join(self.fea_dir, slide_id + '.h5'))
            patch_feature = f['features']
            patch_feature = torch.as_tensor(np.array(patch_feature), dtype=torch.float32)
            patch_features.append(patch_feature)
        return patch_features

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index: int):

        slide_id = self.slide_ids[index]
        label = self.labels[index]

        # if label < 3:
        #     label = 0
        # else:
        #     label = 1

        label = torch.tensor(label, dtype=torch.long)

        if self.preload:
            patch_feature = self.patch_features[index]
            return slide_id, patch_feature, label
        else:
            f = h5py.File(os.path.join(self.fea_dir, slide_id + '.h5'))
            patch_feature = f['features']
            patch_feature = torch.as_tensor(patch_feature, dtype=torch.float32)
            return slide_id, patch_feature, label


# class WSIwithCluster(Dataset):

#     def __init__(self,
#                  data_csv: Union[str, Path],
#                  num_clusters: int = 10,
#                  indices: Iterable[str] = None,
#                  num_sample_patches: Union[int, float, None] = None,
#                  fixed_size: bool = False,
#                  shuffle: bool = False,
#                  num_condidate_negative_patches = 1000,
#                  device = 0,
#                  patch_random: bool = False) -> None:
#         super(WSIwithCluster, self).__init__()
#         self.data_csv = data_csv
#         self.num_clusters = num_clusters
#         self.indices = indices   
#         self.num_sample_patches = num_sample_patches
#         self.fixed_size = fixed_size
#         self.patch_random = patch_random
#         self.samples = self.process_data()
#         self.device = device
#         if self.indices is None:
#             self.indices = self.samples.index.values
#         if shuffle:
#             self.shuffle()
#         self.patch_dim = np.load(self.samples.at[self.samples.index[0], 'features_filepath'])['img_features'].shape[-1]

#         self.patch_features = self.load_patch_features()
#         # self.condidate_negative_patches = self.get_negative_patches(num_condidate_negative_patches)
#         self.patch_features = self.preload_cluster_features()

#     def preload_cluster_features(self):
#         cluster_patch_features = {}
#         for case_id in self.indices:
#             patch_feature = self.patch_features[case_id]
#             if self.num_sample_patches is not None:
#                 patch_feature = self.sample_feat(patch_feature)
#             if self.fixed_size:
#                 patch_feature = self.fix_size(patch_feature)
#             patch_feature = self.init_cluster(patch_feature)
#             cluster_patch_features[case_id] = patch_feature
#         return cluster_patch_features


#     def init_cluster(self, patch_feature):
#         km = KMeans(n_clusters=self.num_clusters)
#         km.fit(patch_feature)
#         centers, labels = km.cluster_centers_, km.labels_
#         cluster_patch_feature = [[] for i in range(self.num_clusters)]
#         for i in range(patch_feature.shape[0]):
#             cluster_patch_feature[labels[i]].append(torch.from_numpy(patch_feature[i]))
#         for i in range(len(cluster_patch_feature)):
#             cluster_patch_feature[i] = torch.stack(cluster_patch_feature[i], dim=0)
#         return cluster_patch_feature


#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
#         case_id = self.indices[index]
#         patch_feature = self.patch_features[case_id]
#         label = self.samples.at[case_id, 'label']
#         label = torch.tensor(label, dtype=torch.long)
#         return patch_feature, label, case_id


#     def shuffle(self) -> None:
#         """Shuffle the order of WSIs. """
#         random.shuffle(self.indices)

#     def process_data(self):
#         """Load the `data_csv` file by `indices`. """
#         data_csv = pd.read_csv(self.data_csv)
#         data_csv.set_index(keys='case_id', inplace=True)
#         if self.indices is not None:
#             samples = data_csv.loc[self.indices]
#         else:
#             samples = data_csv
#         return samples

#     def load_patch_features(self) -> Dict[str, np.ndarray]:
#         """Load the all the patch features of all WSIs. """
#         patch_features = {}
#         for case_id in self.indices:
#             patch_features[case_id] = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']
#         return patch_features

#     def sample_feat(self, patch_feature: np.ndarray) -> np.ndarray:
#         """Sample features by `num_sample_patches`. """
#         num_patches = patch_feature.shape[0]
#         if self.num_sample_patches is not None and num_patches > self.num_sample_patches:
#             sample_indices = np.random.choice(num_patches, size=self.num_sample_patches, replace=False)
#             sample_indices = sorted(sample_indices)
#             patch_feature = patch_feature[sample_indices]
#         if self.patch_random:
#             np.random.shuffle(patch_feature)
#         return patch_feature

#     def fix_size(self, patch_feature: np.ndarray) -> np.ndarray:
#         """Fixed the shape of each WSI feature. """
#         if patch_feature.shape[0] < self.num_sample_patches:
#             margin = self.num_sample_patches - patch_feature.shape[0]
#             feat_pad = np.zeros(shape=(margin, self.patch_dim))
#             feat = np.concatenate((patch_feature, feat_pad))
#         else:
#             feat = patch_feature[:self.num_sample_patches]
#         return 

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # 3定义model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = WSIDataset('/data5/yhhu/czi/CLAM/FEATURES_DIRECTORY/h5_files', '/data5/yhhu/czi/code/data_split/label1_test copy.csv', preload=False)
    print(len(dataset))
    slide_id, patch_feature, label = dataset[0]
    patch_feature.to(device)
    print(slide_id, patch_feature.shape)
