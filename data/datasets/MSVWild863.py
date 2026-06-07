from __future__ import division, print_function, absolute_import
import glob
import warnings
import os.path as osp
from .bases import BaseImageDataset
import os
import random
import re
class MSVWild863(BaseImageDataset):
    dataset_dir = 'WMVEID863'

    def __init__(self, root='', verbose=True, **kwargs):
        super(MSVWild863, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir)
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated.'
            )

        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'test')

        self._check_before_run()

        train = self.get_data(self.train_dir, relabel=True)
        query = self.get_data(self.query_dir, relabel=False)
        gallery = self.get_data(self.gallery_dir, relabel=False,ratio=1)
        if verbose:
            print("=> MSVWild863 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths_RGB = glob.glob(osp.join(dir_path, 'vis', '*.jpg'))
        pid_container = set()
        for img_path_RGB in img_paths_RGB:
            jpg_name = img_path_RGB.split('/')[-1]
            pid = int(jpg_name.split('_')[0][0:6])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path_RGB in img_paths_RGB:
            img = []
            jpg_name = img_path_RGB.split('/')[-1]
            img_path_NI = osp.join(dir_path, 'ni', jpg_name)
            img_path_TI = osp.join(dir_path, 'th', jpg_name)
            img.append(img_path_RGB)
            img.append(img_path_NI)
            img.append(img_path_TI)
            pid = int(jpg_name.split('_')[0][0:6])
            camid = int(jpg_name.split('_')[1][3])
            trackid = -1
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img, pid, camid, trackid))
            # print("11111")
        return data
    def get_data(self, folder, relabel=False, ratio = 1):
        vids = os.listdir(folder)
        
        if ratio != 1:
            print('randomly sample ',ratio, 'ids for ttt')
            vids = random.sample(vids, int(len(vids)*ratio))
        labels = [int(vid) for vid in vids]

        if relabel:
            label_map = dict()
            for i, lab in enumerate(labels):
                label_map[lab] = i
        cam_set = set()
        img_info = []
        for vid in vids:
            id_vimgs = os.listdir(os.path.join(folder, vid, "vis"))
            id_nimgs = os.listdir(os.path.join(folder, vid, "ni"))
            # print(vid)
            id_timgs = os.listdir(os.path.join(folder, vid, "th"))
            for i, img in enumerate(id_vimgs):
                vpath = os.path.join(folder, vid, "vis", id_vimgs[i])
                npath = os.path.join(folder, vid, "ni", id_nimgs[i])
                tpath = os.path.join(folder, vid, "th", id_timgs[i])
                label = label_map[int(vid)] if relabel else int(vid)


                night = re.search('n+\d',img).group(0)[1]
                cam = re.search('v+\d',img).group(0)[1]
                cam = int(cam)
                night = int(night)
                cam_set.add(cam)
                img_info.append(((vpath, npath, tpath), label, cam, -1))

        return img_info
