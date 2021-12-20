import os
import astropy.io.fits as fits
from astropy import wcs
from astropy.coordinates import SkyCoord
from skimage import filters
import numpy as np
from skimage import measure, morphology
from scipy import ndimage
import pandas as pd
import time
from astropy.stats import SigmaClip
from photutils.background import StdBackgroundRMS
from time import sleep, ctime
from threading import Thread
import matplotlib.pyplot as plt

from DensityClust.clustring_subfunc import \
    kc_coord_3d, kc_coord_2d, get_xyz, setdiff_nd


class Data:
    def __init__(self, data_path=''):
        self.data_path = data_path
        self.wcs = None
        self.rms = None
        self.data_cube = None
        self.shape = None
        self.exist_file = False
        self.state = False
        self.n_dim = None
        self.file_type = None
        self.rms_ = None
        self.data_header = None
        self.read_file()
        self.get_wcs()
        self.calc_background_rms()

    def read_file(self):
        if os.path.exists(self.data_path):
            self.exist_file = True
            self.file_type = self.data_path.split('.')[-1]
            if self.file_type == 'fits':
                self.data_cube = fits.getdata(self.data_path)
                self.data_header = fits.getheader(self.data_path)
                self.state = True
            if self.state:
                self.shape = self.data_cube.shape
                self.n_dim = self.data_cube.ndim
            else:
                print('data read error!')
        else:
            print('the file not exists!')

    def get_wcs(self):
        """
        得到wcs信息
        :return:
        data_wcs
        """
        if self.exist_file:
            data_header = self.data_header
            keys = data_header.keys()

            try:
                key = [k for k in keys if k.endswith('4')]
                [data_header.remove(k) for k in key]
                data_header.remove('VELREF')
            except:
                pass

            data_wcs = wcs.WCS(data_header)
            self.wcs = data_wcs

    def calc_background_rms(self):
        """
         This functions finds an estimate of the RMS noise in the supplied data data_cube.
        :return: bkgrms_value
        """
        sigma_clip = SigmaClip(sigma=3.0)
        bkgrms = StdBackgroundRMS(sigma_clip)
        data = self.data_cube
        bkgrms_value = bkgrms.calc_background_rms(data)
        self.rms_ = bkgrms_value

        data_header = self.data_header
        keys = data_header.keys()
        key = [k for k in keys]
        if 'RMS' in key:
            self.rms = data_header['RMS']
            print('noise is %.4f' % data_header['RMS'])

    def summary(self):
        print('=' * 30)
        print('data file: \n%s' % self.data_path)
        print('the rms of data: %.5f' % self.rms)
        if self.n_dim == 3:
            print('data shape: [%d %d %d]' % self.data_cube.shape)
        if self.n_dim == 2:
            print('data shape: [%d %d]' % self.data_cube.shape)
        print('='*30)


class Param:
    """
        para.rhomin: Minimum density
        para.deltamin: Minimum delta
        para.v_min: Minimum volume
        para.noise: The noise level of the data, used for data truncation calculation
        para.dc: Standard deviation of Gaussian filtering
    """
    def __init__(self, rms=0.23, dc=0.6):
        self.rhomin = rms * 3
        self.deltamin = 4
        self.v_min = 27
        self.noise = rms * 2
        self.dc = dc
        self.gradmin = 0.01

    def set_para(self, gradmin, rhomin, deltamin, v_min, rms, dc):
        self.gradmin = gradmin
        self.rhomin = rhomin
        self.deltamin = deltamin
        self.v_min = v_min
        self.noise = 2 * rms
        self.dc = dc

    def set_para_by_data(self, data):
        if data.state and data.rms is not None:
            self.rhomin = data.rms * 3
            self.noise = data.rms * 2

    def summary(self):
        table_title = ['rhomin[3*rms]', 'deltamin[4]', 'v_min[27]', 'gradmin[0.01]', 'noise[2*rms]', 'dc']
        para = np.array([[self.rhomin, self.deltamin, self.v_min, self.gradmin, self.noise, self.dc]])
        para_pd = pd.DataFrame(para, columns=table_title)
        print('='*30)
        print(para_pd)
        print('='*30)


class Detect_result:
    def __init__(self):
        self.out = None
        self.mask = None
        self.outcat = None
        self.outcat_wcs = None
        self.data = None
        self.para = None

    def save_outcat(self, outcat_name):
        """
        # 保存LDC检测的直接结果，即单位为像素
        :param outcat_name: 核表的路径
        :return:
        """
        outcat = self.outcat
        outcat_colums = outcat.shape[1]
        if outcat_colums == 10:
            # 2d result
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
            dataframe = pd.DataFrame(outcat, columns=table_title)
            dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Cen1': 3, 'Cen2': 3,
                                         'Size1': 3, 'Size2': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 13:
            # 3d result
            dataframe = outcat.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                      'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 11:
            # fitting 2d data result
            fit_outcat = outcat
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'theta', 'Peak',
                           'Sum', 'Volume']
            dataframe = pd.DataFrame(fit_outcat, columns=table_title)
            dataframe = dataframe.round(
                {'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Cen1': 3, 'Cen2': 3, 'Size1': 3, 'Size2': 3, 'theta': 3, 'Peak': 3,
                 'Sum': 3, 'Volume': 3})
        else:
            print('outcat columns is %d' % outcat_colums)
            return

        dataframe.to_csv(outcat_name, sep='\t', index=False)

    def save_outcat_wcs(self, outcat_wcs_name):
        """
        # 保存LDC检测的直接结果，即单位为像素
        :return:
        """
        outcat_wcs = self.outcat_wcs
        outcat_colums = outcat_wcs.shape[1]
        if outcat_colums == 10:
            # 2d result
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
            dataframe = pd.DataFrame(outcat_wcs, columns=table_title)
            dataframe = dataframe.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Cen1': 3, 'Cen2': 3,
                                         'Size1': 3, 'Size2': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 13:
            # 3d result
            dataframe = outcat_wcs.round({'ID': 0, 'Peak1': 0, 'Peak2': 0, 'Peak3': 0, 'Cen1': 3, 'Cen2': 3, 'Cen3': 3,
                                          'Size1': 3, 'Size2': 3, 'Size3': 3, 'Peak': 3, 'Sum': 3, 'Volume': 3})

        elif outcat_colums == 11:
            # fitting 2d data result
            fit_outcat = outcat_wcs
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'theta', 'Peak',
                           'Sum', 'Volume']
            dataframe = pd.DataFrame(fit_outcat, columns=table_title)
            dataframe = dataframe.round(
                {'ID': 0, 'Peak1': 3, 'Peak2': 3, 'Cen1': 3, 'Cen2': 3, 'Size1': 3, 'Size2': 3, 'theta': 3, 'Peak': 3,
                 'Sum': 3, 'Volume': 3})
        else:
            print('outcat columns is %d' % outcat_colums)
            return

        dataframe.to_csv(outcat_wcs_name, sep='\t', index=False)

    def save_result(self, out_name, mask_name):
        mask = self.mask
        out = self.out
        if mask is not None:
            if os.path.isfile(mask_name):
                os.remove(mask_name)
                fits.writeto(mask_name, mask)
            else:
                fits.writeto(mask_name, mask)
        else:
            print('mask is None!')

        if out is not None:
            if os.path.isfile(out_name):
                os.remove(out_name)
                fits.writeto(out_name, self.out)
            else:
                fits.writeto(out_name, self.out)
        else:
            print('out is None!')

    def make_plot_wcs_1(self):
        """
        在积分图上绘制检测结果
        """
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        plt.rcParams['xtick.color'] = 'red'
        plt.rcParams['ytick.color'] = 'red'
        data_name = r'R2_data\data_9\0180-005\0180-005_L.fits'
        fits_path = data_name.replace('.fits', '')
        title = fits_path.split('\\')[-1]
        # fig_name = os.path.join(fits_path, title + '.png')

        outcat_wcs = self.outcat_wcs
        wcs = self.data.wcs
        data_cube = self.data.data_cube
        outcat_wcs_c = SkyCoord(frame="galactic", l=outcat_wcs['Cen1'].values, b=outcat_wcs['Cen2'].values, unit="deg")

        fig = plt.figure(figsize=(10, 8.5), dpi=100)

        axes0 = fig.add_axes([0.15, 0.1, 0.7, 0.82], projection=wcs.celestial)
        axes0.set_xticks([])
        axes0.set_yticks([])
        if self.data.n_dim == 3:
            im0 = axes0.imshow(data_cube.sum(axis=0))
        else:
            im0 = axes0.imshow(data_cube)
        axes0.plot_coord(outcat_wcs_c, 'r*', markersize=2.5)
        axes0.set_xlabel("Galactic Longutide", fontsize=12)
        axes0.set_ylabel("Galactic Latitude", fontsize=12)
        axes0.set_title(title, fontsize=12)

        pos = axes0.get_position()
        pad = 0.01
        width = 0.02
        axes1 = fig.add_axes([pos.xmax + pad, pos.ymin, width, 1 * (pos.ymax - pos.ymin)])

        cbar = fig.colorbar(im0, cax=axes1)
        cbar.set_label('K m s${}^{-1}$')
        plt.show()

    def log(self):
        outcat = self.outcat
        self.para.summary()
        self.data.summary()

        print('%10s' % 'Result' + '='*30)
        print('The number of clumps: %d' % outcat.shape[0])
        print('=' * 30)


class LocalDensityClust:

    def __init__(self, data, para):
        # 参数初始化
        self.data = data
        self.para = para
        self.para.set_para_by_data(data)

        self.result = Detect_result()

        self.xx = None
        self.delta = None
        self.IndNearNeigh = None
        self.Gradient = None
        # size_x, size_y, size_z = self.data.shape
        # self.size_x, self.size_y, self.size_z = size_x, size_y, size_z
        maxed = 0
        ND = 1
        for item in self.data.shape:
            maxed += item**2
            ND *= item

        self.maxed = maxed**0.5
        self.ND = ND

        self.detect()
        self.result.outcat_wcs = self.change_pix2word()

    def kc_coord(self, point_ii_xy, r):
        """
        :param point_ii_xy: 当前点坐标(x,y,z)
        :param xm: size_x
        :param ym: size_y
        :param zm: size_z
        :param r: 2 * r + 1
        :return:
        返回delta_ii_xy点r邻域的点坐标
        """
        n_dim = self.data.n_dim
        if n_dim == 3:
            # xm, ym, zm = self.size_z, self.size_y, self.size_x
            zm, ym, xm = self.data.shape
            it = point_ii_xy[0]
            jt = point_ii_xy[1]
            kt = point_ii_xy[2]

            xyz_min = np.array([[1, it - r], [1, jt - r], [1, kt - r]])
            xyz_min = xyz_min.max(axis=1)

            xyz_max = np.array([[xm, it + r], [ym, jt + r], [zm, kt + r]])
            xyz_max = xyz_max.min(axis=1)

            x_arange = np.arange(xyz_min[0], xyz_max[0] + 1)
            y_arange = np.arange(xyz_min[1], xyz_max[1] + 1)
            v_arange = np.arange(xyz_min[2], xyz_max[2] + 1)

            [p_k, p_i, p_j] = np.meshgrid(x_arange, y_arange, v_arange, indexing='ij')
            Index_value = np.column_stack([p_k.flatten(), p_i.flatten(), p_j.flatten()])
            Index_value = setdiff_nd(Index_value, np.array([point_ii_xy]))

            ordrho_jj = np.matmul(Index_value - 1, np.array([[1], [xm], [ym * xm]]))
            ordrho_jj.reshape([1, ordrho_jj.shape[0]])

        else:
            """
            kc_coord_2d(point_ii_xy, xm, ym, r):
            bt = kc_coord_2d(point_ii_xy, size_y, size_x, k)
            size_x, size_y = data.shape
            :param point_ii_xy: 当前点坐标(x,y)
            :param xm: size_x
            :param ym: size_y
            :param r: 2 * r + 1
            :return:
            返回point_ii_xy点r邻域的点坐标
            
            """
            ym, xm = self.data.shape
            it = point_ii_xy[0]
            jt = point_ii_xy[1]

            xyz_min = np.array([[1, it - r], [1, jt - r]])
            xyz_min = xyz_min.max(axis=1)

            xyz_max = np.array([[xm, it + r], [ym, jt + r]])
            xyz_max = xyz_max.min(axis=1)

            x_arrange = np.arange(xyz_min[0], xyz_max[0] + 1)
            y_arrange = np.arange(xyz_min[1], xyz_max[1] + 1)

            [p_k, p_i] = np.meshgrid(x_arrange, y_arrange, indexing='ij')
            Index_value = np.column_stack([p_k.flatten(), p_i.flatten()])
            Index_value = setdiff_nd(Index_value, np.array([point_ii_xy]))

            ordrho_jj = np.matmul(Index_value - 1, np.array([[1], [xm]]))
            ordrho_jj.reshape([1, ordrho_jj.shape[0]])

        return ordrho_jj[:, 0], Index_value

    def detect(self):
        deltamin = self.para.deltamin
        data = self.data.data_cube

        k1 = 1  # 第1次计算点的邻域大小
        k2 = np.ceil(deltamin).astype(np.int32)   # 第2次计算点的邻域大小
        self.xx = get_xyz(data)  # xx: 3D data coordinates  坐标原点是 1

        data_filter = filters.gaussian(data, self.para.dc)
        rho = data_filter.flatten()
        rho_Ind = np.argsort(-rho)
        rho_sorted = rho[rho_Ind]
        # delta 记录距离，
        # IndNearNeigh 记录：两个密度点的联系 % index of nearest neighbor with higher density
        self.delta = np.zeros(self.ND, np.float32)    # np.iinfo(np.int32).max-->2147483647-->1290**3
        self.IndNearNeigh = np.zeros(self.ND, np.int32) + self.ND
        self.Gradient = np.zeros(self.ND, np.float32)

        self.delta[rho_Ind[0]] = self.maxed
        self.IndNearNeigh[rho_Ind[0]] = rho_Ind[0]

        t0_ = time.time()
        print('First step: calculating delta and Gradient.' + '-' * 20)

        for ii in range(1, self.ND):
            # 密度降序排序后，即密度第ii大的索引(在rho中)
            ordrho_ii = rho_Ind[ii]
            rho_ii = rho_sorted[ii]   # 第ii大的密度值
            if rho_ii >= self.para.noise:
                delta_ordrho_ii = self.maxed
                Gradient_ordrho_ii = 0
                IndNearNeigh_ordrho_ii = 0
                point_ii_xy = self.xx[ordrho_ii, :]

                get_value = True  # 判断是否需要在大循环中继续执行，默认需要，一旦在小循环中赋值成功，就不在大循环中运行
                idex, bt = self.kc_coord(point_ii_xy, k1)
                for ordrho_jj, item in zip(idex, bt):
                    rho_jj = rho[ordrho_jj]  # 根据索引在rho里面取值
                    dist_i_j = np.sqrt(((point_ii_xy - item) ** 2).sum())  # 计算两点间的距离
                    gradient = (rho_jj - rho_ii) / dist_i_j
                    if dist_i_j <= delta_ordrho_ii and gradient >= 0:
                        delta_ordrho_ii = dist_i_j
                        Gradient_ordrho_ii = gradient
                        IndNearNeigh_ordrho_ii = ordrho_jj
                        get_value = False

                if get_value:
                    # 表明，在(2 * k1 + 1) * (2 * k1 + 1) * (2 * k1 + 1)的邻域中没有找到比该点高，距离最近的点，则在更大的邻域中搜索
                    idex, bt = self.kc_coord(point_ii_xy, k2)
                    for ordrho_jj, item in zip(idex, bt):
                        rho_jj = rho[ordrho_jj]  # 根据索引在rho里面取值
                        dist_i_j = np.sqrt(((point_ii_xy - item) ** 2).sum())  # 计算两点间的距离
                        gradient = (rho_jj - rho_ii) / dist_i_j
                        if dist_i_j <= delta_ordrho_ii and gradient >= 0:
                            delta_ordrho_ii = dist_i_j
                            Gradient_ordrho_ii = gradient
                            IndNearNeigh_ordrho_ii = ordrho_jj
                            get_value = False

                if get_value:
                    delta_ordrho_ii = k2 + 0.0001
                    Gradient_ordrho_ii = -1
                    IndNearNeigh_ordrho_ii = self.ND

                self.delta[ordrho_ii] = delta_ordrho_ii
                self.Gradient[ordrho_ii] = Gradient_ordrho_ii
                self.IndNearNeigh[ordrho_ii] = IndNearNeigh_ordrho_ii
            else:
               pass

        delta_sorted = np.sort(-1 * self.delta) * -1
        self.delta[rho_Ind[0]] = delta_sorted[1]
        t1_ = time.time()
        print(' '*10 + 'delata, rho and Gradient are calculated, using %.2f seconds.' % (t1_ - t0_))

        t0_ = time.time()
        LDC_outcat, mask, out = self.extro_outcat(rho_Ind, rho)
        t1_ = time.time()
        print(' '*10 + 'Outcats are calculated, using %.2f seconds.' % (t1_ - t0_))
        self.result.outcat = LDC_outcat
        self.result.outcat_wcs = self.change_pix2word()
        self.result.mask = mask
        self.result.out = out
        self.result.data = self.data
        self.result.para = self.para

    def extro_outcat(self, rho_Ind, rho):
        deltamin = self.para.deltamin
        data = self.data.data_cube
        dim = self.data.n_dim

        # Initialize the return result: mask and out
        mask = np.zeros_like(data, dtype=np.int32)
        out = np.zeros_like(data, dtype=np.float32)
        print('Second step: calculating Outcats.' + '-' * 30)
        # 根据密度和距离来确定类中心
        clusterInd = -1 * np.ones(self.ND + 1)
        clust_index = np.intersect1d(np.where(rho > self.para.rhomin), np.where(self.delta > deltamin))

        clust_num = len(clust_index)
        # icl是用来记录第i个类中心在xx中的索引值
        icl = np.zeros(clust_num, dtype=np.int32)
        n_clump = 0
        for ii in range(clust_num):
            i = clust_index[ii]
            icl[n_clump] = i
            n_clump += 1
            clusterInd[i] = n_clump
        # assignation 将其他非类中心分配到离它最近的类中心中去
        # clusterInd = -1 表示该点不是类的中心点，属于其他点，等待被分配到某个类中去
        # 类的中心点的梯度Gradient被指定为 - 1

        for i in range(self.ND):
            ordrho_i = rho_Ind[i]
            if clusterInd[ordrho_i] == -1:  # not centroid
                clusterInd[ordrho_i] = clusterInd[self.IndNearNeigh[ordrho_i]]
            else:
                self.Gradient[ordrho_i] = -1  # 将类中心点的梯度设置为-1

        clump_volume = np.zeros(n_clump)
        for i in range(n_clump):
            clump_volume[i] = np.where(clusterInd == (i + 1))[0].shape[0]
        # centInd [类中心点在xx坐标下的索引值，类中心在centInd的索引值: 代表类别编号]
        centInd = []
        for i, item in enumerate(clump_volume):
            if item >= self.para.v_min:
                centInd.append([icl[i], i])
        centInd = np.array(centInd, np.int32)

        # 通过梯度确定边界后，还需要进一步利用最小体积来排除假核
        n_clump = centInd.shape[0]
        clump_sum, clump_volume, clump_peak = np.zeros([n_clump, 1]), np.zeros([n_clump, 1]), np.zeros([n_clump, 1])
        clump_Cen, clump_size = np.zeros([n_clump, dim]), np.zeros([n_clump, dim])
        clump_Peak = np.zeros([n_clump, dim], np.int32)
        clump_ii = 0
        if dim == 3:
            for i, item_cent in enumerate(centInd):
                rho_cluster_i = np.zeros(self.ND)
                index_cluster_i = np.where(clusterInd == (item_cent[1] + 1))[0]  # centInd[i, 1] --> item[1] 表示第i个类中心的编号
                clump_rho = rho[index_cluster_i]
                rho_max_min = clump_rho.max() - clump_rho.min()
                Gradient_ = self.Gradient.copy()
                grad_clump_i = Gradient_ / rho_max_min
                mask_grad = np.where(grad_clump_i > self.para.gradmin)[0]
                index_cc = np.intersect1d(mask_grad, index_cluster_i)
                rho_cluster_i[index_cluster_i] = rho[index_cluster_i]
                rho_cc_mean = rho[index_cc].mean()
                index_cc_rho = np.where(rho_cluster_i > rho_cc_mean)[0]
                index_cluster_rho = np.union1d(index_cc, index_cc_rho)

                cl_1_index_ = self.xx[index_cluster_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
                # clusterInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
                clustNum = cl_1_index_.shape[0]
                cl_i = np.zeros(data.shape, np.int32)
                for j, item in enumerate(cl_1_index_):
                    cl_i[item[2], item[1], item[0]] = 1

                # 形态学处理
                L = ndimage.binary_fill_holes(cl_i).astype(np.int32)
                L = measure.label(L)  # Labeled input image. Labels with value 0 are ignored.
                STATS = measure.regionprops(L)

                Ar_sum = []
                for region in STATS:
                    coords = region.coords  # 经过验证，坐标原点为0
                    temp = 0
                    for j, item in enumerate(coords):
                        temp += data[item[0], item[1], item[2]]
                    Ar_sum.append(temp)
                Ar = np.array(Ar_sum)
                ind = np.where(Ar == Ar.max())[0]
                L[L != ind[0] + 1] = 0
                cl_i = L / (ind[0] + 1)
                coords = STATS[ind[0]].coords  # 最大的连通域对应的坐标
                clustNum = coords.shape[0]
                if clustNum > self.para.v_min:
                    coords = coords[:, [2, 1, 0]]
                    clump_i_ = np.zeros(coords.shape[0])
                    for j, item in enumerate(coords):
                        clump_i_[j] = data[item[2], item[1], item[0]]

                    clustsum = clump_i_.sum() + 0.0001  # 加一个0.0001 防止分母为0
                    clump_Cen[clump_ii, :] = np.matmul(clump_i_, coords) / clustsum
                    clump_volume[clump_ii, 0] = clustNum
                    clump_sum[clump_ii, 0] = clustsum

                    x_i = coords - clump_Cen[clump_ii, :]
                    clump_size[clump_ii, :] = 2.3548 * np.sqrt((np.matmul(clump_i_, x_i ** 2) / clustsum)
                                                               - (np.matmul(clump_i_, x_i) / clustsum) ** 2)
                    clump_i = data * cl_i
                    out = out + clump_i
                    mask = mask + cl_i * (clump_ii + 1)
                    clump_peak[clump_ii, 0] = clump_i.max()
                    clump_Peak[clump_ii, [2, 1, 0]] = np.argwhere(clump_i == clump_i.max())[0]
                    clump_ii += 1
                else:
                    pass
        else:
            for i, item_cent in enumerate(centInd):  # centInd[i, 1] --> item[1] 表示第i个类中心的编号
                rho_cluster_i = np.zeros(self.ND)
                index_cluster_i = np.where(clusterInd == (item_cent[1] + 1))[0]  # centInd[i, 1] --> item[1] 表示第i个类中心的编号
                clump_rho = rho[index_cluster_i]
                rho_max_min = clump_rho.max() - clump_rho.min()
                Gradient_ = self.Gradient.copy()
                grad_clump_i = Gradient_ / rho_max_min
                mask_grad = np.where(grad_clump_i > self.para.gradmin)[0]
                index_cc = np.intersect1d(mask_grad, index_cluster_i)
                rho_cluster_i[index_cluster_i] = rho[index_cluster_i]
                rho_cc_mean = rho[index_cc].mean()
                index_cc_rho = np.where(rho_cluster_i > rho_cc_mean)[0]
                index_cluster_rho = np.union1d(index_cc, index_cc_rho)

                cl_1_index_ = self.xx[index_cluster_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
                # clusterInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
                clustNum = cl_1_index_.shape[0]

                cl_i = np.zeros(data.shape, np.int32)
                index_cc_rho = np.where(rho_cluster_i > rho_cc_mean)[0]
                index_clust_rho = np.union1d(index_cc, index_cc_rho)

                cl_1_index_ = self.xx[index_clust_rho, :] - 1  # -1 是为了在data里面用索引取值(从0开始)
                # clustInd  标记的点的编号是从1开始，  没有标记的点的编号为-1
                for j, item in enumerate(cl_1_index_):
                    cl_i[item[1], item[0]] = 1
                # 形态学处理
                L = ndimage.binary_fill_holes(cl_i).astype(int)
                L = measure.label(L)  # Labeled input image. Labels with value 0 are ignored.
                STATS = measure.regionprops(L)

                Ar_sum = []
                for region in STATS:
                    coords = region.coords  # 经过验证，坐标原点为0
                    coords = coords[:, [1, 0]]
                    temp = 0
                    for j, item in enumerate(coords):
                        temp += data[item[1], item[0]]
                    Ar_sum.append(temp)
                Ar = np.array(Ar_sum)
                ind = np.where(Ar == Ar.max())[0]
                L[L != ind[0] + 1] = 0
                cl_i = L / (ind[0] + 1)
                coords = STATS[ind[0]].coords  # 最大的连通域对应的坐标

                clustNum = coords.shape[0]

                if clustNum > self.para.v_min:
                    coords = coords[:, [1, 0]]
                    clump_i_ = np.zeros(coords.shape[0])
                    for j, item in enumerate(coords):
                        clump_i_[j] = data[item[1], item[0]]

                    clustsum = sum(clump_i_) + 0.0001  # 加一个0.0001 防止分母为0
                    clump_Cen[clump_ii, :] = np.matmul(clump_i_, coords) / clustsum
                    clump_volume[clump_ii, 0] = clustNum
                    clump_sum[clump_ii, 0] = clustsum

                    x_i = coords - clump_Cen[clump_ii, :]
                    clump_size[clump_ii, :] = 2.3548 * np.sqrt((np.matmul(clump_i_, x_i ** 2) / clustsum)
                                                              - (np.matmul(clump_i_, x_i) / clustsum) ** 2)
                    clump_i = data * cl_i
                    out = out + clump_i
                    mask = mask + cl_i * (clump_ii + 1)
                    clump_peak[clump_ii, 0] = clump_i.max()
                    clump_Peak[clump_ii, [1, 0]] = np.argwhere(clump_i == clump_i.max())[0]
                    clump_ii += 1
                else:
                    pass

        clump_Peak = clump_Peak + 1
        clump_Cen = clump_Cen + 1  # python坐标原点是从0开始的，在这里整体加1，改为以1为坐标原点
        id_clumps = np.array([item + 1 for item in range(n_clump)], np.int32).T
        id_clumps = id_clumps.reshape([n_clump, 1])

        LDC_outcat = np.column_stack(
            (id_clumps, clump_Peak, clump_Cen, clump_size, clump_peak, clump_sum, clump_volume))
        LDC_outcat = LDC_outcat[:clump_ii, :]
        if dim == 3:
            table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak',
                           'Sum', 'Volume']
        else:
            table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']

        LDC_outcat = pd.DataFrame(LDC_outcat, columns=table_title)

        return LDC_outcat, mask, out

    def change_pix2word(self):
        """
        将算法检测的结果(像素单位)转换到天空坐标系上去
        :return:
        outcat_wcs
        ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3', 'Peak', 'Sum', 'Volume']
        -->3d

         ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2',  'Size1', 'Size2', 'Peak', 'Sum', 'Volume']
         -->2d
        """
        outcat = self.result.outcat
        if outcat is None:
            return None
        else:
            outcat_column = outcat.shape[1]
            data_wcs = self.data.wcs
            if outcat_column == 10:
                # 2d result
                peak1, peak2 = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], 1)
                cen1, cen2 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], 1)
                size1, size2 = np.array([outcat['Size1'] * 30, outcat['Size2'] * 30])

                clump_Peak = np.column_stack([peak1, peak2])
                clump_Cen = np.column_stack([cen1, cen2])
                clustSize = np.column_stack([size1, size2])
                clustPeak, clustSum, clustVolume = np.array([outcat['Peak'], outcat['Sum'], outcat['Volume']])

                id_clumps = []  # MWSIP017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
                for item_l, item_b in zip(cen1, cen2):
                    str_l = 'MWSIP' + ('%.03f' % item_l).rjust(7, '0')
                    if item_b < 0:
                        str_b = '-' + ('%.03f' % abs(item_b)).rjust(6, '0')
                    else:
                        str_b = '+' + ('%.03f' % abs(item_b)).rjust(6, '0')
                    id_clumps.append(str_l + str_b)
                id_clumps = np.array(id_clumps)
                table_title = ['ID', 'Peak1', 'Peak2', 'Cen1', 'Cen2', 'Size1', 'Size2', 'Peak', 'Sum', 'Volume']

            elif outcat_column == 13:
                # 3d result
                peak1, peak2, peak3 = data_wcs.all_pix2world(outcat['Peak1'], outcat['Peak2'], outcat['Peak3'], 1)
                cen1, cen2, cen3 = data_wcs.all_pix2world(outcat['Cen1'], outcat['Cen2'], outcat['Cen3'], 1)
                size1, size2, size3 = np.array([outcat['Size1'] * 30, outcat['Size2'] * 30, outcat['Size3'] * 0.166])
                clustPeak, clustSum, clustVolume = np.array([outcat['Peak'], outcat['Sum'], outcat['Volume']])

                clump_Peak = np.column_stack([peak1, peak2, peak3 / 1000])
                clump_Cen = np.column_stack([cen1, cen2, cen3 / 1000])
                clustSize = np.column_stack([size1, size2, size3])
                id_clumps = []  # MWISP017.558+00.150+020.17  分别表示：银经：17.558°， 银纬：0.15°，速度：20.17km/s
                for item_l, item_b, item_v in zip(cen1, cen2, cen3 / 1000):
                    str_l = 'MWISP' + ('%.03f' % item_l).rjust(7, '0')
                    if item_b < 0:
                        str_b = '-' + ('%.03f' % abs(item_b)).rjust(6, '0')
                    else:
                        str_b = '+' + ('%.03f' % abs(item_b)).rjust(6, '0')
                    if item_v < 0:
                        str_v = '-' + ('%.03f' % abs(item_v)).rjust(6, '0')
                    else:
                        str_v = '+' + ('%.03f' % abs(item_v)).rjust(6, '0')
                    id_clumps.append(str_l + str_b + str_v)
                id_clumps = np.array(id_clumps)
                table_title = ['ID', 'Peak1', 'Peak2', 'Peak3', 'Cen1', 'Cen2', 'Cen3', 'Size1', 'Size2', 'Size3',
                               'Peak', 'Sum', 'Volume']
            else:
                print('outcat columns is %d' % outcat_column)
                return None

            outcat_wcs = np.column_stack(
                (id_clumps, clump_Peak, clump_Cen, clustSize, clustPeak, clustSum, clustVolume))
            outcat_wcs = pd.DataFrame(outcat_wcs, columns=table_title)
            return outcat_wcs


if __name__ == '__main__':
    # data_name = r'F:\LDC_python\detection\R2_data\data_9\data_9\0175+000\0175+000_L.fits'
    data_name = r'F:\LDC_python\detection\R2_data\data_9\0185-005\0185-005_L.fits'
    # data_name = r'F:\LDC_python\detection\test_data\2d_simulated_clump\gaussian_out_360.fits'
    # data_name = r'F:\LDC_python\detection\test_data\high_density_clump_data\s_out_000.fits'
    #
    # data_name = r'F:\LDC_python\detection\test_data\M16 data\hdu0_mosaic_L_3D.fits'
    data = Data(data_path=data_name)
    para = Param()

    data.summary()
    para.summary()
    # ldc = LocalDensityClust(data=data, para=para)
    # ldc.result.log()
    # ldc.result.make_plot_wcs_1()
    # ldc.result.save_result('mask.fits', 'out.fits')
    # ldc.result.save_outcat_wcs('outcat_wcs.txt')
    # ldc.result.save_outcat('outcat.txt')
    import pywt
    help(pywt.threshold)






