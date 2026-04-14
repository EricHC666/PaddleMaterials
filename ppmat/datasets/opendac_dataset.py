# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import annotations

import math
import os
import os.path as osp
import pickle
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle.distributed as dist
from paddle.io import Dataset

from ppmat.datasets.build_structure import BuildStructure
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.models import build_graph_converter
from ppmat.utils import download
from ppmat.utils import logger
from ppmat.utils.io import count_samples_json_lines
from ppmat.utils.io import read_json_lines
from ppmat.utils.misc import is_equal


class OPENDACDataset(Dataset):
    """OPENDAC 数据集处理类。

    该数据集的处理类完全参考自 mp2024_dataset.py，输入输出格式能够无缝适配 MEGNet 等模型。

    **数据格式**
    与 mp2024 一致，数据集中的每个样本都被表示为一个 `dict`。包含材料的各种物理、化学属性以及结构（structure）等。

    参数说明:
        path (str): 数据集的本地保存路径。如果路径下不存在文件，则会自动下载。
        property_names (Optional[list[str]], optional): 你想在训练或测试中加载的属性名称列表。默认为 None。
        build_structure_cfg (Dict, optional): 用户通过从字典或 CIF 字符串构建 Pymatgen 晶体结构的配置。如果不指定则采用默认设置。默认为 None。
        build_graph_cfg (Dict, optional): 将结构数据构建为图数据的配置，适配特定图网络时的关键参数。默认为 None。
        transforms (Optional[Callable], optional): 加载每个样本时使用的预处理/转换函数。默认为 None。
        cache_path (Optional[str], optional): 如果设置了此缓存路径，解析后的结构与图数据将直接写入/读取该路径。默认为 None。
        overwrite (bool, optional): 是否强制覆盖现有的缓存文件。默认为 False。
        filter_unvalid (bool, optional): 是否过滤掉无效的样本（例如缺失目标属性或者是无效的图数据的样本）。默认为 True。
    """

    # OpenDAC 占位链接和信息（可根据实际的下载链接和 MD5 修改）
    name = "opendac_train"
    url = "https://paddle-org.bj.bcebos.com/paddlematerial/datasets/opendac/opendac_train.zip"
    md5 = "00000000000000000000000000000000"

    def __init__(
        self,
        path: str,
        property_names: Optional[list[str]] = None,
        build_structure_cfg: Dict = None,
        build_graph_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache_path: Optional[str] = None,
        overwrite: bool = False,
        filter_unvalid: bool = True,
        **kwargs,  # 用于传参兼容性
    ):
        super().__init__()

        # 如果给定的数据集路径不存在，则自动下载数据集
        if not osp.exists(path):
            logger.message("未找到该数据集，将自动进行下载。")
            root_path = download.get_datasets_path_from_url(self.url, self.md5)
            path = osp.join(root_path, self.name, osp.basename(path))

        self.path = path
        # 兼容处理单属性名字符串输入为列表
        if isinstance(property_names, str):
            property_names = [property_names]

        # 如果没有给定结构的配置，填入默认值
        if build_structure_cfg is None:
            build_structure_cfg = {
                "format": "dict",
                "primitive": False,
                "niggli": True,
                "num_cpus": 1,
            }
            logger.message(
                "未指定 build_structure_cfg 结构构建配置，将使用默认配置: " f"{build_structure_cfg}"
            )

        self.property_names = property_names if property_names is not None else []
        self.build_structure_cfg = build_structure_cfg
        self.build_graph_cfg = build_graph_cfg
        self.transforms = transforms

        # 确定缓存路径
        if cache_path is not None:
            self.cache_path = cache_path
        else:
            # 根据图转换器名称与截断半径（cutoff）等动态生成缓存文件夹名
            if build_graph_cfg is not None:
                graph_converter_name = re.sub(
                    r"(?<!^)([A-Z])",
                    r"_\1",
                    build_graph_cfg.get("__class_name__", "Converter"),
                ).lower()
                cutoff_name = str(
                    int(build_graph_cfg.get("__init_params__", {}).get("cutoff", 5))
                )
                self.cache_path = osp.join(
                    osp.split(path)[0]
                    + "_cache_"
                    + graph_converter_name
                    + "_cutoff_"
                    + cutoff_name,
                    osp.splitext(osp.basename(path))[0],
                )
            else:
                self.cache_path = osp.join(
                    osp.split(path)[0] + "_cache_structures",
                    osp.splitext(osp.basename(path))[0],
                )
        logger.info(f"缓存路径设置为: {self.cache_path}")
        os.makedirs(self.cache_path, exist_ok=True)

        self.overwrite = overwrite
        self.filter_unvalid = filter_unvalid

        # 计算原始数据文件中的样本总数
        if osp.exists(self.path):
            num_samples_raw_file = count_samples_json_lines(self.path)
            logger.info(f"原始记录文件包含 {num_samples_raw_file} 个样本。")
        else:
            num_samples_raw_file = 0
            logger.warning("未找到原始记录文件。")

        # 检查是否已缓存我们需要的属性数据
        property_cache_path = osp.join(self.cache_path, "properties")
        if osp.exists(property_cache_path):
            try:
                for property_name in self.property_names:
                    data = self.load_from_cache(
                        osp.join(property_cache_path, f"{property_name}.pkl"),
                    )
                    logger.info(
                        f"从缓存路径 {property_cache_path} 中加载了 {len(data)} 个 "
                        f"{property_name} 属性值。"
                    )
                    if len(data) != num_samples_raw_file:
                        logger.warning(
                            f"缓存中的属性数目 ({len(data)}) 与原始样本数 ({num_samples_raw_file}) "
                            f"不匹配，请检查是否需要覆盖缓存重新生成。"
                        )
                logger.info("属性缓存已找到，将直接从缓存加载。")
            except Exception as e:
                logger.warning(e)
                logger.warning(f"从缓存中加载属性文件失败，将重新构建缓存的属性。")
                overwrite = True
        else:
            logger.info("没有找到属性缓存文件夹，将读取并建立缓存。")
            overwrite = True

        # 检查是否所有的晶体结构都已经生成并缓存了
        structure_cache_path = osp.join(self.cache_path, "structures")
        if osp.exists(structure_cache_path) and not overwrite:
            logger.info("已找到晶体结构的缓存，将直接从中加载。")

            files_structure = [
                f for f in os.listdir(structure_cache_path) if f.endswith(".pkl")
            ]
            num_cached_structures = len(files_structure)
            if num_samples_raw_file == num_cached_structures:
                logger.info(f"所有的原始数据都已成功转为晶体结构缓存，其总数为 {num_cached_structures}。")
            else:
                logger.warning(
                    f"缓存的晶体结构数量 ({num_cached_structures}) 与原始样本量 "
                    f"({num_samples_raw_file}) 不匹配。若有问题请设置覆盖写入。"
                )
        else:
            logger.info("结构缓存未找到，开始构建结构缓存...")
            os.makedirs(structure_cache_path, exist_ok=True)
            os.makedirs(property_cache_path, exist_ok=True)
            self.row_data, self.num_samples = self.read_data(path)
            logger.info(f"从 {path} 中成功加载了 {self.num_samples} 条数据。")
            self.property_data = self.read_property_data(
                self.row_data, self.property_names
            )

            # 只在第 0 号进程中进行数据的解析构建，从而避免多卡并行时的互相干扰
            if dist.get_rank() == 0:
                self.save_to_cache(
                    osp.join(self.cache_path, "build_structure_cfg.pkl"),
                    build_structure_cfg,
                )
                # 利用相关配置将原始数据转化为 Pymatgen Structure 等对象
                structures = BuildStructure(**build_structure_cfg)(
                    self.row_data.get("structure", [])
                )

                # 分布式循环缓存每一个样本的结构
                for i in range(self.num_samples):
                    self.save_to_cache(
                        osp.join(structure_cache_path, f"{i:010d}.pkl"),
                        structures[i],
                    )
                logger.info(f"成功地将 {self.num_samples} 个结构保存至 {structure_cache_path}")

                # 一并缓存属性信息
                for property_name in self.property_names:
                    data = self.property_data[property_name]
                    self.save_to_cache(
                        osp.join(property_cache_path, f"{property_name}.pkl"),
                        data,
                    )
                    logger.info(
                        f"成功缓存 {self.num_samples} 条属性 {property_name} 到 "
                        f"{property_cache_path}"
                    )

            # 同步所有分布式进程
            if dist.is_initialized():
                dist.barrier()

        # 图缓存部分的检验和建立配置
        graph_cache_path = osp.join(self.cache_path, "graphs")
        graph_cache_exists = build_graph_cfg is not None and osp.exists(
            graph_cache_path
        )
        if graph_cache_exists and not overwrite:
            try:
                build_graph_cfg_cache = self.load_from_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl")
                )
                if is_equal(build_graph_cfg_cache, build_graph_cfg):
                    logger.info("缓存中的图构建配置(build_graph_cfg)与目前的设置一致，将会复用缓存图数据。")
                else:
                    logger.warning("构建的图配置与此前缓存中的图配置存在差异，将重新构建并覆盖。")
                    overwrite = True
            except Exception as e:
                logger.warning(e)
                logger.warning("未能读取构建图的配置文件，准备强制重新构图。")
                overwrite = True

        if (
            build_graph_cfg is not None
            and osp.exists(graph_cache_path)
            and not overwrite
        ):
            logger.info("图数据的缓存已经成功找到，准备从中加载。")
            files_graph = [
                f for f in os.listdir(graph_cache_path) if f.endswith(".pkl")
            ]
            num_cached_graphs = len(files_graph)
            if (
                hasattr(self, "num_cached_structures")
                and self.num_cached_structures == num_cached_graphs
            ):
                pass
            else:
                pass
        elif build_graph_cfg is not None:
            logger.info("未能找到相关图的缓存文件。将启动构建过程...")
            os.makedirs(graph_cache_path, exist_ok=True)
            if dist.get_rank() == 0:
                self.save_to_cache(
                    osp.join(self.cache_path, "build_graph_cfg.pkl"), build_graph_cfg
                )
                converter = build_graph_converter(build_graph_cfg)

                # 如果此前是在缓存中，此处需要先调出所有的结构数据，再去转为 graph
                structures = [
                    self.load_from_cache(
                        osp.join(structure_cache_path, f"{i:010d}.pkl")
                    )
                    for i in range(num_samples_raw_file)
                ]

                graphs = converter(structures)
                for i in range(len(graphs)):
                    self.save_to_cache(
                        osp.join(graph_cache_path, f"{i:010d}.pkl"), graphs[i]
                    )
                logger.info(f"保存了图至: {graph_cache_path}")

            if dist.is_initialized():
                dist.barrier()

        # 获取最终的属性清单及其索引列表
        self.property_data = {
            property_name: self.load_from_cache(
                osp.join(property_cache_path, f"{property_name}.pkl")
            )
            for property_name in self.property_names
        }

        self.structures = [
            osp.join(structure_cache_path, f)
            for f in sorted(
                os.listdir(structure_cache_path),
                key=lambda x: int(x.replace(".pkl", "")),
            )
        ]

        if build_graph_cfg is not None:
            self.graphs = [
                osp.join(graph_cache_path, f)
                for f in sorted(
                    os.listdir(graph_cache_path),
                    key=lambda x: int(x.replace(".pkl", "")),
                )
            ]
        else:
            self.graphs = None

        self.num_samples = len(self.structures)

        # 过滤无效属性
        if filter_unvalid:
            self.filter_unvalid_by_property()
        if self.graphs is not None:
            self.filter_unvalid_by_graph()

    def filter_unvalid_by_graph(self):
        """剔除那些不具有合法或者有效边的图样本对象"""
        reserve_idx = []
        for i, g in enumerate(self.graphs):
            data = self.load_from_cache(g)
            if data is not None:
                reserve_idx.append(i)

        for key in self.property_data.keys():
            self.property_data[key] = [self.property_data[key][i] for i in reserve_idx]
        self.structures = [self.structures[i] for i in reserve_idx]
        self.graphs = [self.graphs[i] for i in reserve_idx]
        logger.warning(f"由于图数据合法性的筛查，丢弃了之后保留: {len(reserve_idx)} 个样本。")
        self.num_samples = len(self.structures)

    def read_data(self, path: str):
        """读取指定的数据文件

        参数：
            path (str): 数据所在的具体路径
        """
        data = read_json_lines(path)
        num_samples = len(data.get("structure", []))
        return data, num_samples

    def read_property_data(self, data: Dict, property_names: list[str]):
        """根据给定的属性名从原初数据中读取其数值。"""
        property_data = {}
        for property_name in property_names:
            if property_name not in data:
                raise ValueError(f"原始数据中未包含您申明的属性名： {property_name}")
            property_data[property_name] = data[property_name]
        return property_data

    def save_to_cache(self, cache_path: str, data: Any):
        """通过 Pickle 将结构、图或者配置文件保存到缓存中"""
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_from_cache(self, cache_path: str):
        """从二进制 PKL 缓存内加载并反序列化"""
        if osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"找不到对应的缓存文件：{cache_path}")

    def filter_unvalid_by_property(self):
        """丢弃目标属性值为 NaN、Null、Inf 等无用属性的目标样本。"""
        for property_name in self.property_names:
            data = self.property_data[property_name]
            reserve_idx = []
            for i, data_item in enumerate(data):
                if isinstance(data_item, str) or (
                    data_item is not None and not math.isnan(float(data_item))
                ):
                    reserve_idx.append(i)
            # 全量更新对应所有内部指标
            for key in self.property_data.keys():
                self.property_data[key] = [
                    self.property_data[key][i] for i in reserve_idx
                ]

            self.structures = [self.structures[i] for i in reserve_idx]
            if self.graphs is not None:
                self.graphs = [self.graphs[i] for i in reserve_idx]
            logger.warning(
                f"对目标属性 '{property_name}' 进行了过滤， 保留了包含有效数据的样本 {len(reserve_idx)} 条"
            )
        self.num_samples = len(self.structures)

    def get_structure_array(self, structure):
        """提取 pymatgen structure 对象的相应字段生成字典数组。"""
        atom_types = np.array([site.specie.Z for site in structure])
        lattice_parameters = structure.lattice.parameters
        lengths = np.array(lattice_parameters[:3], dtype="float32").reshape(1, 3)
        angles = np.array(lattice_parameters[3:], dtype="float32").reshape(1, 3)
        lattice = structure.lattice.matrix.astype("float32")

        structure_array = {
            "frac_coords": ConcatData(structure.frac_coords.astype("float32")),
            "cart_coords": ConcatData(structure.cart_coords.astype("float32")),
            "atom_types": ConcatData(atom_types),
            "lattice": ConcatData(lattice.reshape(1, 3, 3)),
            "lengths": ConcatData(lengths),
            "angles": ConcatData(angles),
            "num_atoms": ConcatData(np.array([tuple(atom_types.shape)[0]])),
        }
        return structure_array

    def __getitem__(self, idx: int):
        """加载数据集中特定索引位置索引元素供外部模型进行利用。

        通过读取已缓存好的图或结构及其属性返回可供给 Paddle 神经网络的包含各类特征张量的字典！

        Args:
            idx (int): 数据集中对应样本下标

        Returns:
            dict: 包含目标属性、材料 ID 以及材料拓扑 Graph 等张量的词典。
        """
        data = {}
        # 判断有无生成 graph 数据
        if self.graphs is not None:
            graph = self.graphs[idx]
            if isinstance(graph, str):
                graph = self.load_from_cache(graph)
            data["graph"] = graph
        else:
            structure = self.structures[idx]
            if isinstance(structure, str):
                structure = self.load_from_cache(structure)
            data["structure_array"] = self.get_structure_array(structure)

        # 根据我们索要的内容塞属性字段的 Float 向量
        for property_name in self.property_names:
            if property_name in self.property_data:
                data[property_name] = np.array(
                    [self.property_data[property_name][idx]]
                ).astype("float32")
            else:
                raise KeyError(f"未找到属性 {property_name} 。")

        data["id"] = (
            self.property_data.get("id", [])[idx] if "id" in self.property_data else idx
        )

        # 将我们自定义的一些算数转换器变换加给我们的张量（如 Scaling）
        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        """返回本数据集所拥有样本的总量"""
        return self.num_samples
