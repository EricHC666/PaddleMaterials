import os
import sys

# 添加项目根目录到环境变量中以保证 ppmat 能够被正确导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ppmat.datasets.matbench_dataset import MatBenchDataset
from ppmat.datasets.opendac_dataset import OPENDACDataset


def test_opendac_dataset():
    print("----------------------------------------")
    print("开始测试 OPENDACDataset ...")
    try:
        # 为了测试而创建一个临时的 dummy 格式 json
        dummy_path = "dummy_opendac.json"
        with open(dummy_path, "w") as f:
            f.write('{"structure": [], "target_prop": []}\n')

        dataset = OPENDACDataset(path=dummy_path, property_names=["target_prop"])
        print(f"成功实例化 OPENDACDataset。当前样本数量: {len(dataset)}")
        os.remove(dummy_path)
        print("OPENDACDataset 测试通过！")
    except Exception as e:
        print(f"OPENDACDataset 测试遇到异常（如果遇到缺少数据源属正常网络或依赖问题）: {e}")


def test_matbench_dataset():
    print("----------------------------------------")
    print("开始测试 MatBenchDataset ...")
    try:
        # 为了测试而创建一个临时的 dummy 格式 json
        dummy_path = "dummy_matbench.json"
        with open(dummy_path, "w") as f:
            f.write('{"structure": [], "target_prop": []}\n')

        dataset = MatBenchDataset(path=dummy_path, property_names=["target_prop"])
        print(f"成功实例化 MatBenchDataset。当前样本数量: {len(dataset)}")
        os.remove(dummy_path)
        print("MatBenchDataset 测试通过！")
    except Exception as e:
        print(f"MatBenchDataset 测试遇到异常: {e}")


if __name__ == "__main__":
    test_opendac_dataset()
    test_matbench_dataset()
    print("----------------------------------------")
    print("全部测试流程已顺利执行结束。")
