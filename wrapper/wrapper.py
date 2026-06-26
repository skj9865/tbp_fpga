import sys
import os
import json
from types import SimpleNamespace

# Add ff_algorithm directory to path so its internal imports work
_FF_DIR = os.path.dirname(os.path.abspath(__file__))
if _FF_DIR not in sys.path:
    sys.path.insert(0, _FF_DIR)

from core.base_algorithm import BaseAlgorithm
from core.registry import register_algorithm


# Dataset name mapping: framework name -> FF internal name
_DATASET_MAP = {
    "cifar10": "cifar",
    "cifar": "cifar",
    "mnist": "mnist",
    "svhn": "svhn",
}


@register_algorithm
class ForwardForwardAlgorithm(BaseAlgorithm):

    def __init__(self):
        self._config = {}
        self._dataset = "cifar"
        self._device_num = 0
        self._batchsize = 100
        self._seed = 1234

    def name(self) -> str:
        return "ff"

    def configure(self, config: dict) -> None:
        self._config = config
        dataset = config.get("dataset", config.get("default_dataset", "cifar"))
        self._dataset = _DATASET_MAP.get(dataset, dataset)
        self._device_num = int(config.get("device", "cuda:0").split(":")[-1])
        self._batchsize = config.get("batchsize", 100)
        self._seed = config.get("seed", 1234)

    def get_supported_datasets(self) -> list:
        return ["cifar10", "mnist", "svhn"]

    def _load_args_and_loaders(self):
        """Load FF-native arguments and data loaders based on dataset."""
        # Change to ff_algorithm dir so config.json is found
        original_dir = os.getcwd()
        os.chdir(_FF_DIR)

        try:
            from utils.orig_scff.get_arguments import get_arguments_cifar, get_argument_mnist
            from utils.orig_scff.get_loaders import get_train_cifar, get_train_mnist, get_train_svhn

            if self._dataset == "mnist":
                args = get_argument_mnist()
                loaders = get_train_mnist(batchsize=self._batchsize)
            elif self._dataset == "svhn":
                args = get_arguments_cifar()
                loaders = get_train_svhn(batchsize=self._batchsize)
            else:  # cifar (default)
                args = get_arguments_cifar()
                loaders = get_train_cifar(batchsize=self._batchsize, augment="no")

            return args, loaders
        finally:
            os.chdir(original_dir)

    def train(self, **kwargs) -> dict:
        """Run FF training and evaluation."""
        original_dir = os.getcwd()
        os.chdir(_FF_DIR)

        try:
            from main_SCFF import main
            args, loaders = self._load_args_and_loaders()
            dims = (1, 2, 3)

            accuracy = main(
                device_num=self._device_num,
                tr_and_eval=1,  # train + evaluate
                save_model=True,
                loaders=loaders,
                NL=args.NL,
                lr=args.lr,
                weight_decay=args.weight_decay,
                gamma=args.gamma,
                lamda=args.lamda,
                threshold1=args.th1,
                threshold2=args.th2,
                act=args.act,
                concats=args.concats,
                alleps=args.alleps,
                seed_num=self._seed,
                dims=dims,
                period=args.period,
                mode=self._dataset,
            )
            return {"accuracy": accuracy, "dataset": self._dataset}
        finally:
            os.chdir(original_dir)

    def evaluate(self, **kwargs) -> dict:
        """Run FF evaluation only (no training)."""
        original_dir = os.getcwd()
        os.chdir(_FF_DIR)

        try:
            from main_SCFF import main
            args, loaders = self._load_args_and_loaders()
            dims = (1, 2, 3)

            accuracy = main(
                device_num=self._device_num,
                tr_and_eval=0,  # evaluate only
                save_model=False,
                loaders=loaders,
                NL=args.NL,
                lr=args.lr,
                weight_decay=args.weight_decay,
                gamma=args.gamma,
                lamda=args.lamda,
                threshold1=args.th1,
                threshold2=args.th2,
                act=args.act,
                concats=args.concats,
                alleps=args.alleps,
                seed_num=self._seed,
                dims=dims,
                period=args.period,
                mode=self._dataset,
            )
            return {"accuracy": accuracy, "dataset": self._dataset}
        finally:
            os.chdir(original_dir)
