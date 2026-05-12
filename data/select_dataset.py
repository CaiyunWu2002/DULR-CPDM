"""
--------------------------------------------
select dataset
--------------------------------------------
Hongyi Zheng (github: https://github.com/natezhenghy)
--------------------------------------------
Kai Zhang (github: https://github.com/cszn)
--------------------------------------------
"""

import os
from copy import deepcopy
from glob import glob
from typing import Any, Dict, List, Union

from data.dataset_polar import DatasetPolar


def select_dataset(opt_dataset: Dict[str, Any], phase: str
                   ) -> Union[DatasetPolar, List[DatasetPolar]]:
    if opt_dataset['type'] == 'polar':
        D=DatasetPolar
    else:
        raise NotImplementedError

    if phase == 'train':
        dataset = D(opt_dataset)
        return dataset
    else:
        datasets: List[DatasetPolar] = []
        paths = glob(os.path.join(opt_dataset['dataroot_H'], '*'))
        opt_dataset_sub = deepcopy(opt_dataset)
        datasets.append(D(opt_dataset_sub))
        return datasets
