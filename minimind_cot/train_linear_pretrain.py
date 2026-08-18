# -*- coding: utf-8 -*-
"""Linear（Gated DeltaNet）版 pretrain 训练包装：
用 sys.modules 把 model_minimind 替换为 model_minimind_linear，然后运行 train_pretrain.py。
用法（从 minimind 根目录）:
  python trainer/train_linear_pretrain.py --save_weight linear_pretrain_yuhua_v2 \
    --data_path ../dataset/pretrain_yuhua_pure.jsonl --from_weight none ...
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import importlib
sys.modules['model.model_minimind'] = importlib.import_module('model.model_minimind_linear')
import runpy
runpy.run_path(os.path.join(ROOT, 'trainer', 'train_pretrain.py'), run_name='__main__')