# adaption_screen/data

请将 `ft` 和 `pred` 脚本需要的以下数据文件放到本目录（`adaption_screen/data`）：

- `protein_embeddings.pkl` - 蛋白序列表征（ESM2 embeddings）
- `esmfold_protein_embeddings.pkl` - 蛋白结构表征（ESMFold embeddings）
- `text_embeddings.npy` - 实验上下文文本表征
- `dataset_qc.csv` - 质量控制后的原始数据集

`ft` 和 `pred` 中的脚本会通过 `../data/...` 路径读取这些文件。
