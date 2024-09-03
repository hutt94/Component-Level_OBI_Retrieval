# Component-Level OBI Retrieval
Source code and dataset for ICMR'24 paper "[Component-Level Oracle Bone Inscription Retrieval](https://dl.acm.org/doi/abs/10.1145/3652583.3658116)" (Best Paper Candidate)

## To Do List
1. -[x] code
2. -[x] dataset

## Dataset
In **OBI Component 20**, we have selected 20 common OBI components. Due to the different forms each component can take, we chose representative forms to display in the following diagram. 

![image](https://github.com/user-attachments/assets/82687c53-2ead-4eb0-ab37-a13e110ccd04)

Then, we collected 11,335 OBI character images from the [小學堂](https://xiaoxue.iis.sinica.edu.tw/) based on these components. We invited [Prof. Pui-ling Tang](https://web.chinese.hku.hk/en/people/staff/113/) and Ms. Peiying Zhang from the School of Chinese, the University of Hong Kong to screen these characters, removing images that did not contain the 20 selected components, leaving us with 9,245 OBI character images. Within these images, Ms. Zhang further annotated the specific positions of the components in 1,012 OBI character images, striving to cover the different forms of the same component. Ultimately, OBI Component 20 contains a total of 10,257 OBI images, of which 9,245 are OBI characters and 1,012 are OBI components. Their distribution is as shown in the table below.

| Component| # Character | # Component | Component| # Character | # Component |
| :--: | :--: | :--: | :--: | :--: | :--: |
| 日 | 371 | 18 | 女 | 641 | 29
| 月 | 106 | 41 | 子 | 179 | 33
| 雨 | 152 | 17 | 目 | 422 | 31
| 阜 | 115 | 16 | 攴 | 414 | 91
| 水 | 622 | 41 | 止 | 1132 | 72
| 屮 | 267 | 14 | 衣 | 69 | 51
| 木 | 465 | 24 | 口 | 1592 | 42
| 犬 | 204 | 117 | 王 | 55 | 8
| 大 | 385 | 32 | 矢 | 383 |32
| 人 | 1403 | 226 | 刀 | 268 | 77

Considering copyright issues, if you need to use this dataset, please provide the following information (either in Chinese or English) in an email to cszkhu@comp.hkbu.edu.hk, and we will provide you with the dataset download link within 5 working days after receiving your email: 
1. your name,
2. your institution,
3. the intended use of the dataset,
4. and a declaration ensuring that it will not be used for commercial profit.

## Code
Train the model:
```
python train.py --componet_path components_file_name /
                --character_path characters_file_name /
                --epoch 80 /
                --batch_size=32 /
                --num_class=20
```

Test the model:
```
python test.py --componet_path components_file_name /
               --character_path characters_file_name
```

Visualize the retrieval results:
```
python visual.py --componet_path components_file_name /
                 --character_path characters_file_name /
                 --k=10
```

## Citation
```
@inproceedings{hu2024component,
  title={Component-Level Oracle Bone Inscription Retrieval},
  author={Hu, Zhikai and Cheung, Yiu-ming and Zhang, Yonggang and Zhang, Peiying and Tang, Pui-ling},
  booktitle={Proceedings of the 2024 International Conference on Multimedia Retrieval},
  pages={647--656},
  year={2024}
}
```

## Acknowledgments
We would like to thank [小學堂](https://xiaoxue.iis.sinica.edu.tw/) for sharing the public OBI data. We are also grateful to [Mr. Changxing Li](https://github.com/li1changxing) for his assistance with the data collection and code implementation.
