# Component-Level OBI Retrieval
Source code and dataset for ICMR'24 paper "[Component-Level Oracle Bone Inscription Retrieval](https://dl.acm.org/doi/abs/10.1145/3652583.3658116)" (Best Paper Candidate)

## To Do List
1. -[] Task Definition
2. -[x] code
3. -[x] dataset
4. -[] Methods
5. -[] Results and Analysis

## Dataset
In **OBI Component 20**, we have selected 20 common OBI components. Due to the different forms each component can take, we chose representative forms to display in the following diagram. 

![image](https://github.com/user-attachments/assets/82687c53-2ead-4eb0-ab37-a13e110ccd04)

Then, we collected 11,335 OBI character images from the [小學堂](https://xiaoxue.iis.sinica.edu.tw/) based on these components. We invited [Prof. Pui-ling Tang](https://web.chinese.hku.hk/en/people/staff/113/) and Ms. Peiying Zhang from the School of Chinese, the University of Hong Kong to screen these characters, removing images that did not contain the 20 selected components, leaving us with 9,245 OBI character images. Within these images, Ms. Zhang further annotated the specific positions of the components in 1,012 OBI character images, striving to cover the different forms of the same component. Ultimately, OBI Component 20 contains a total of 10,257 OBI images, of which 9,245 are OBI characters and 1,012 are OBI components. Their distribution is as shown in the table below.

| ID | Component| # Character | # Component | ID | Component| # Character | # Component |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 0 | 日 | 371 | 18 | 10 | 女 | 641 | 29
| 1 | 月 | 106 | 41 | 11 | 子 | 179 | 33
| 2 | 雨 | 152 | 17 | 12 | 目 | 422 | 31
| 3 | 阜 | 115 | 16 | 13 | 攴 | 414 | 91
| 4 | 水 | 622 | 41 | 14 | 止 | 1132 | 72
| 5 | 屮 | 267 | 14 | 15 | 衣 | 69 | 51
| 6 | 木 | 465 | 24 | 16 | 口 | 1592 | 42
| 7 | 犬 | 204 | 117 | 17 | 王 | 55 | 8
| 8 | 大 | 385 | 32 | 18 | 矢 | 383 |32
| 9 | 人 | 1403 | 226 | 19 | 刀 | 268 | 77

For more details about the dataset, please refer to [here](https://github.com/hutt94/Component-Level_OBI_Retrieval/tree/main/OBI_Component_20).

Considering copyright issues, if you need to use this dataset, please provide the following information (either in Chinese or English) in an email to cszkhu@comp.hkbu.edu.hk, and we will provide you with the dataset download link within 5 working days after receiving your email (**It should be a valid .edu email that matches your institution**): 
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
