# Component-Level-OBI-Retrieval
Source code and dataset for ICMR'24 paper "[Component-Level Oracle Bone Inscription Retrieval](https://dl.acm.org/doi/abs/10.1145/3652583.3658116)"

## Dataset
In **OBI Component 20**, we have selected 20 common OBI components. Due to the different forms each component can take, we chose representative forms to display in the following diagram. 
![image](https://github.com/user-attachments/assets/82687c53-2ead-4eb0-ab37-a13e110ccd04)
Then, we collected 11,335 OBI character images from the [小學堂](https://xiaoxue.iis.sinica.edu.tw/) based on these components. We invited [Dr. Pui-ling Tang](https://web.chinese.hku.hk/en/people/staff/113/) and Ms. Peiying Zhang from the School of Chinese, the University of Hong Kong to screen these characters, removing images that did not contain the 20 selected components, leaving us with 9,245 OBI character images. Within these images, Ms. Zhang further annotated the specific positions of the components in 1,012 OBI character images, striving to cover the different forms of the same component. Ultimately, OBI Component 20 contains a total of 10,257 OBI images, of which 9,245 are OBI characters and 1,012 are OBI components. Their distribution is as shown in the table below.


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
