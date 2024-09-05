# Dataset
## Data Naming Rules
1. All data containing the same component are placed in the same folder. For example, folder 0 contains all data with the component "sun".
2. Within each folder, files ending with "_a" indicate annotated components, while the rest are OBI characters.
3. The naming convention for OBI character data is:
```
#A_#Bno#C.png
```
- #A is a number starting from 1, used to count the number of characters downloaded from Xiaoxue Tang. After expert screening, some data were removed, so it is not always continuous.
- #B is the ID of the character in [小學堂](https://xiaoxue.iis.sinica.edu.tw/).
- #C represents the #C-th form of the character, as the same character can have different forms.

## Data Format
The data provided by [小學堂](https://xiaoxue.iis.sinica.edu.tw/) is in PNG format. To facilitate your use of the data, we use MATLAB as an example to demonstrate the data format.
```
% Reading character data:
[a,~,c]=imread('1_43no1.png');
subplot(121);imshow(a); % A is an all-zero 401*413*3 matrix.
subplot(122);imshow(c); % C is a 401*413 matrix representing transparency.
```
![4bc36e98e550d920a75e8800d27147f](https://github.com/user-attachments/assets/c0e21bcb-8f2e-445e-a2be-7d8c7f9002ac)

```
% Reading component data:
[a,~,c]=imread('1_43no1_a.png');
subplot(121);imshow(a); % A contains the component marked in red.
subplot(122);imshow(c); % C is a 401*413 matrix representing transparency.
```
![9477c4c6a9595ca0d625a2dd60d1d12](https://github.com/user-attachments/assets/87c0c246-828f-4db5-a925-dde10b1fddf0)

You can also refer to the [preprocess](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datasets.py) function in this code to handle the data.

## Data Division
We use the data from [TR_OBI_cha.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datalist/TR_OBI_cha.txt) and [TR_OBI_com.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datalist/TR_OBI_com.txt) for training.
The data from [TE_OBI_com.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datalist/TE_OBI_com.txt) is used as the query set, and the data from [R_OBI_cha.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datalist/R_OBI_cha.txt) is used as the retrieval set.

You will find that the number of OBI characters in [R_OBI_cha.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datalist/R_OBI_cha.txt) differs from the division in the paper. This is because the collected dataset contains characters that include multiple components (these characters are recorded in [mapper.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/mapper.txt)). During the retrieval phase, we combined these characters. This also means that when calculating mAP or precision, both [R_OBI_cha.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datalist/R_OBI_cha.txt) and [mapper.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/mapper.txt) need to be referenced. Relevant code is provided in [line 87 of test.py](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/test.py) for your reference.

Feel free to discuss any issues you encounter in the issue section, or you can also reach out via email (cszkhu@comp.hkbu.edu.hk).
