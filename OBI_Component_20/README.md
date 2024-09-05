# Dataset
## Data Naming Rules
All data containing the same component are placed in the same folder. For example, folder 0 contains all data with the component "sun".
Within each folder, files ending with "_a" indicate annotated components, while the rest are OBI characters.
The naming convention for OBI character data is:
```
#A_#Bno#C.png
```
- #A is a number starting from 1, used to count the number of characters downloaded from Xiaoxue Tang. After expert screening, some data were removed, so it is not always continuous.
- #B is the ID of the character in Xiaoxue Tang.
- #C represents the #C-th form of the character, as the same character can have different forms.

## Data Format
The data provided by Xiaoxue Tang is in PNG format. To facilitate your use of the data, we use MATLAB as an example to demonstrate the data format.
```
% Reading character data:
[a,~,c]=imread('1_43no1.png');
subplot(121);imshow(a); % A is a 401*413*3 matrix filled with zeros.
subplot(122);imshow(c); % C is a 401*413 matrix representing transparency.
```
![f647f84134041771bd2d7f35f999ad1](https://github.com/user-attachments/assets/3993681d-5449-4954-a8d3-6befb0fb213a)

```
% Reading component data:
[a,~,c]=imread('1_43no1_a.png');
subplot(121);imshow(a); % A contains the component marked in red.
subplot(122);imshow(c); % C is a 401*413 matrix representing transparency.
```
![48dd549405baabf373f930554fdc78a](https://github.com/user-attachments/assets/31ea15dd-e6f5-4db8-9739-f3c5f4f417e1)

You can also refer to the [preprocess](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datasets.py) function in this code to handle the data.

## Data Division
We use the data from [TR_OBI_cha.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datalist/TR_OBI_cha.txt) and [TR_OBI_com.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datalist/TR_OBI_com.txt) for training.
The data from [TE_OBI_com.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datalist/TE_OBI_com.txt) is used as the query set, and the data from [R_OBI_cha.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/datalist/R_OBI_cha.txt) is used as the retrieval set.

You will find that the number of OBI characters in R_OBI_cha.txt differs from the division in the paper. This is because the collected dataset contains characters that include multiple components (these characters are recorded in [mapper.txt](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/mapper.txt)). During the retrieval phase, we combined these characters. This also means that when calculating mAP or precision, both R_OBI_cha.txt and mapper.txt need to be referenced. Relevant code is provided in line 87 of [test.py](https://github.com/hutt94/Component-Level_OBI_Retrieval/blob/main/test.py) for your reference.

Feel free to discuss any issues you encounter in the issue section, or you can also reach out via email.
