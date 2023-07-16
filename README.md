# **Book Recommendation Algorithm**

This is the recommendation algorithm used for the [Boka](https://github.com/kieubaduong/Boka) application - a book reading application.
## **Instructor**

- **Instructor:** Nguyen Tan Toan

## **Project Directory Structure**

```
├── archive
│   ├── crawled_dataset
│   ├── processed_dataset
│   └── raw_dataset
├── evaluate-precision
│   ├── item-based
│   └── user-based
├── preprocessing
│   ├── book_dataset
│   ├── rating_dataset
│   └── user_dataset
├── src
│   ├── content-based
│   ├── item-based
│   └── user-based
├── test
└── traditional-approach

```

## **Directory Description**

- **archive**: Contains dataset collections, including CSV files. This directory holds the original data, data after crawling, and data after preprocessing.
- **preprocessing**: Contains functions for processing each dataset.
- **traditional-approach**: Includes a notebook documenting the entire preprocessing process and the recommendation algorithm using classical methods on the original dataset (without additional crawled fields).
- **src**: Contains models applying three popular algorithms, including content-based and collaborative filtering.
- **evaluate-precision**: Contains models modifying the input and output of algorithms in the **`src`** directory to evaluate the accuracy of each algorithm.

## **Google Colab Notebook**

You can find the Google Colab notebook for this project [here](https://drive.google.com/drive/folders/17nWzjQ0EDQM8JXHJQQ77ASAmPgYDcq7b?usp=sharing)

<table>
  <tr>
    <td align="center">
        <img src="https://avatars.githubusercontent.com/u/75083331?v=4" width="100px;" alt=""/>
                <br />
                <sub><b>Kieu Ba Duong</b></sub>
                </a><br />
                <sub>Mobile developer</sub>
                <br />
                <sub>ML reseacher</sub>
    <td align="center"><img src="https://avatars.githubusercontent.com/u/75603028?v=4" width="100px;" alt=""/><br />
                <sub><b>Do Thanh Dat</b></sub>
                </a><br />
                <sub>Backend developer</sub>
                <br />
                <sub>Project manager</sub>
  </tr>
  
</table>
