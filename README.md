# Music Genre Classification using CNNs

## Overview
This project aims to develop a robust music genre classification model using Convolutional Neural Networks (CNNs). The GTZAN dataset is used as the foundation, and feature representations such as Mel-Frequency Cepstral Coefficients (MFCC) and Short-Time Fourier Transform (STFT) are extracted to capture the spectral and temporal characteristics of music. The CNN model is trained to discriminate between different genres based on the learned patterns in the combined STFT and MFCC features.

## Objectives
1. Conduct a literature review of recent papers to identify high-performing music genre classification models.
2. Expand the GTZAN dataset by adding more songs from reliable sources.
3. Implement and compare MFCC and STFT as feature extraction techniques.
4. Evaluate model performance using the original and expanded datasets to assess the impact of increased data.
5. Analyze genre-wise accuracy improvements to identify model strengths and weaknesses.
6. Compare the effectiveness of MFCC and STFT as feature selectors for the CNN model.
7. Investigate the impact of varying epoch numbers on model performance and accuracy.

## Challenges
1. Limited dataset size and diversity in commonly used benchmarks, leading to potential overfitting and reduced generalization.
2. Uncertainty about the most effective feature extraction techniques for music genre classification.
3. Lack of comprehensive studies on the impact of dataset expansion, feature selection, and training parameters on genre-wise accuracy.
4. Insufficient understanding of how CNN architectures and hyperparameters affect classification performance across genres.


## Conclusions

Training feature -MFCC/STFT
old is the original dataset
new is the original dataset with added songs in few genres which showed lesser accuracy in our literature survey.

This project is not the biggest of achievements as per the results, but we got to learn a lot. 


| Genre | MFCC-old | MFCC-new | STFT-old | STFT-New |
|-------|----------|----------|----------|----------|
| Blues | 46.67    | 13.33    | 60       | 26.67    |
| Classical | 77.27 | 61.11    | 81.82    | 61.11    |
| Country | 33.33   | 45       | 33.33    | 35       |
| Disco | 45.83    | 20.69    | 37.5     | 27.59    |
| Hip-Hop | 60      | 58.82    | 45       | 11.76    |
| Jazz | 42.11     | 50       | 47.37    | 27.27    |
| Metal | 86.36    | 81.82    | 63.64    | 81.82    |
| Pop | 61.9      | 26.32    | 61.9     | 63.16    |
| Reggae | 39.29   | 52.17    | 50       | 43.48    |
| Rock | 17.65    | 34.62    | 5.88     | 15.38    |
| Total | 52.5     | 42.04    | 50       | 38.05    |

(Accuracy in Percentage) 

*refer to  [[1]](https://doi.org/10.1109/ASYU.2018.8554016) for a good result comparison.

Probably this project will continue with more efforts towards using DSP algorithms and techniques before feature extraction to improve the accuracy. Techniques including but not restricted to :
* low, high and band pass filters, 
* better usage of percussive part of an audio in predicting the genre, 
* using spotify API to modify the dataset to automate testing for 100s of more songs. 

### To make it run:
This project was run on colab so change locations of GTZAN dataset accordingly. Most things should run fine since there are no dependencies other than the dataset itself.

Use TPM runtime as many a times the CPU runtime has timed-out or produced insufficient memory error and had to restart. 

Also, once data is preprocessed, preferably save it using python ```pickle``` and write to drive. Load it whenever wanted to train the model directly. It *will* save a ton of time on feature extraction.



### Literature Survey
1. Ahmet Elbir, H. Bilal Çam, M. Emre Iyican, B. Öztürk and N. Aydin, "Music Genre Classification and Recommendation by Using Machine Learning Techniques," *2018 Innovations in Intelligent Systems and Applications Conference (ASYU)*, Adana, Turkey, 2018, pp. 1-5, doi: [10.1109/ASYU.2018.8554016](https://doi.org/10.1109/ASYU.2018.8554016).
2. S. Vishnupriya and K. Meenakshi, "Automatic Music Genre Classification using Convolution Neural Network," *2018 International Conference on Computer Communication and Informatics (ICCCI)*, Coimbatore, India, 2018, pp. 1-4, doi: [10.1109/ICCCI.2018.8441340](https://doi.org/10.1109/ICCCI.2018.8441340).

3. N. M. Patil and M. U. Nemade, "Music Genre Classification Using MFCC, K-NN and SVM Classifier," *International Journal of Computer Engineering in Research Trends*, vol. 4, no. 2, pp. 62-68, 2017. Available: [https://ijcert.org/ems/ijcert_papers/V4I206.pdf](https://ijcert.org/ems/ijcert_papers/V4I206.pdf)

4. Ö. Kilickaya, "Genre Classification and Musical Features Analysis," *International Journal of Latest Engineering Research and Applications (IJLERA)*, vol. 9, no. 4, pp. 15-23, 2024. [Online]. Available: [https://www.researchgate.net/publication/379958695](https://www.researchgate.net/publication/379958695)

5. B. R. Maale and S. Ifath, "Neural Network Music Genre Classification," *International Journal of Research Publication and Reviews*, vol. 2, no. 7, pp. 1826-1830, 2021.

6. S. Oramas, F. Barbieri, O. Nieto, and X. Serra, "Multimodal Deep Learning for Music Genre Classification," *Transactions of the International Society for Music Information Retrieval*, vol. 1, no. 1, pp. 4-21, 2018. DOI: [https://doi.org/10.5334/tismir.10](https://doi.org/10.5334/tismir.10)

7. L. Shi, C. Li and L. Tian, "Music Genre Classification Based on Chroma Features and Deep Learning," *2019 Tenth International Conference on Intelligent Control and Information Processing (ICICIP)*, Marrakesh, Morocco, 2019, pp. 81-86, doi: [10.1109/ICICIP47338.2019.9012215](https://doi.org/10.1109/ICICIP47338.2019.9012215).

8. P. Verma, A. Chandrakar, N. Taunk, N. Agrawal, and R. Agrawal, "DeepBeats: Music Genre Classification using LSTM and RNN," *International Journal of Research Publication and Reviews*, vol. 5, no. 4, pp. 1234-1245, 2024. [Online]. Available: [https://ijrpr.com/uploads/V5ISSUE4/IJRPR25922.pdf](https://ijrpr.com/uploads/V5ISSUE4/IJRPR25922.pdf)

9. W. Zhang, W. Lei, X. Xu, and X. Xing, "Improved Music Genre Classification with Convolutional Neural Networks," in *Proceedings of Interspeech 2016*, San Francisco, CA, USA, 2016, pp. 3304-3308. [Online]. Available: [https://www.isca-archive.org/interspeech_2016/zhang16h_interspeech.pdf](https://www.isca-archive.org/interspeech_2016/zhang16h_interspeech.pdf)

10. S. K. Prabhakar and S.-W. Lee, "Holistic Approaches to Music Genre Classification using Efficient Transfer and Deep Learning Techniques," *Expert Systems with Applications*, vol. 210, 2023. [Online]. Available: [https://www.sciencedirect.com/science/article/pii/S0957417422016815](https://www.sciencedirect.com/science/article/pii/S0957417422016815)

11. H. C. Ceylan, N. Hardalaç, A. C. Kara, and F. Hardalaç, "Automatic Music Genre Classification and Its Relation with Music Education," *World Journal of Education*, vol. 11, no. 2, pp. 36-45, 2021.

12. N. Pelchat and C. M. Gelowitz, "Neural Network Music Genre Classification," in *Canadian Journal of Electrical and Computer Engineering*, vol. 43, no. 3, pp. 170-173, Summer 2020, doi: [10.1109/CJECE.2020.2970144](https://doi.org/10.1109/CJECE.2020.2970144).

13. Xie, C., Song, H., Zhu, H. *et al.* Music genre classification based on res-gated CNN and attention mechanism. *Multimed Tools Appl* **83**, 13527–13542 (2024). https://doi.org/10.1007/s11042-023-15277-1

14. S. Allamy and A. L. Koerich, "1D CNN Architectures for Music Genre Classification," *2021 IEEE Symposium Series on Computational Intelligence (SSCI)*, Orlando, FL, USA, 2021, pp. 01-07, doi: [10.1109/SSCI50451.2021.9659979](https://doi.org/10.1109/SSCI50451.2021.9659979).

15. R. S. R. Dr and I. Yaseen, "Intelligent Music Genre Classification using CNN," *International Journal of Science Technology & Engineering (IJSTE)*, vol. 8, no. 10, pp. 45-52, Apr. 2022.

16. Q. Kong, X. Feng, and Y. Li, "Music Genre Classification using Convolution Neural Network," *IAENG International Journal of Computer Science*, vol. 17, no. 1, pp. 23-36, 2014. Available: [https://ismir2014.ismir.net/LBD/LBD17.pdf](https://ismir2014.ismir.net/LBD/LBD17.pdf)

17. Y.-H. Cheng, P.-C. Chang, D.-M. Nguyen, and C.-N. Kuo, "Automatic Music Genre Classification Based on CRNN," *Engineering Letters*, vol. 29, no. 1, pp. 232-239, 2021. Available: [https://www.engineeringletters.com/issues_v29/issue_1/EL_29_1_36.pdf](https://www.engineeringletters.com/issues_v29/issue_1/EL_29_1_36.pdf)

18. L. Feng, S. Liu, and J. Yao, "Music Genre Classification with Paralleling Recurrent Convolutional Neural Network," *IEEE Transactions on Multimedia*, vol. 32, no. 2, pp. 303–319, 2017. Available: [https://arxiv.org/pdf/1712.08370v1](https://arxiv.org/pdf/1712.08370v1)

19. J. Dias, V. Pillai, A. Shah, and H. Deshmukh, "Music Genre Classification & Recommendation System using CNN," *Proceedings of the 7th International Conference on Innovations and Research in Technology and Engineering (ICIRTE-2022)*, organized by VPPCOE & VA, Mumbai-22, INDIA, 2022. Available: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4111849](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4111849)

20. S. Oramas, F. Barbieri, O. Nieto, and X. Serra, "Multimodal Deep Learning for Music Genre Classification," *Transactions of the International Society for Music Information Retrieval*, vol. 1, no. 1, pp. 4-21, 2018. Available: [https://repositori.upf.edu/bitstream/handle/10230/35647/oramas_tismir_multimodal.pdf](https://repositori.upf.edu/bitstream/handle/10230/35647/oramas_tismir_multimodal.pdf)