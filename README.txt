Car segmentation project

This directory contains the data for the segmenation of car parts by Deloitte. 

The folders:
- carseg_raw_data
    - 5doors: CAD images of a five door car with black background
    - landscape: Various landscapes which can be used for adding other backgrounds if engineered for it
    - opel: CAD images of a three door OPEL car with difficult background
    - train
        - photo: Real photos of cars
        - cycleGAN: Images generated using cycleGAN


- clean_data: Data saved in numpy format to easily start training models. 
              First 3 channels is the normalised input (the actual image), The following are the one hot vector of each 
              car part and the last channel is the mask with all car part each class encoded as an integer.

    - <photo name from carseg_raw_data>.npy


###################### TEST DATA ########################
Real car image IDs to use for testing 

Raw images: carseg_raw_data\train\photo
Cleaned data: cleaned_data\<test_id>.npy

TEST IDS:
0_a.jpg
1_a.jpg
2_a.jpg
3_a.jpg
5_a.jpg
6_a.jpg
10_a.jpg
11_a.jpg
12_a.jpg
19_a.jpg
20_a.jpg
21_a.jpg
22_a.jpg
24_a.jpg
26_a.jpg
28_a.jpg
29_a.jpg
32_a.jpg
33_a.jpg
35_a.jpg
36_a.jpg
39_a.jpg
40_a.jpg
43_a.jpg
45_a.jpg
46_a.jpg
47_a.jpg
50_a.jpg
51_a.jpg
52_a.jpg


NOTE: REMEMBER TO DELETE DATA FROM ALL SOURCES FOLLOWING THE PROJECT