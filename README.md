# Emergence_row_alignment

This is the code related to our paper 'Geo-referencing Single Images from Consumer UAV using Image Processing' authored by Aijing Feng, Chin Nee Vong, Jing Zhou, Lance S. Conway, Jianfeng Zhou, Earl D. Vories, Kenneth A. Sudduth and Newell R. Kitchen.

Feng, A., Vong, C. N., Zhou, J., Conway, L. S., Zhou, J., Vories, E. D., ... & Kitchen, N. R. (2023). Developing an image processing pipeline to improve the position accuracy of single UAV images. Computers and Electronics in Agriculture, 206, 107650.

## Goal and novelty
The goal of this study was to develop a real-time image processing pipeline to process individual UAV images with improved geo-reference accuracy for further crop emergence mapping at field-scale. 

The novelty of this study is to provide an improved geo-reference accuracy image processing workflow in processing sequential individual UAV images.
The position accuracy had been improved to 0.17 ± 0.13 m and 0.57 ± 0.28 m (average ± standard deviation) for cotton and corn field, respectively, when compared to the accuracy obtained from an image processing workflow of a previous study (1.72 ± 1.37 m and 2.13 ± 1.89 m).
Further, the workflow also includes field-scale mapping of different crop emergence parameters such as stand count, canopy area, day after first emergence, and plant spacing standard deviation.
Meanwhile, the processing time of the workflow are 10.4 and 5.7 s/image for cotton and corn field, respectively, which is extremely lower than the time needed for image stitching from commercial software (88.7 and 97.4 s/image).

This new method can act as a low-cost real time tool to quantify crop early emergence in a shorter time and lower cost. Further application can be exploring relationship between crop emergence and environmental factors, weather conditions, and different treatments for researchers as well as field scouting for farmers especially for area inaccessible by ground vehicles.

## Workflow
![alt text](https://github.com/AJFeng/Emergence_row_alignment/blob/main/Fig3.PNG)

![alt text](https://github.com/AJFeng/Emergence_row_alignment/blob/main/Fig5.png)

Fig. 5. Crop row alignment. There were 10 cotton rows identified manually in the first image from the cotton field. The numbers of 9 cotton rows identified by the SHT in the second image frame were aligned with the first image frame based on the distance in pixels in the E-W direction (tx) from the geometric transformation matrix M. The distance in pixels in the N-S direction (ty) determined the image position within the entire crop row.

![alt text](https://github.com/AJFeng/Emergence_row_alignment/blob/main/Fig6.png)

Fig. 6. Illustration of determining image positions within each entire crop row. Black cross in each combined image represents the center position for each image in their combined image.


## Position accuracy evaluation using GCPs
![alt text](https://github.com/AJFeng/Emergence_row_alignment/blob/main/Fig8.png)

Fig. 8. Location of ground control points (GCPs) in a) corn field and b) cotton field as well as demonstration of distance comparison between two different kinds of systems for the position accuracy evaluation: b) ground RTK measurement and c) pipeline measurement.


## Mapping results
![alt text](https://github.com/AJFeng/Emergence_row_alignment/blob/main/Fig11.png)

Fig. 11. Cotton field emergence maps of a) stand count and b) canopy size with full dimension of 152 crop rows × 315 m length for each crop row and their down-sampled maps with dimension of 38 × 63 in c and d, respectively, where each data point equates to a 4 m × 5 m area. A crop yield map with the same 4 m × 5 m cell size is shown in e. 

![alt text](https://github.com/AJFeng/Emergence_row_alignment/blob/main/Fig12.png)

Fig. 12. Schematic diagram showing the treatment and non-treatment crop rows (a) and their emergence maps of stand count (plant m-1, b), mean days to imaging after emergence (DAEmean, days, c), and standard deviation of plant spacing (PSstd, cm m-1, d) in corn field.
