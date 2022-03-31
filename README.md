# imageProcessing_Ex1

## brief:
• Loading grayscale and RGB image representations.</br>
• Displaying figures and images.</br>
• Transforming RGB color images back and forth from the YIQ color space.</br>
• Performing intensity transformations: histogram equalization.</br>
• Performing optimal quantization</br>

## ex1_utils
1. imReadAndConvert():
    Reads an image, and returns the image converted as requested (GRAY_SCALE (1) or RGB (2))
  
2. imDisplay(): </br>
    Reads an image as RGB or GRAY_SCALE and displays it</br>
    ![image](https://user-images.githubusercontent.com/77155986/161082709-e35f04eb-fb9b-4311-9250-2990a9af5ae5.png)
    ![image](https://user-images.githubusercontent.com/77155986/161082781-a0db1699-7e5b-4665-936a-e6fc92ee16f4.png) </br>
4. transformRGB2YIQ(): </br>
    Converts an RGB image to YIQ color space</br>
    ![image](https://user-images.githubusercontent.com/77155986/161083126-e16bff4f-1cf0-4d73-9069-a3f8e867830a.png)</br>
6. transformYIQ2RGB(): </br>Converts an YIQ image to RGB color space
7. hsitogramEqualize(): </br>
    Equalizes the histogram of an image https://en.wikipedia.org/wiki/Histogram_equalization</br>
    ![image](https://user-images.githubusercontent.com/77155986/161083362-5dd45a79-e8e4-4cb5-a58e-2ae025396447.png)
    ![image](https://user-images.githubusercontent.com/77155986/161083434-d8597522-e2cc-488a-85f3-8522b9717767.png)</br>

9. quantizeImage(): </br>Quantized an image in to **nQuant** colors, in nIter iterations
    https://en.wikipedia.org/wiki/Quantization </br>
    * isGray(): if RGB -> convert to YIQ -> take Y, elif GRAY -> Y = imgOrig</br>
    ![image](https://user-images.githubusercontent.com/77155986/161083607-b182bc08-f0c5-4631-845d-137ab536a1a5.png)</br>


## gamma
gammaDisplay(): GUI for gamma correction. https://en.wikipedia.org/wiki/Gamma_correction
    * gammaCorrect(): this func gets the desired gamma from trackbar, and by LUT makes the correct img. </br>
    0 = dark, 200 = bright <\br>
    Vout = pow(Vin,(1/gamma))<\br>
    * getWidthHeight(): returns resized img width and height.

python version: 3.8
