import numpy as np

def my_imfilter(image, imfilter):
    """function which imitates the default behavior of the build in scipy.misc.imfilter function.

    Input:
        image: A 3d array represent the input image.
        imfilter: The gaussian filter.
    Output:
        output: The filtered image.
    """
    # =================================================================================
    # TODO:                                                                           
    # This function is intended to behave like the scipy.ndimage.filters.correlate    
    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         
    # of the filter matrix.)                                                          
    # Your function should work for color images. Simply filter each color            
    # channel independently.                                                          
    # Your function should work for filters of any width and height                   
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This       
    # restriction makes it unambigious which pixel in the filter is the center        
    # pixel.                                                                          
    # Boundary handling can be tricky. The filter can't be centered on pixels         
    # at the image boundary without parts of the filter being out of bounds. You      
    # should simply recreate the default behavior of scipy.signal.convolve2d --       
    # pad the input image with zeros, and return a filtered image which matches the   
    # input resolution. A better approach is to mirror the image content over the     
    # boundaries for padding.                                                         
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can 
    # see the desired behavior.                                                       
    # When you write your actual solution, you can't use the convolution functions    
    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   
    # Simply loop over all the pixels and do the actual computation.                  
    # It might be slow.                        
    
    # NOTE:                                                                           
    # Some useful functions:                                                        
    #     numpy.pad (https://numpy.org/doc/stable/reference/generated/numpy.pad.html)      
    #     numpy.sum (https://numpy.org/doc/stable/reference/generated/numpy.sum.html)                                     
    # =================================================================================

    # ============================== Start OF YOUR CODE ===============================
     # 獲取輸入圖像和濾波器的維度
    image_height, image_width, num_channels = image.shape
    filter_height, filter_width = imfilter.shape

    # 確保濾波器的維度是奇數
    if filter_height % 2 == 0 or filter_width % 2 == 0:
        raise ValueError("濾波器的維度必須是奇數。")

    # 使用零初始化輸出圖像
    output = np.zeros_like(image)

    # 遍歷輸入圖像中的所有像素
    for i in range(image_height):
        for j in range(image_width):
            for k in range(num_channels):
                # 初始化濾波後的值為零
                filtered_value = 0.0

                # 遍歷濾波器的元素
                for m in range(filter_height):
                    for n in range(filter_width):
                        # 計算在輸入圖像中的坐標
                        ii = i + m - filter_height // 2
                        jj = j + n - filter_width // 2

                        # 檢查邊界條件
                        if ii >= 0 and ii < image_height and jj >= 0 and jj < image_width:
                            # 進行卷積操作：相乘並累加
                            filtered_value += image[ii, jj, k] * imfilter[m, n]

                # 將濾波後的值賦予輸出像素
                output[i, j, k] = filtered_value
    
    # =============================== END OF YOUR CODE ================================

    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can
    # see the desired behavior.
    # import scipy.ndimage as ndimage
    # output = np.zeros_like(image)
    # for ch in range(image.shape[2]):
    #    output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')

    return output