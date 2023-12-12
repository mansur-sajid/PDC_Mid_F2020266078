#include <stdio.h>
#include <stdlib.h>
#include<omp.h>


#define IMAGE_SIZE 15000
#define KERNEL_SIZE 3

#define IMAGE_WIDTH 15000
#define IMAGE_HEIGHT 15000


typedef struct {
 int red;
 int green;
 int blue;
} Pixel;

void blur_image(int **image, float kernel[KERNEL_SIZE][KERNEL_SIZE], int **result) {
	#pragma omp parallel for collapse(2)
	for (int i = 1; i < IMAGE_SIZE - 1; ++i) {
		for (int j = 1; j < IMAGE_SIZE - 1; ++j) {
			float sum = 0.0;
			#pragma omp simd reduction(+:sum)
			for (int m = -1; m <= 1; ++m) {
				for (int n = -1; n <= 1; ++n) {
					sum += image[i + m][j + n] * kernel[m + 1][n + 1];
				}
			}
			result[i][j] = (int)(sum + 0.5); // Round to the nearest integer for simplicity
		}
	}
}

void grayscaleConversion(Pixel **inputImage, Pixel **outputImage) {
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < IMAGE_HEIGHT; i++) {
		for (int j = 0; j < IMAGE_WIDTH; j++) {
			// Grayscale conversion formula
			int gray = 0.299 * inputImage[i][j].red + 0.587 * inputImage[i][j].green + 0.114 *
			inputImage[i][j].blue;
			// Ensure the resulting value is within the valid range [0, 255]
			gray = (gray < 0) ? 0 : (gray > 255) ? 255 : gray;
			// Assign the grayscale value to the output pixel
			outputImage[i][j].red = gray;
			outputImage[i][j].green = gray;
			outputImage[i][j].blue = gray;
		}
	}
}

int main() {
 // Initialize image and blur kernel
	int **image = (int **)malloc(IMAGE_SIZE * sizeof(int *));
    int **blurred_image = (int **)malloc(IMAGE_SIZE * sizeof(int *));
    Pixel **inputImage = (Pixel **)malloc(IMAGE_HEIGHT * sizeof(Pixel *));
    Pixel **outputImage = (Pixel **)malloc(IMAGE_HEIGHT * sizeof(Pixel *));
    
    for (int i = 0; i < IMAGE_SIZE; ++i) {
        image[i] = (int *)malloc(IMAGE_SIZE * sizeof(int));
        blurred_image[i] = (int *)malloc(IMAGE_SIZE * sizeof(int));
    }

    for (int i = 0; i < IMAGE_HEIGHT; ++i) {
        inputImage[i] = (Pixel *)malloc(IMAGE_WIDTH * sizeof(Pixel));
        outputImage[i] = (Pixel *)malloc(IMAGE_WIDTH * sizeof(Pixel));
    }
    
	
	float blur_kernel[KERNEL_SIZE][KERNEL_SIZE] = {{1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0},
												   {1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0},
												   {1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0}};

	// Initialize the image with random values
	for (int i = 0; i < IMAGE_SIZE; ++i) {
		for (int j = 0; j < IMAGE_SIZE; ++j) {
			image[i][j] = rand() % 256; // or you can use any method to initialize
		}
	}
	for (int i = 0; i < IMAGE_HEIGHT; ++i) {
		for (int j = 0; j < IMAGE_WIDTH; ++j) {
			inputImage[i][j].red = rand() % 256;
			inputImage[i][j].green = rand() % 256;
			inputImage[i][j].blue = rand() % 256;
		}
	}
	
	blur_image(image, blur_kernel, blurred_image);
	grayscaleConversion(inputImage, outputImage);
	
	for (int i = 0; i < IMAGE_SIZE; ++i) {
        free(image[i]);
        free(blurred_image[i]);
    }

    for (int i = 0; i < IMAGE_HEIGHT; ++i) {
        free(inputImage[i]);
        free(outputImage[i]);
    }

    free(image);
    free(blurred_image);
    free(inputImage);
    free(outputImage);
    
	printf("Image blurring complete.\n");  
	printf("Grayscale conversion complete.\n");
	
	return 0;
}
