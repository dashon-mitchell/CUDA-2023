//nvcc SimpleJuliaSet.cu -o SimpleJuliaSet -lglut -lGL -lm
// This is a simple Julia set which is repeated iterations of 
// Znew = Zold + C whre Z and Care imaginary numbers.
// After so many tries if Zinitial escapes color it black if it stays around color it red.

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 1024
#define A  -0.824  //real
#define B  -0.1711   //imaginary

unsigned int window_width = 1024;
unsigned int window_height = 1024;

float xMin = -2.0;
float xMax =  2.0;
float yMin = -2.0;
float yMax =  2.0;

float stepSizeX = (xMax - xMin)/((float)window_width);
float stepSizeY = (yMax - yMin)/((float)window_height);

//Globals
float *A_CPU; //CPU pointers
float *A_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid

//This will be the layout of the parallel space we will be using.
void SetUpCudaDevices()
{
	BlockSize.x = 1024; 
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1024; 
	GridSize.y = 1;
	GridSize.z = 1;
}

//Sets a side memory on the GPU and CPU for our use.
void AllocateMemory()
{					
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,N*N*sizeof(float));

	//Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(N*N*sizeof(float));
}
void CleanUp()
{
	free(A_CPU); 
	cudaFree(A_GPU); 
}

void errorCheck(const char *file, int line)
{
	cudaError_t error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

__global__ void compute(float *a, float xmin,float ymin, float dx, float dy) 
{
	float mag,maxMag,temp;
	float maxCount = 200;
	float count = 0;
	maxMag = 10;
	mag = 0.0;
	int id = threadIdx.x+ blockDim.x*blockIdx.x;
	float x=xmin+ dx *threadIdx.x;
	float y=ymin+ dy *blockIdx.x;
	while (mag < maxMag && count < maxCount) 
	{
		// Zn = Zo*Zo + C
		// or xn + yni = (xo + yoi)*(xo + yoi) + A + Bi
		// xn = xo*xo - yo*yo + A (real Part) and yn = 2*xo*yo + B (imagenary part)
		temp = x; // We will be changing the x but weneed its old value to hind y.	
		x = x*x - y*y + A;
		y = (2.0 * temp * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < maxCount) 
	{
		a[id]=0.0;
	}
	else
	{
		a[id]=1.0;
	}
	
}



void display(void)
{
	float *pixels; 
	int k;
	pixels = (float *)malloc(window_width*window_height*3*sizeof(float));
	k=0;
	int i=0;
	while(k < 3*N*N) 
	{
		pixels[k] = A_CPU[i] ;	//Red on or off returned from color
		pixels[k+1] = 0.0; 	//Green off
		pixels[k+2] = 0.0;	//Blue off
		k=k+3;			//Skip to next pixel
		i++;
		printf("%f",A_CPU[i]);
	}
	

	glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
}


int main(int argc, char** argv)
{ 
	//Set the thread structure that you will be using on the GPU	
	SetUpCudaDevices();

	//Partitioning off the memory that you will be using.
	AllocateMemory();
	
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Fractals man, fractals.");
	
	//Copy Memory from CPU to GPU
	cudaMemcpyAsync(A_GPU, A_CPU, N*N*sizeof(float), cudaMemcpyHostToDevice);
	errorCheck(__FILE__, __LINE__);
	
	compute<<<GridSize,BlockSize>>>(A_GPU,xMin, yMin, stepSizeX, stepSizeY);
	
	//Copy Memory from GPU to CPU	
	cudaMemcpyAsync(A_CPU, A_GPU, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	errorCheck(__FILE__, __LINE__);
	
   	glutDisplayFunc(display);
   	glutMainLoop();
   	
	CleanUp();
}
