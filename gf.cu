#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <iostream>

using namespace std;

#define BLOCKDIM_X		16
#define BLOCKDIM_Y		16

#define GRIDDIM_X		256
#define GRIDDIM_Y		256
#define MASK_WIDTH		5

__constant__ int d_const_Gaussian[MASK_WIDTH*MASK_WIDTH]; //常量

static __global__ void kernel_GaussianFilt(int width, int height, int byte_per_pixel, unsigned char *d_src_imgbuf, unsigned char *d_guassian_imgbuf);
int parseInt(int , char* );
int read(FILE*, int, int);
int** parse_bmp(const char* filepath, int* width, int* height);
void write_buffer(int value, FILE* file, int length);
void write_file(const char* filepath, int width, int height, int** data);
unsigned char * transformToUCharVector(int ** data, int width, int height, int byte_per_pixel);
int ** transformToIntMatrix(unsigned char * data, int width, int height, int byte_per_pixel);

unsigned long GetTickCount()
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);

}


int main(int argc, char **argv)
{
	char* input_path;
	char* output_path;
	if (argc != 3) {
		input_path = "/home/cloud/workspace/hand/test1.bmp";
		output_path = "/home/cloud/workspace/hand/result.bmp";
	} else {
		input_path = argv[1];
		output_path = argv[2];
	}

	// char * input_path = "C:\\Users\\congj\\Desktop\\result.bmp";
	// char * output_path = "C:\\Users\\congj\\Desktop\\gs.bmp";
	printf("input_path: %s\n", input_path);
	printf("output_path: %s\n", output_path);
	
	struct cudaDeviceProp pror;
	cudaGetDeviceProperties(&pror, 0);
	cout << "maxThreadsPerBlock=" << pror.maxThreadsPerBlock << endl;

	long start, end;
	long time = 0;

	start = GetTickCount();
	cudaEvent_t startt, stop; 
	cudaEventCreate(&startt);
	cudaEventCreate(&stop);
	cudaEventRecord(startt, 0);

	unsigned char *h_src_imgbuf;
	int width, height, byte_per_pixel = 3;

	int **d = parse_bmp(input_path, &width, &height);
	h_src_imgbuf = transformToUCharVector(d, width, height, byte_per_pixel);

	printf("width: %d, height: %d, byte_per_pixel: %d\n", width, height, byte_per_pixel);

	int size1 = width * height *byte_per_pixel * sizeof(unsigned char);

	//host memory
	unsigned char *h_guassian_imgbuf = new unsigned char[width*height*byte_per_pixel];

	//device memory
	unsigned char *d_src_imgbuf;
	unsigned char *d_guassian_imgbuf;

	cudaMalloc((void**)&d_src_imgbuf, size1);
	cudaMalloc((void**)&d_guassian_imgbuf, size1);

	//copy data from host to device
	cudaMemcpy(d_src_imgbuf, h_src_imgbuf, size1, cudaMemcpyHostToDevice);

	//gaussian matrix constant memory
	int Gaussian[25] = { 1,4,7,4,1,
						4,16,26,16,4,
						7,26,41,26,7,
						4,16,26,16,4,
						1,4,7,4,1 };//sum is 273
	cudaMemcpyToSymbol(d_const_Gaussian, Gaussian, 25 * sizeof(int));

	int bx = ceil((double)width / BLOCKDIM_X); // 40
	int by = ceil((double)height / BLOCKDIM_Y); //26

	if (bx > GRIDDIM_X) bx = GRIDDIM_X;
	if (by > GRIDDIM_Y) by = GRIDDIM_Y;
	//suppose width=638, height=411

	dim3 grid(bx, by); //40,26
	dim3 block(BLOCKDIM_X, BLOCKDIM_Y); //16,16

	//kernel
	kernel_GaussianFilt<<<grid, block>>>(width, height, byte_per_pixel, d_src_imgbuf, d_guassian_imgbuf);
	cudaMemcpy(h_guassian_imgbuf, d_guassian_imgbuf, size1, cudaMemcpyDeviceToHost);

	// saveBmp(output_path, h_guassian_imgbuf, width, height, byte_per_pixel);
	write_file(output_path, width, height, transformToIntMatrix(h_guassian_imgbuf, width, height, byte_per_pixel));
	//
	cudaFree(d_src_imgbuf);
	cudaFree(d_guassian_imgbuf);

	delete[]h_src_imgbuf;
	delete[]h_guassian_imgbuf;

	end = GetTickCount();
	//InterlockedExchangeAdd(&time, end - start); //window api
	__sync_fetch_and_add(&time, end - start); // linux api
	cout << "Total time GPU:";
	cout << time << endl;

	return 0;
}

static __global__ void kernel_GaussianFilt(int width, int height, int byte_per_pixel, unsigned char *d_src_imgbuf, unsigned char *d_dst_imgbuf)
{
	const int tix = blockDim.x * blockIdx.x + threadIdx.x;
	const int tiy = blockDim.y * blockIdx.y + threadIdx.y;
	/*cout << threadIdx.x << endl;
	cout << threadIdx.y << endl;*/
	const int threadTotalX = blockDim.x * gridDim.x;
	const int threadTotalY = blockDim.y * gridDim.y;

	for (int ix = tix; ix < height; ix += threadTotalX)
		for (int iy = tiy; iy < width; iy += threadTotalY)
		{
			for (int k = 0; k < byte_per_pixel; k++)
			{
				int sum = 0;
				int tempPixelValue = 0;
				for (int m = -2; m <= 2; m++)
				{
					for (int n = -2; n <= 2; n++)
					{
						if (ix + m < 0 || iy + n < 0 || ix + m >= height || iy + n >= width)
							tempPixelValue = 0;
						else
							tempPixelValue = *(d_src_imgbuf + (ix + m)*width*byte_per_pixel + (iy + n)*byte_per_pixel + k);
						sum += tempPixelValue * d_const_Gaussian[(m + 2) * 5 + n + 2];
					}
				}

				if (sum / 273 < 0)
					*(d_dst_imgbuf + (ix)*width*byte_per_pixel + (iy)*byte_per_pixel + k) = 0;
				else if (sum / 273 > 255)
					*(d_dst_imgbuf + (ix)*width*byte_per_pixel + (iy)*byte_per_pixel + k) = 255;
				else
					*(d_dst_imgbuf + (ix)*width*byte_per_pixel + (iy)*byte_per_pixel + k) = sum / 273;
			}
		}
}

int parseInt(int length, char* s)
{
	int result = 0;
	int shift = 0;
	for (int i = 0; i < length; i++)
	{
		//cout << hex << (int)(s[i] & 0x000000ff) << endl;
		result += (s[i]& 0x000000ff) << shift;
		shift += 8;
	}
	return result;
}

int read(FILE* file, int offset, int length)
{
	static char buff[4];
	fseek(file, offset, 0);
	fread(buff, sizeof(char), length, file);
	//current = offset + length;

	//cout << "current: " << ftell(file) << endl;
	return parseInt(length, buff);
}

int** parse_bmp(const char* filepath, int* width, int* height)
{
	FILE * file = fopen(filepath, "rb");
	if (!file) 
	{
		cerr << "文件打开失败。" << endl;
		exit(-1);
	}
	fseek(file, 0x0A, 0);

	int content_offset = read(file, 0x0A, 4);
	*width = read(file, 0x12, 4);
	*height = read(file, 0x16, 4);

	int** result;
	result = (int**)malloc(sizeof(int*) * 3);
	for (int i = 0; i < 3; i++)
	{
		result[i] = (int*)malloc(sizeof(int) * (*width) * (*height));
	}

	fseek(file, content_offset, 0);

	int pixel_acount = (*width) * (*height);

	int byte_in_row = *width * 24/8;
	int actual_byte_in_row = byte_in_row + 4 - byte_in_row % 4;

	cout << byte_in_row << endl;
	cout << actual_byte_in_row << endl;

	char* buffer;
	buffer = (char *)malloc(sizeof(char) * actual_byte_in_row);
	for (int i = 0; i < *height; i++) {
		fread(buffer, sizeof(char), actual_byte_in_row, file);
		for (int j = 0; j < *width; j++) {
			result[0][i* *width + j] = buffer[3 * j] & 0x000000ff;
			result[1][i* *width + j] = buffer[3 * j + 1] & 0x000000ff;
			result[2][i* *width + j] = buffer[3 * j + 2] & 0x000000ff;
		}
	}
	

	fclose(file);
	return result;
}

void write_buffer(int value, FILE* file, int length)
{
	static char buffer[4];
	for (int i = 0; i < length; i++)
	{
		char v_low8 = value & 0x000000ff;
		//cout << hex << int(v_low8) << endl;
		value = value >> 8;
		buffer[i] = v_low8;
	}
	for (int i = length - 1; i >= 0; i--)
		fwrite(buffer+length-1-i, sizeof(char), 1, file);
}

void write_file(const char* filepath, int width, int height, int** data)
{
	FILE *file = fopen(filepath, "wb");
	if (!file)
	{
		cerr << "文件打开错误" << endl;
		exit(-1);
	}
	char buffer[4];
	buffer[0] = 0x42;
	buffer[1] = 0x4D;
	fwrite(buffer, sizeof(char), 2, file); //写入BM

	int byte_in_row = width * 24 / 8;
	int actual_byte_in_row = byte_in_row + 4 - byte_in_row % 4;

	int size = 54 + actual_byte_in_row*height;
	write_buffer(size, file, 4); //写入文件大小的字节数
	write_buffer(0, file, 2); //写入保留字节 2个字节
	write_buffer(0, file, 2); //写入保留字节 2个字节
	write_buffer(54, file, 4); //写入偏移量，4个字节
	write_buffer(40, file, 4); //写入头部长度
	write_buffer(width, file, 4); //写入宽度
	write_buffer(height, file, 4); //写入高度
	write_buffer(1, file, 2); //平面数，总是被设置为1
	write_buffer(24, file, 2); //每像素位数
	write_buffer(0, file, 4); //不压缩
	write_buffer(height*actual_byte_in_row, file, 4); //图像字节数
	write_buffer(0, file, 4); //图像字节数
	write_buffer(0, file, 4); //图像字节数
	write_buffer(0, file, 4); //图像字节数
	write_buffer(0, file, 4); //图像字节数

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
		{
			write_buffer(data[0][i*width + j], file, 1);
			write_buffer(data[1][i*width + j], file, 1);
			write_buffer(data[2][i*width + j], file, 1);
		}
		if (actual_byte_in_row - byte_in_row != 0)
			write_buffer(0, file, actual_byte_in_row - byte_in_row);
		cout << i << endl;
	}
	

	fclose(file);
}

unsigned char * transformToUCharVector(int ** data, int width, int height, int byte_per_pixel)
{
	unsigned char * result;
	result = (unsigned char *)malloc(sizeof(unsigned char) * width * height * byte_per_pixel);
	for (int i=0; i<width*height; i++)
	{
		for (int j=0; j<byte_per_pixel; j++)
		{
			result[i*byte_per_pixel + j] = (unsigned char)(data[j][i] & 0x000000ff);
		}
	}
	return result;
}

int ** transformToIntMatrix(unsigned char * data, int width, int height, int byte_per_pixel)
{
	int ** result;
	result = (int**)malloc(sizeof(int *)*byte_per_pixel);
	
	for (int i=0; i<byte_per_pixel; i++)
		result[i] = (int *)malloc(sizeof(int)*width*height);
	
	for (int i=0; i<width*height; i++)
		for (int j=0; j<byte_per_pixel; j++) 
			result[j][i] = data[i*byte_per_pixel+j] & 0x000000ff;
	
	return result;
}
