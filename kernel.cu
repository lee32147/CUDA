#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <omp.h>

#define Zad1


#ifdef Zad1
int coresPerSM(cudaDeviceProp prop) {
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{0x30, 192},
		{0x32, 192},
		{0x35, 192},
		{0x37, 192},
		{0x50, 128},
		{0x52, 128},
		{0x53, 128},
		{0x60,  64},
		{0x61, 128},
		{0x62, 128},
		{0x70,  64},
		{0x72,  64},
		{0x75,  64},
		{-1, -1} };

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((prop.major << 4) + prop.minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
		"MapSMtoCores for SM %d.%d is undefined."
		"  Default to use %d Cores/SM\n",
		prop.major, prop.minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

int main()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Nazwa urzadzenia: %s\n", prop.name);
	printf("Czestotliwosc zegara [KHz]: %d\n", prop.memoryClockRate);
	printf("Przepustowosc pamieci [bity]: %d\n", prop.memoryBusWidth);
	printf("Compute Capability: %d\n", coresPerSM(prop));
	printf("Liczba multiprocesorow: %d\n", prop.multiProcessorCount);
	printf("Liczba rdzeni: %d\n", (coresPerSM(prop)) * prop.multiProcessorCount);

	cudaSetDevice(0);
	cudaDeviceReset();
    return 0;
}
#endif

#ifdef Zad2

int main()
{
	cudaSetDevice(0);

	char *charmib1, *charmib8, *charmib96, *charmib256, *a1, *a8, *a96, *a256;
	int *intmib1, *intmib8, *intmib96, *intmib256, *b1, *b8, *b96, *b256;
	float *floatmib1, *floatmib8, *floatmib96, *floatmib256, *c1, *c8, *c96, *c256;
	double *doublemib1, *doublemib8, *doublemib96, *doublemib256, *d1, *d8, *d96, *d256;
	charmib1 = new char[1024 * 1024];
	charmib8 = new char[8 * 1024 * 1024];
	charmib96 = new char[96 * 1024 * 1024];
	charmib256 = new char[256 * 1024 * 1024];
	intmib1 = new int[1024 * 1024 / 4];
	intmib8 = new int[2 * 1024 * 1024];
	intmib96 = new int[24 * 1024 * 1024];
	intmib256 = new int[64 * 1024 * 1024];
	floatmib1 = new float[1024 * 1024 / 4];
	floatmib8 = new float[2 * 1024 * 1024];
	floatmib96 = new float[24 * 1024 * 1024];
	floatmib256 = new float[64 * 1024 * 1024];
	doublemib1 = new double[1024 * 1024 / 8];
	doublemib8 = new double[1024 * 1024];
	doublemib96 = new double[12 * 1024 * 1024];
	doublemib256 = new double[32 * 1024 * 1024];

	cudaMalloc(&a1, 1024 * 1024 * sizeof(char));
	cudaMalloc(&a8, 1024 * 1024 * 8 * sizeof(char));
	cudaMalloc(&a96, 1024 * 1024 * 96 * sizeof(char));
	cudaMalloc(&a256, 1024 * 1024 * 256 * sizeof(char));
	cudaMalloc(&b1, 1024 * 1024 * sizeof(int) / 4);
	cudaMalloc(&b8, 1024 * 1024 * 2 * sizeof(int));
	cudaMalloc(&b96, 1024 * 1024 * 24 * sizeof(int));
	cudaMalloc(&b256, 1024 * 1024 * 64 * sizeof(int));
	cudaMalloc(&c1, 1024 * 1024 * sizeof(float) / 4);
	cudaMalloc(&c8, 1024 * 1024 * 2 * sizeof(float));
	cudaMalloc(&c96, 1024 * 1024 * 24 * sizeof(float));
	cudaMalloc(&c256, 1024 * 1024 * 64 * sizeof(float));
	cudaMalloc(&d1, 1024 * 1024 * sizeof(double) / 8);
	cudaMalloc(&d8, 1024 * 1024 * sizeof(double));
	cudaMalloc(&d96, 1024 * 1024 * 12 * sizeof(double));
	cudaMalloc(&d256, 1024 * 1024 * 32 * sizeof(double));
	
	cudaEvent_t start, stop;
	float czas;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaMemcpy(a1, charmib1, 1024 * 1024 * sizeof(char), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (char, 1 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(a8, charmib8, 1024 * 1024 * 8 *sizeof(char), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (char, 8 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(a96, charmib96, 1024 * 1024 * 96 *sizeof(char), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (char, 96 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(a256, charmib256, 1024 * 1024 * 256 * sizeof(char), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (char, 256 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(b1, intmib1, 1024 * 1024 * sizeof(int) / 4, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (int, 1 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(b8, intmib8, 1024 * 1024 * 2 * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (int, 8 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(b96, intmib96, 1024 * 1024 * 24 * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (int, 96 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(b256, intmib256, 1024 * 1024 * 64 * sizeof(int), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (int, 256 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(c1, floatmib1, 1024 * 1024 * sizeof(float) / 4, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (float, 1 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(c8, floatmib8, 1024 * 1024 * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (float, 8 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(c96, floatmib96, 1024 * 1024 * 24 * sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (float, 96 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(c256, floatmib256, 1024 * 1024 * 64 * sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (float, 256 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(d1, doublemib1, 1024 * 1024 * sizeof(double) / 8, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (double, 1 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(d8, doublemib8, 1024 * 1024 * sizeof(double), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (double, 8 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(d96, doublemib96, 1024 * 1024 * 12 * sizeof(double), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (double, 96 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(d256, doublemib256, 1024 * 1024 * 32 * sizeof(double), cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania HostToDevice (double, 256 MiB) [ms]: %f\n\n", czas);

	cudaEventRecord(start, 0);
	cudaMemcpy(charmib1, a1, 1024 * 1024 * sizeof(char), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (char, 1 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(charmib8, a8, 1024 * 1024 * 8 * sizeof(char), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (char, 8 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(charmib96, a96, 1024 * 1024 * 64 * sizeof(char), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (char, 96 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(charmib256, a256, 1024 * 1024 * 256 * sizeof(char), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (char, 256 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(intmib1, b1, 1024 * 1024 * sizeof(int) / 4, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (int, 1 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(intmib8, b8, 1024 * 1024 * 2 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (int, 8 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(intmib96, b96, 1024 * 1024 * 24 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (int, 96 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(intmib256, b256, 1024 * 1024 * 64 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (int, 256 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(floatmib1, c1, 1024 * 1024 * sizeof(float) / 4, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (float, 1 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(floatmib8, c8, 1024 * 1024 * 2 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (float, 8 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(floatmib96, c96, 1024 * 1024 * 24 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (float, 96 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(floatmib256, c256, 1024 * 1024 * 64 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (float, 256 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(doublemib1, d1, 1024 * 1024 * sizeof(double) / 8, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (double, 1 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(doublemib8, d8, 1024 * 1024 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (double, 8 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(doublemib96, d96, 1024 * 1024 * 12 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (double, 96 MiB) [ms]: %f\n", czas);
	cudaEventRecord(start, 0);
	cudaMemcpy(doublemib256, d256, 1024 * 1024 * 32 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&czas, start, stop);
	printf("Czas przesylania DeviceToHost (double, 256 MiB) [ms]: %f\n", czas);

	delete[] charmib1;
	delete[] charmib8;
	delete[] charmib96;
	delete[] charmib256;
	delete[] intmib1;
	delete[] intmib8;
	delete[] intmib96;
	delete[] intmib256;
	delete[] floatmib1;
	delete[] floatmib8;
	delete[] floatmib96;
	delete[] floatmib256;
	delete[] doublemib1;
	delete[] doublemib8;
	delete[] doublemib96;
	delete[] doublemib256;

	cudaFree(a1);
	cudaFree(a8);
	cudaFree(a96);
	cudaFree(a256);
	cudaFree(b1);
	cudaFree(b8);
	cudaFree(b96);
	cudaFree(b256);
	cudaFree(c1);
	cudaFree(c8);
	cudaFree(c96);
	cudaFree(c256);
	cudaFree(d1);
	cudaFree(d8);
	cudaFree(d96);
	cudaFree(d256);
	cudaDeviceReset();
	return 0;
}
#endif

#ifdef Zad3

__global__ void kernelMnozenie(int *a, int *b, int *c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	c[i] = a[i] * b[i];
}

__global__ void kernelDodawanie(int *a, int *b, int *c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void kernelPotegowanie(int *a, int *b, int *c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int wynik = 1;
	for (int j = 0; j < b[i]; j++)
	{
		wynik *= a[i];
	}
	c[i] = wynik;
}

__global__ void kernelMnozenie(float *a, float *b, float *c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	c[i] = a[i] * b[i];
}

__global__ void kernelDodawanie(float *a, float *b, float *c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void kernelPotegowanie(float *a, float *b, float *c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	float wynik = 1;
	for (float j = 0; j < b[i]; j++)
	{
		wynik *= a[i];
	}
	c[i] = wynik;
}

__global__ void kernelMnozenie(double *a, double *b, double *c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	c[i] = a[i] * b[i];
}

__global__ void kernelDodawanie(double *a, double *b, double *c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void kernelPotegowanie(double *a, double *b, double *c)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	double wynik = 1;
	for (double j = 0; j < b[i]; j++)
	{
		wynik *= a[i];
	}
	c[i] = wynik;
}

void dodawanieCPU(int *a, int *b, int *c, int rozmiar);
void mnozenieCPU(int *a, int *b, int *c, int rozmiar);
void potegowanieCPU(int *a, int *b, int *c, int rozmiar);
void dodawanieCPU(float *a, float *b, float *c, int rozmiar);
void mnozenieCPU(float *a, float *b, float *c, int rozmiar);
void potegowanieCPU(float *a, float *b, float *c, int rozmiar);
void dodawanieCPU(double *a, double *b, double *c, int rozmiar);
void mnozenieCPU(double *a, double *b, double *c, int rozmiar);
void potegowanieCPU(double *a, double *b, double *c, int rozmiar);

int main()
{
	cudaSetDevice(0);
	
	/*const int rozmiar = 9;
	int a[rozmiar] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int b[rozmiar] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int c[rozmiar] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int *dev_a;
	int *dev_b;
	int *dev_c;
	
	cudaMalloc(&dev_a, rozmiar * sizeof(int));
	cudaMalloc(&dev_b, rozmiar * sizeof(int));
	cudaMalloc(&dev_c, rozmiar * sizeof(int));
	cudaMemcpy(dev_a, a, rozmiar * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, rozmiar * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, rozmiar * sizeof(int), cudaMemcpyHostToDevice);
	kernelDodawanie << <1, rozmiar >> > (dev_a, dev_b, dev_c);
	cudaDeviceSynchronize();
	cudaMemcpy(c, dev_c, rozmiar * sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nDodawanie GPU\n");
	for (int i = 0; i < rozmiar; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	kernelMnozenie << <1, rozmiar >> > (dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, rozmiar * sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nMnozenie GPU\n");
	for (int i = 0; i < rozmiar; i++)
	{
		printf("%d * %d = %d\n", a[i], b[i], c[i]);
	}
	kernelPotegowanie << <1, rozmiar >> > (dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, rozmiar * sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nPotegowanie GPU\n");
	for (int i = 0; i < rozmiar; i++)
	{
		printf("%d ^ %d = %d\n", a[i], b[i], c[i]);
	}
	dodawanieCPU(a, b, c, rozmiar);
	printf("\nDodawanie CPU\n");
	for (int i = 0; i < rozmiar; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	mnozenieCPU(a, b, c, rozmiar);
	printf("\nMnozenie CPU\n");
	for (int i = 0; i < rozmiar; i++)
	{
		printf("%d * %d = %d\n", a[i], b[i], c[i]);
	}
	potegowanieCPU(a, b, c, rozmiar);
	printf("\nPotegowanie CPU\n");
	for (int i = 0; i < rozmiar; i++)
	{
		printf("%d ^ %d = %d\n", a[i], b[i], c[i]);
	}
	*/

	const int rozmiar = 1024 * 1024;
	int liczbaBlokow;
	int rozmiarBloku = 1024;
	int *aint1 = new int[rozmiar / 4];
	int *bint1 = new int[rozmiar / 4];
	int *cint1 = new int[rozmiar / 4];
	int *aint4 = new int[rozmiar];
	int *bint4 = new int[rozmiar];
	int *cint4 = new int[rozmiar];
	int *aint8 = new int[rozmiar * 2];
	int *bint8 = new int[rozmiar * 2];
	int *cint8 = new int[rozmiar * 2];
	int *aint16 = new int[rozmiar * 4];
	int *bint16 = new int[rozmiar * 4];
	int *cint16 = new int[rozmiar * 4];
	float *afloat1 = new float[rozmiar / 4];
	float *bfloat1 = new float[rozmiar / 4];
	float *cfloat1 = new float[rozmiar / 4];
	float *afloat4 = new float[rozmiar];
	float *bfloat4 = new float[rozmiar];
	float *cfloat4 = new float[rozmiar];
	float *afloat8 = new float[rozmiar * 2];
	float *bfloat8 = new float[rozmiar * 2];
	float *cfloat8 = new float[rozmiar * 2];
	float *afloat16 = new float[rozmiar * 4];
	float *bfloat16 = new float[rozmiar * 4];
	float *cfloat16 = new float[rozmiar * 4];
	double *adouble1 = new double[rozmiar / 8];
	double *bdouble1 = new double[rozmiar / 8];
	double *cdouble1 = new double[rozmiar / 8];
	double *adouble4 = new double[rozmiar / 2];
	double *bdouble4 = new double[rozmiar / 2];
	double *cdouble4 = new double[rozmiar / 2];
	double *adouble8 = new double[rozmiar];
	double *bdouble8 = new double[rozmiar];
	double *cdouble8 = new double[rozmiar];
	double *adouble16 = new double[rozmiar * 2];
	double *bdouble16 = new double[rozmiar * 2];
	double *cdouble16 = new double[rozmiar * 2];
	int *dev_aint1;
	int *dev_aint4;
	int *dev_aint8;
	int *dev_aint16;
	int *dev_bint1;
	int *dev_bint4;
	int *dev_bint8;
	int *dev_bint16;
	int *dev_cint1;
	int *dev_cint4;
	int *dev_cint8;
	int *dev_cint16;
	float *dev_afloat1;
	float *dev_afloat4;
	float *dev_afloat8;
	float *dev_afloat16;
	float *dev_bfloat1;
	float *dev_bfloat4;
	float *dev_bfloat8;
	float *dev_bfloat16;
	float *dev_cfloat1;
	float *dev_cfloat4;
	float *dev_cfloat8;
	float *dev_cfloat16;
	double *dev_adouble1;
	double *dev_adouble4;
	double *dev_adouble8;
	double *dev_adouble16;
	double *dev_bdouble1;
	double *dev_bdouble4;
	double *dev_bdouble8;
	double *dev_bdouble16;
	double *dev_cdouble1;
	double *dev_cdouble4;
	double *dev_cdouble8;
	double *dev_cdouble16;
	
	cudaMalloc(&dev_aint1, rozmiar * sizeof(int) / 4);
	cudaMalloc(&dev_aint4, rozmiar * sizeof(int));
	cudaMalloc(&dev_aint8, rozmiar * sizeof(int) * 2);
	cudaMalloc(&dev_aint16, rozmiar * sizeof(int) * 4);
	cudaMalloc(&dev_bint1, rozmiar * sizeof(int) / 4);
	cudaMalloc(&dev_bint4, rozmiar * sizeof(int));
	cudaMalloc(&dev_bint8, rozmiar * sizeof(int) * 2);
	cudaMalloc(&dev_bint16, rozmiar * sizeof(int) * 4);
	cudaMalloc(&dev_cint1, rozmiar * sizeof(int)) / 4;
	cudaMalloc(&dev_cint4, rozmiar * sizeof(int));
	cudaMalloc(&dev_cint8, rozmiar * sizeof(int) * 2);
	cudaMalloc(&dev_cint16, rozmiar * sizeof(int) * 4);
	cudaMalloc(&dev_afloat1, rozmiar * sizeof(float) / 4);
	cudaMalloc(&dev_afloat4, rozmiar * sizeof(float));
	cudaMalloc(&dev_afloat8, rozmiar * sizeof(float) * 2);
	cudaMalloc(&dev_afloat16, rozmiar * sizeof(float) * 4);
	cudaMalloc(&dev_bfloat1, rozmiar * sizeof(float) / 4);
	cudaMalloc(&dev_bfloat4, rozmiar * sizeof(float));
	cudaMalloc(&dev_bfloat8, rozmiar * sizeof(float) * 2);
	cudaMalloc(&dev_bfloat16, rozmiar * sizeof(float) * 4);
	cudaMalloc(&dev_cfloat1, rozmiar * sizeof(float) / 4);
	cudaMalloc(&dev_cfloat4, rozmiar * sizeof(float));
	cudaMalloc(&dev_cfloat8, rozmiar * sizeof(float) * 2);
	cudaMalloc(&dev_cfloat16, rozmiar * sizeof(float) * 4);
	cudaMalloc(&dev_adouble1, rozmiar * sizeof(double) / 8);
	cudaMalloc(&dev_adouble4, rozmiar * sizeof(double) / 2);
	cudaMalloc(&dev_adouble8, rozmiar * sizeof(double));
	cudaMalloc(&dev_adouble16, rozmiar * sizeof(double) * 2);
	cudaMalloc(&dev_bdouble1, rozmiar * sizeof(double) / 8);
	cudaMalloc(&dev_bdouble4, rozmiar * sizeof(double) / 2);
	cudaMalloc(&dev_bdouble8, rozmiar * sizeof(double));
	cudaMalloc(&dev_bdouble16, rozmiar * sizeof(double) * 2);
	cudaMalloc(&dev_cdouble1, rozmiar * sizeof(double) / 8);
	cudaMalloc(&dev_cdouble4, rozmiar * sizeof(double) / 2);
	cudaMalloc(&dev_cdouble8, rozmiar * sizeof(double));
	cudaMalloc(&dev_cdouble16, rozmiar * sizeof(double) * 2);

	cudaMemcpy(dev_aint1, aint1, rozmiar * sizeof(int) / 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aint4, aint4, rozmiar * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aint8, aint8, rozmiar * sizeof(int) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_aint16, aint16, rozmiar * sizeof(int) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bint1, bint1, rozmiar * sizeof(int) / 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bint4, bint4, rozmiar * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bint8, bint8, rozmiar * sizeof(int) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bint16, bint16, rozmiar * sizeof(int) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cint1, cint1, rozmiar * sizeof(int) / 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cint4, cint4, rozmiar * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cint8, cint8, rozmiar * sizeof(int) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cint16, cint16, rozmiar * sizeof(int) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_afloat1, afloat1, rozmiar * sizeof(float) / 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_afloat4, afloat4, rozmiar * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_afloat8, afloat8, rozmiar * sizeof(float) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_afloat16, afloat16, rozmiar * sizeof(float) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bfloat1, bfloat1, rozmiar * sizeof(float) / 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bfloat4, bfloat4, rozmiar * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bfloat8, bfloat8, rozmiar * sizeof(float) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bfloat16, bfloat16, rozmiar * sizeof(float) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cfloat1, cfloat1, rozmiar * sizeof(float) / 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cfloat4, cfloat4, rozmiar * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cfloat8, cfloat8, rozmiar * sizeof(float) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cfloat16, cfloat16, rozmiar * sizeof(float) * 4, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_adouble1, adouble1, rozmiar * sizeof(double) / 8, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_adouble4, adouble4, rozmiar * sizeof(double) / 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_adouble8, adouble8, rozmiar * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_adouble16, adouble16, rozmiar * sizeof(double) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bdouble1, bdouble1, rozmiar * sizeof(double) / 8, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bdouble4, bdouble4, rozmiar * sizeof(double) / 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bdouble8, bdouble8, rozmiar * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_bdouble16, bdouble16, rozmiar * sizeof(double) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cdouble1, cdouble1, rozmiar * sizeof(double) / 8, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cdouble4, cdouble4, rozmiar * sizeof(double) / 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cdouble8, cdouble8, rozmiar * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cdouble16, cdouble16, rozmiar * sizeof(double) * 2, cudaMemcpyHostToDevice);

	float czasGPU;
	cudaEvent_t startGPU, stopGPU;
	double startCPU, stopCPU;

	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_aint1, dev_bint1, dev_cint1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint1, dev_cint1, rozmiar * sizeof(int) / 4, cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (int, 1MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_aint4, dev_bint4, dev_cint4);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint4, dev_cint4, rozmiar * sizeof(int), cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (int, 4MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_aint8, dev_bint8, dev_cint8);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint8, dev_cint8, rozmiar * sizeof(int) * 2, cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (int, 8MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_aint16, dev_bint16, dev_cint16);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint16, dev_cint16, rozmiar * sizeof(int) * 4, cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (int, 16MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat1, dev_bfloat1, dev_cfloat1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat1, dev_cfloat1, rozmiar * sizeof(float) / 4, cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (float, 1MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat4, dev_bfloat4, dev_cfloat4);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat4, dev_cfloat4, rozmiar * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (float, 4MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat8, dev_bfloat8, dev_cfloat8);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat8, dev_cfloat8, rozmiar * sizeof(float) * 2, cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (float, 8MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat16, dev_bfloat16, dev_cfloat16);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat16, dev_cfloat16, rozmiar * sizeof(float) * 4, cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (float, 16MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 8 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble1, dev_bdouble1, dev_cdouble1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble1, dev_cdouble1, rozmiar * sizeof(double) / 8, cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (double, 1MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble4, dev_bdouble4, dev_cdouble4);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble4, dev_cdouble4, rozmiar * sizeof(double) / 2, cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (double, 4MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble8, dev_bdouble8, dev_cdouble8);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble8, dev_cdouble8, rozmiar * sizeof(double), cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (double, 8MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelDodawanie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble16, dev_bdouble16, dev_cdouble16);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble16, dev_cdouble16, rozmiar * sizeof(double) * 2, cudaMemcpyDeviceToHost);
	printf("Czas dodawania GPU (double, 16MiB) [ms]: %f\n\n", czasGPU);

	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_aint1, dev_bint1, dev_cint1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint1, dev_cint1, rozmiar * sizeof(int) / 4, cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (int, 1MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_aint4, dev_bint4, dev_cint4);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint4, dev_cint4, rozmiar * sizeof(int), cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (int, 4MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_aint8, dev_bint8, dev_cint8);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint8, dev_cint8, rozmiar * sizeof(int) * 2, cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (int, 8MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_aint16, dev_bint16, dev_cint16);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint16, dev_cint16, rozmiar * sizeof(int) * 4, cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (int, 16MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat1, dev_bfloat1, dev_cfloat1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat1, dev_cfloat1, rozmiar * sizeof(float) / 4, cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (float, 1MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat4, dev_bfloat4, dev_cfloat4);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat4, dev_cfloat4, rozmiar * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (float, 4MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat8, dev_bfloat8, dev_cfloat8);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat8, dev_cfloat8, rozmiar * sizeof(float) * 2, cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (float, 8MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat16, dev_bfloat16, dev_cfloat16);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat16, dev_cfloat16, rozmiar * sizeof(float) * 4, cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (float, 16MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 8 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble1, dev_bdouble1, dev_cdouble1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble1, dev_cdouble1, rozmiar * sizeof(double) / 8, cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (double, 1MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble4, dev_bdouble4, dev_cdouble4);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble4, dev_cdouble4, rozmiar * sizeof(double) / 2, cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (double, 4MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble8, dev_bdouble8, dev_cdouble8);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble8, dev_cdouble8, rozmiar * sizeof(double), cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (double, 8MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelMnozenie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble16, dev_bdouble16, dev_cdouble16);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble16, dev_cdouble16, rozmiar * sizeof(double) * 2, cudaMemcpyDeviceToHost);
	printf("Czas mnozenia GPU (double, 16MiB) [ms]: %f\n\n", czasGPU);

	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_aint1, dev_bint1, dev_cint1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint1, dev_cint1, rozmiar * sizeof(int) / 4, cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (int, 1MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_aint4, dev_bint4, dev_cint4);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint4, dev_cint4, rozmiar * sizeof(int), cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (int, 4MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_aint8, dev_bint8, dev_cint8);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint8, dev_cint8, rozmiar * sizeof(int) * 2, cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (int, 8MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_aint16, dev_bint16, dev_cint16);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cint16, dev_cint16, rozmiar * sizeof(int) * 4, cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (int, 16MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat1, dev_bfloat1, dev_cfloat1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat1, dev_cfloat1, rozmiar * sizeof(float) / 4, cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (float, 1MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat4, dev_bfloat4, dev_cfloat4);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat4, dev_cfloat4, rozmiar * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (float, 4MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat8, dev_bfloat8, dev_cfloat8);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat8, dev_cfloat8, rozmiar * sizeof(float) * 2, cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (float, 8MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 4 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_afloat16, dev_bfloat16, dev_cfloat16);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cfloat16, dev_cfloat16, rozmiar * sizeof(float) * 4, cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (float, 16MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 8 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble1, dev_bdouble1, dev_cdouble1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble1, dev_cdouble1, rozmiar * sizeof(double) / 8, cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (double, 1MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar / 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble4, dev_bdouble4, dev_cdouble4);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble4, dev_cdouble4, rozmiar * sizeof(double) / 2, cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (double, 4MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble8, dev_bdouble8, dev_cdouble8);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble8, dev_cdouble8, rozmiar * sizeof(double), cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (double, 8MiB) [ms]: %f\n", czasGPU);
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	liczbaBlokow = (rozmiar * 2 - rozmiarBloku + 1) / rozmiarBloku;
	cudaEventRecord(startGPU, 0);
	kernelPotegowanie << <liczbaBlokow, rozmiarBloku >> > (dev_adouble16, dev_bdouble16, dev_cdouble16);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	cudaMemcpy(cdouble16, dev_cdouble16, rozmiar * sizeof(double) * 2, cudaMemcpyDeviceToHost);
	printf("Czas potegowanie GPU (double, 16MiB) [ms]: %f\n\n", czasGPU);
 
	startCPU = omp_get_wtime();
	dodawanieCPU(aint1, bint1, cint1, rozmiar / 4);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (int, 1MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	dodawanieCPU(aint4, bint4, cint4, rozmiar);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (int, 4MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	dodawanieCPU(aint8, bint8, cint8, rozmiar * 2);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (int, 8MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	dodawanieCPU(aint16, bint16, cint16, rozmiar * 4);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (int, 16MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	dodawanieCPU(afloat1, bfloat1, cfloat1, rozmiar / 4);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (float, 1MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	dodawanieCPU(afloat4, bfloat4, cfloat4, rozmiar);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (float, 4MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	dodawanieCPU(afloat8, bfloat8, cfloat8, rozmiar * 2);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (float, 8MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	dodawanieCPU(afloat16, bfloat16, cfloat16, rozmiar * 4);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (float, 16MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	dodawanieCPU(adouble1, bdouble1, cdouble1, rozmiar / 8);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (double, 1MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	dodawanieCPU(adouble4, bdouble4, cdouble4, rozmiar / 2);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (double, 4MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	dodawanieCPU(adouble8, bdouble8, cdouble8, rozmiar);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (double, 8MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	dodawanieCPU(adouble16, bdouble16, cdouble16, rozmiar * 2);
	stopCPU = omp_get_wtime();
	printf("Czas dodawania CPU (double, 16MiB) [ms]: %f\n\n", 1000.0 * (stopCPU - startCPU));

	startCPU = omp_get_wtime();
	mnozenieCPU(aint1, bint1, cint1, rozmiar / 4);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (int, 1MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozenieCPU(aint4, bint4, cint4, rozmiar);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (int, 4MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozenieCPU(aint8, bint8, cint8, rozmiar * 2);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (int, 8MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozenieCPU(aint16, bint16, cint16, rozmiar * 4);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (int, 16MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozenieCPU(afloat1, bfloat1, cfloat1, rozmiar / 4);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (float, 1MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozenieCPU(afloat4, bfloat4, cfloat4, rozmiar);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (float, 4MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozenieCPU(afloat8, bfloat8, cfloat8, rozmiar * 2);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (float, 8MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozenieCPU(afloat16, bfloat16, cfloat16, rozmiar * 4);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (float, 16MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozenieCPU(adouble1, bdouble1, cdouble1, rozmiar / 8);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (double, 1MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozenieCPU(adouble4, bdouble4, cdouble4, rozmiar / 2);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (double, 4MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozenieCPU(adouble8, bdouble8, cdouble8, rozmiar);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (double, 8MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozenieCPU(adouble16, bdouble16, cdouble16, rozmiar * 2);
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia CPU (double, 16MiB) [ms]: %f\n\n", 1000.0 * (stopCPU - startCPU));

	startCPU = omp_get_wtime();
	potegowanieCPU(aint1, bint1, cint1, rozmiar / 4);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (int, 1MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	potegowanieCPU(aint4, bint4, cint4, rozmiar);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (int, 4MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	potegowanieCPU(aint8, bint8, cint8, rozmiar * 2);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (int, 8MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	potegowanieCPU(aint16, bint16, cint16, rozmiar * 4);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (int, 16MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	potegowanieCPU(afloat1, bfloat1, cfloat1, rozmiar / 4);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (float, 1MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	potegowanieCPU(afloat4, bfloat4, cfloat4, rozmiar);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (float, 4MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	potegowanieCPU(afloat8, bfloat8, cfloat8, rozmiar * 2);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (float, 8MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	potegowanieCPU(afloat16, bfloat16, cfloat16, rozmiar * 4);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (float, 16MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	potegowanieCPU(adouble1, bdouble1, cdouble1, rozmiar / 8);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (double, 1MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	potegowanieCPU(adouble4, bdouble4, cdouble4, rozmiar / 2);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (double, 4MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	potegowanieCPU(adouble8, bdouble8, cdouble8, rozmiar);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (double, 8MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	potegowanieCPU(adouble16, bdouble16, cdouble16, rozmiar * 2);
	stopCPU = omp_get_wtime();
	printf("Czas potegowania CPU (double, 16MiB) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));

	delete[] aint1;
	delete[] aint4;
	delete[] aint8;
	delete[] aint16;
	delete[] bint1;
	delete[] bint4;
	delete[] bint8;
	delete[] bint16;
	delete[] cint1;
	delete[] cint4;
	delete[] cint8;
	delete[] cint16;
	delete[] afloat1;
	delete[] afloat4;
	delete[] afloat8;
	delete[] afloat16;
	delete[] bfloat1;
	delete[] bfloat4;
	delete[] bfloat8;
	delete[] bfloat16;
	delete[] cfloat1;
	delete[] cfloat4;
	delete[] cfloat8;
	delete[] cfloat16;
	delete[] adouble1;
	delete[] adouble4;
	delete[] adouble8;
	delete[] adouble16;
	delete[] bdouble1;
	delete[] bdouble4;
	delete[] bdouble8;
	delete[] bdouble16;
	delete[] cdouble1;
	delete[] cdouble4;
	delete[] cdouble8;
	delete[] cdouble16;

	cudaFree(dev_aint1);
	cudaFree(dev_aint4);
	cudaFree(dev_aint8);
	cudaFree(dev_aint16);
	cudaFree(dev_bint1);
	cudaFree(dev_bint4);
	cudaFree(dev_bint8);
	cudaFree(dev_bint16);
	cudaFree(dev_cint1);
	cudaFree(dev_cint4);
	cudaFree(dev_cint8);
	cudaFree(dev_cint16);
	cudaFree(dev_afloat1);
	cudaFree(dev_afloat4);
	cudaFree(dev_afloat8);
	cudaFree(dev_afloat16);
	cudaFree(dev_bfloat1);
	cudaFree(dev_bfloat4);
	cudaFree(dev_bfloat8);
	cudaFree(dev_bfloat16);
	cudaFree(dev_cfloat1);
	cudaFree(dev_cfloat4);
	cudaFree(dev_cfloat8);
	cudaFree(dev_cfloat16);
	cudaFree(dev_adouble1);
	cudaFree(dev_adouble4);
	cudaFree(dev_adouble8);
	cudaFree(dev_adouble16);
	cudaFree(dev_bdouble1);
	cudaFree(dev_bdouble4);
	cudaFree(dev_bdouble8);
	cudaFree(dev_bdouble16);
	cudaFree(dev_cdouble1);
	cudaFree(dev_cdouble4);
	cudaFree(dev_cdouble8);
	cudaFree(dev_cdouble16);

	cudaDeviceReset();
	return 0;
}

void dodawanieCPU(int *a, int *b, int *c, int rozmiar)
{
	for (int i = 0; i < rozmiar; i++)
	{
		c[i] = a[i] + b[i];
	}
}

void mnozenieCPU(int *a, int *b, int *c, int rozmiar)
{
	for (int i = 0; i < rozmiar; i++)
	{
		c[i] = a[i] * b[i];
	}
}

void potegowanieCPU(int *a, int *b, int *c, int rozmiar)
{
	int wynik;
	for (int i = 0; i < rozmiar; i++)
	{
		wynik = 1;
		for (int j = 0; j < b[i]; j++)
		{
			wynik *= a[i];
		}
		c[i] = wynik;
	}
}

void dodawanieCPU(float *a, float *b, float *c, int rozmiar)
{
	for (int i = 0; i < rozmiar; i++)
	{
		c[i] = a[i] + b[i];
}
}

void mnozenieCPU(float *a, float *b, float *c, int rozmiar)
{
	for (int i = 0; i < rozmiar; i++)
	{
		c[i] = a[i] * b[i];
	}
}

void potegowanieCPU(float *a, float *b, float *c, int rozmiar)
{
	float wynik;
	for (int i = 0; i < rozmiar; i++)
	{
		wynik = 1;
		for (float j = 0; j < b[i]; j++)
		{
			wynik *= a[i];
		}
		c[i] = wynik;
	}
}

void dodawanieCPU(double *a, double *b, double *c, int rozmiar)
{
	for (int i = 0; i < rozmiar; i++)
	{
		c[i] = a[i] + b[i];
	}
}

void mnozenieCPU(double *a, double *b, double *c, int rozmiar)
{
	for (int i = 0; i < rozmiar; i++)
	{
		c[i] = a[i] * b[i];
	}
}

void potegowanieCPU(double *a, double *b, double *c, int rozmiar)
{
	double wynik;
	for (int i = 0; i < rozmiar; i++)
	{
		wynik = 1;
		for (double j = 0; j < b[i]; j++)
		{
			wynik *= a[i];
		}
		c[i] = wynik;
	}
}


#endif

#ifdef Zad4
#include <math.h>
__global__ void kernelDodawanieMacierzy(float *a, float *b, float *c, int rozmiar)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < rozmiar && j < rozmiar)
	{
		c[i * rozmiar + j] = a[i * rozmiar + j] + b[i * rozmiar + j];
	}
}
__global__ void kernelMnozenieMacierzy(float *a, float *b, float *c, int rozmiar, int sqrtRozmiar)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	float wynik = 0;
	if (i < sqrtRozmiar && j < sqrtRozmiar)
	{
		/*for (int k = 0; k < rozmiar; k++)
		{
			wynik += a[i * rozmiar + k] * b[k * rozmiar + j];
		}*/
		c[i * rozmiar + j] = wynik;
	}
}
__global__ void kernelDodawanieMacierzy(double *a, double *b, double *c, int rozmiar)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < rozmiar && j < rozmiar)
	{
		c[i * rozmiar + j] = a[i * rozmiar + j] + b[i * rozmiar + j];
	}
}
__global__ void kernelMnozenieMacierzy(double *a, double *b, double *c, int rozmiar, int sqrtRozmiar)
{
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	double wynik = 0;
	if (i < sqrtRozmiar && j < sqrtRozmiar)
	{
		for (int k = 0; k < rozmiar; k++)
		{
			wynik += a[i * rozmiar + k] * b[k * rozmiar + j];
		}
		c[i * rozmiar + j] = wynik;
	}
}

void dodawaniemacierzyCPU(float *a, float *b, float *c, int rozmiar);
void mnozeniemacierzyCPU(float *a, float *b, float *c, int rozmiar);
void dodawaniemacierzyCPU(double *a, double *b, double *c, int rozmiar);
void mnozeniemacierzyCPU(double *a, double *b, double *c, int rozmiar);

int main()
{
	cudaSetDevice(0);
	double startCPU, stopCPU;
	const int rozmiar = 1024;
	int liczbaBlokow;
	float czasGPU;
	cudaEvent_t startGPU, stopGPU;
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);

	float *afloat1 = new float[rozmiar * rozmiar / 4];
	float *dev_afloat1;
	cudaMalloc(&dev_afloat1, rozmiar * rozmiar * sizeof(float) / 4);
	cudaMemcpy(dev_afloat1, afloat1, rozmiar * rozmiar * sizeof(float) / 4, cudaMemcpyHostToDevice);
	float *bfloat1 = new float[rozmiar * rozmiar / 4];
	float *dev_bfloat1;
	cudaMalloc(&dev_bfloat1, rozmiar * rozmiar * sizeof(float) / 4);
	cudaMemcpy(dev_bfloat1, bfloat1, rozmiar * rozmiar * sizeof(float) / 4, cudaMemcpyHostToDevice);
	float *cfloat1 = new float[rozmiar * rozmiar / 4];
	float *dev_cfloat1;
	cudaMalloc(&dev_cfloat1, rozmiar * rozmiar * sizeof(float) / 4);
	cudaMemcpy(dev_cfloat1, cfloat1, rozmiar * rozmiar * sizeof(float) / 4, cudaMemcpyHostToDevice);
	liczbaBlokow = (rozmiar * rozmiar / 4 + rozmiar - 1) / rozmiar;
	cudaEventRecord(startGPU, 0);
	kernelDodawanieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_afloat1, dev_bfloat1, dev_cfloat1, rozmiar / 4);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas dodawania macierzy GPU (float, 1) [ms]: %f\n", czasGPU);
	cudaEventRecord(startGPU, 0);
	kernelMnozenieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_afloat1, dev_bfloat1, dev_cfloat1, rozmiar / 4, floor(sqrt(rozmiar / 4)) - 1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas mnozenia macierzy GPU (float, 1) [ms]: %f\n", czasGPU);
	startCPU = omp_get_wtime();
	dodawaniemacierzyCPU(afloat1, bfloat1, cfloat1, ceil(rozmiar * rozmiar / 4));
	stopCPU = omp_get_wtime();
	printf("Czas dodawania macierzy CPU (float, 1) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozeniemacierzyCPU(afloat1, bfloat1, cfloat1, ceil(rozmiar * rozmiar / 4));
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia macierzy CPU (float, 1) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	delete[] afloat1;
	cudaFree(dev_afloat1);
	delete[] bfloat1;
	cudaFree(dev_bfloat1);
	delete[] cfloat1;
	cudaFree(dev_cfloat1);

	float *afloat4 = new float[rozmiar * rozmiar];
	float *dev_afloat4;
	cudaMalloc(&dev_afloat4, rozmiar * rozmiar * sizeof(float));
	cudaMemcpy(dev_afloat4, afloat4, rozmiar * rozmiar * sizeof(float), cudaMemcpyHostToDevice);
	float *bfloat4 = new float[rozmiar * rozmiar];
	float *dev_bfloat4;
	cudaMalloc(&dev_bfloat4, rozmiar * rozmiar * sizeof(float));
	cudaMemcpy(dev_bfloat4, bfloat4, rozmiar * rozmiar * sizeof(float), cudaMemcpyHostToDevice);
	float *cfloat4 = new float[rozmiar * rozmiar];
	float *dev_cfloat4;
	cudaMalloc(&dev_cfloat4, rozmiar * rozmiar * sizeof(float));
	cudaMemcpy(dev_cfloat4, cfloat4, rozmiar * rozmiar * sizeof(float), cudaMemcpyHostToDevice);
	liczbaBlokow = (rozmiar * rozmiar + rozmiar - 1) / rozmiar;
	cudaEventRecord(startGPU, 0);
	kernelDodawanieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_afloat4, dev_bfloat4, dev_cfloat4, rozmiar);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas dodawania macierzy GPU (float, 4) [ms]: %f\n", czasGPU);
	cudaEventRecord(startGPU, 0);
	kernelMnozenieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_afloat4, dev_bfloat4, dev_cfloat4, rozmiar, floor(sqrt(rozmiar)) - 1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas mnozenia macierzy GPU (float, 4) [ms]: %f\n", czasGPU);
	startCPU = omp_get_wtime();
	dodawaniemacierzyCPU(afloat4, bfloat4, cfloat4, ceil(rozmiar * rozmiar));
	stopCPU = omp_get_wtime();
	printf("Czas dodawania macierzy CPU (float, 4) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozeniemacierzyCPU(afloat4, bfloat4, cfloat4, ceil(rozmiar * rozmiar));
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia macierzy CPU (float, 4) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	delete[] afloat4;
	cudaFree(dev_afloat4);
	delete[] bfloat4;
	cudaFree(dev_bfloat4);
	delete[] cfloat4;
	cudaFree(dev_cfloat4);

	float *afloat8 = new float[rozmiar * rozmiar * 2];
	float *dev_afloat8;
	cudaMalloc(&dev_afloat8, rozmiar * rozmiar * sizeof(float) * 2);
	cudaMemcpy(dev_afloat8, afloat8, rozmiar * rozmiar * sizeof(float) * 2, cudaMemcpyHostToDevice);
	float *bfloat8 = new float[rozmiar * rozmiar * 2];
	float *dev_bfloat8;
	cudaMalloc(&dev_bfloat8, rozmiar * rozmiar * sizeof(float) * 2);
	cudaMemcpy(dev_bfloat8, bfloat8, rozmiar * rozmiar * sizeof(float) * 2, cudaMemcpyHostToDevice);
	float *cfloat8 = new float[rozmiar * rozmiar * 2];
	float *dev_cfloat8;
	cudaMalloc(&dev_cfloat8, rozmiar * rozmiar * sizeof(float) * 2);
	cudaMemcpy(dev_cfloat8, cfloat8, rozmiar * rozmiar * sizeof(float) * 2, cudaMemcpyHostToDevice);
	liczbaBlokow = (rozmiar * rozmiar * 2 + rozmiar - 1) / rozmiar;
	cudaEventRecord(startGPU, 0);
	kernelDodawanieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_afloat8, dev_bfloat8, dev_cfloat8, rozmiar * 2);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas dodawania macierzy GPU (float, 8) [ms]: %f\n", czasGPU);
	cudaEventRecord(startGPU, 0);
	kernelMnozenieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_afloat8, dev_bfloat8, dev_cfloat8, rozmiar * 2, floor(sqrt(rozmiar * 2)) - 1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas mnozenia macierzy GPU (float, 8) [ms]: %f\n", czasGPU);
	startCPU = omp_get_wtime();
	dodawaniemacierzyCPU(afloat8, bfloat8, cfloat8, ceil(rozmiar * rozmiar * 2));
	stopCPU = omp_get_wtime();
	printf("Czas dodawania macierzy CPU (float, 8) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozeniemacierzyCPU(afloat8, bfloat8, cfloat8, ceil(rozmiar * rozmiar * 2));
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia macierzy CPU (float, 8) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	delete[] afloat8;
	cudaFree(dev_afloat8);
	delete[] bfloat8;
	cudaFree(dev_bfloat8);
	delete[] cfloat8;
	cudaFree(dev_cfloat8);

	float *afloat16 = new float[rozmiar * rozmiar * 4];
	float *dev_afloat16;
	cudaMalloc(&dev_afloat16, rozmiar * rozmiar * sizeof(float) * 4);
	cudaMemcpy(dev_afloat16, afloat16, rozmiar * rozmiar * sizeof(float) * 4, cudaMemcpyHostToDevice);
	float *bfloat16 = new float[rozmiar * rozmiar * 4];
	float *dev_bfloat16;
	cudaMalloc(&dev_bfloat16, rozmiar * rozmiar * sizeof(float) * 4);
	cudaMemcpy(dev_bfloat16, bfloat16, rozmiar * rozmiar * sizeof(float) * 4, cudaMemcpyHostToDevice);
	float *cfloat16 = new float[rozmiar * rozmiar * 4];
	float *dev_cfloat16;
	cudaMalloc(&dev_cfloat16, rozmiar * rozmiar * sizeof(float) * 4);
	cudaMemcpy(dev_cfloat16, cfloat16, rozmiar * rozmiar * sizeof(float) * 4, cudaMemcpyHostToDevice);
	liczbaBlokow = (rozmiar * rozmiar * 4 + rozmiar - 1) / rozmiar;
	cudaEventRecord(startGPU, 0);
	kernelDodawanieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_afloat16, dev_bfloat16, dev_cfloat16, rozmiar * 4);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas dodawania macierzy GPU (float, 16) [ms]: %f\n", czasGPU);
	cudaEventRecord(startGPU, 0);
	kernelMnozenieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_afloat16, dev_bfloat16, dev_cfloat16, rozmiar * 4, floor(sqrt(rozmiar * 4)) - 1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas mnozenia macierzy GPU (float, 16) [ms]: %f\n", czasGPU);
	startCPU = omp_get_wtime();
	dodawaniemacierzyCPU(afloat16, bfloat16, cfloat16, ceil(rozmiar * rozmiar * 4));
	stopCPU = omp_get_wtime();
	printf("Czas dodawania macierzy CPU (float, 16) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozeniemacierzyCPU(afloat16, bfloat16, cfloat16, ceil(rozmiar * rozmiar * 4));
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia macierzy CPU (float, 16) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	delete[] afloat16;
	cudaFree(dev_afloat16);
	delete[] bfloat16;
	cudaFree(dev_bfloat16);
	delete[] cfloat16;
	cudaFree(dev_cfloat16);

	double *adouble1 = new double[rozmiar * rozmiar / 8];
	double *dev_adouble1;
	cudaMalloc(&dev_adouble1, rozmiar * rozmiar * sizeof(double) / 8);
	cudaMemcpy(dev_adouble1, adouble1, rozmiar * rozmiar * sizeof(double) / 8, cudaMemcpyHostToDevice);
	double *bdouble1 = new double[rozmiar * rozmiar / 8];
	double *dev_bdouble1;
	cudaMalloc(&dev_bdouble1, rozmiar * rozmiar * sizeof(double) / 8);
	cudaMemcpy(dev_bdouble1, bdouble1, rozmiar * rozmiar * sizeof(double) / 8, cudaMemcpyHostToDevice);
	double *cdouble1 = new double[rozmiar * rozmiar / 8];
	double *dev_cdouble1;
	cudaMalloc(&dev_cdouble1, rozmiar * rozmiar * sizeof(double) / 8);
	cudaMemcpy(dev_cdouble1, cdouble1, rozmiar * rozmiar * sizeof(double) / 8, cudaMemcpyHostToDevice);
	liczbaBlokow = (rozmiar * rozmiar / 8 + rozmiar - 1) / rozmiar;
	cudaEventRecord(startGPU, 0);
	kernelDodawanieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_adouble1, dev_bdouble1, dev_cdouble1, rozmiar / 8);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas dodawania macierzy GPU (double, 1) [ms]: %f\n", czasGPU);
	cudaEventRecord(startGPU, 0);
	kernelMnozenieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_adouble1, dev_bdouble1, dev_cdouble1, rozmiar / 8, floor(sqrt(rozmiar / 8)) - 1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas mnozenia macierzy GPU (double, 1) [ms]: %f\n", czasGPU);
	startCPU = omp_get_wtime();
	dodawaniemacierzyCPU(adouble1, bdouble1, cdouble1, ceil(rozmiar * rozmiar / 8));
	stopCPU = omp_get_wtime();
	printf("Czas dodawania macierzy CPU (double, 1) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozeniemacierzyCPU(adouble1, bdouble1, cdouble1, ceil(rozmiar * rozmiar / 8));
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia macierzy CPU (double, 1) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	delete[] adouble1;
	cudaFree(dev_adouble1);
	delete[] bdouble1;
	cudaFree(dev_bdouble1);
	delete[] cdouble1;
	cudaFree(dev_cdouble1);

	double *adouble4 = new double[rozmiar * rozmiar / 2];
	double *dev_adouble4;
	cudaMalloc(&dev_adouble4, rozmiar * rozmiar * sizeof(double) / 2);
	cudaMemcpy(dev_adouble4, adouble4, rozmiar * rozmiar * sizeof(double) / 2, cudaMemcpyHostToDevice);
	double *bdouble4 = new double[rozmiar * rozmiar / 2];
	double *dev_bdouble4;
	cudaMalloc(&dev_bdouble4, rozmiar * rozmiar * sizeof(double) / 2);
	cudaMemcpy(dev_bdouble4, bdouble4, rozmiar * rozmiar * sizeof(double) / 2, cudaMemcpyHostToDevice);
	double *cdouble4 = new double[rozmiar * rozmiar / 2];
	double *dev_cdouble4;
	cudaMalloc(&dev_cdouble4, rozmiar * rozmiar * sizeof(double) / 2);
	cudaMemcpy(dev_cdouble4, cdouble4, rozmiar * rozmiar * sizeof(double) / 2, cudaMemcpyHostToDevice);
	liczbaBlokow = (rozmiar * rozmiar / 2 + rozmiar - 1) / rozmiar;
	cudaEventRecord(startGPU, 0);
	kernelDodawanieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_adouble4, dev_bdouble4, dev_cdouble4, rozmiar / 2);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas dodawania macierzy GPU (double, 4) [ms]: %f\n", czasGPU);
	cudaEventRecord(startGPU, 0);
	kernelMnozenieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_adouble4, dev_bdouble4, dev_cdouble4, rozmiar / 2, floor(sqrt(rozmiar / 2)) - 1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas mnozenia macierzy GPU (double, 4) [ms]: %f\n", czasGPU);
	startCPU = omp_get_wtime();
	dodawaniemacierzyCPU(adouble4, bdouble4, cdouble4, ceil(rozmiar * rozmiar / 2));
	stopCPU = omp_get_wtime();
	printf("Czas dodawania macierzy CPU (double, 4) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozeniemacierzyCPU(adouble4, bdouble4, cdouble4, ceil(rozmiar * rozmiar / 2));
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia macierzy CPU (double, 4) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	delete[] adouble4;
	cudaFree(dev_adouble4);
	delete[] bdouble4;
	cudaFree(dev_bdouble4);
	delete[] cdouble4;
	cudaFree(dev_cdouble4);

	double *adouble8 = new double[rozmiar * rozmiar];
	double *dev_adouble8;
	cudaMalloc(&dev_adouble8, rozmiar * rozmiar * sizeof(double));
	cudaMemcpy(dev_adouble8, adouble8, rozmiar * rozmiar * sizeof(double), cudaMemcpyHostToDevice);
	double *bdouble8 = new double[rozmiar * rozmiar];
	double *dev_bdouble8;
	cudaMalloc(&dev_bdouble8, rozmiar * rozmiar * sizeof(double));
	cudaMemcpy(dev_bdouble8, bdouble8, rozmiar * rozmiar * sizeof(double), cudaMemcpyHostToDevice);
	double *cdouble8 = new double[rozmiar * rozmiar];
	double *dev_cdouble8;
	cudaMalloc(&dev_cdouble8, rozmiar * rozmiar * sizeof(double));
	cudaMemcpy(dev_cdouble8, cdouble8, rozmiar * rozmiar * sizeof(double), cudaMemcpyHostToDevice);
	liczbaBlokow = (rozmiar * rozmiar + rozmiar - 1) / rozmiar;
	cudaEventRecord(startGPU, 0);
	kernelDodawanieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_adouble8, dev_bdouble8, dev_cdouble8, rozmiar);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas dodawania macierzy GPU (double, 8) [ms]: %f\n", czasGPU);
	cudaEventRecord(startGPU, 0);
	kernelMnozenieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_adouble8, dev_bdouble8, dev_cdouble8, rozmiar, floor(sqrt(rozmiar)) - 1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas mnozenia macierzy GPU (double, 8) [ms]: %f\n", czasGPU);
	startCPU = omp_get_wtime();
	dodawaniemacierzyCPU(adouble8, bdouble8, cdouble8, ceil(rozmiar * rozmiar));
	stopCPU = omp_get_wtime();
	printf("Czas dodawania macierzy CPU (double, 8) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozeniemacierzyCPU(adouble8, bdouble8, cdouble8, ceil(rozmiar * rozmiar));
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia macierzy CPU (double, 8) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	delete[] adouble8;
	cudaFree(dev_adouble8);
	delete[] bdouble8;
	cudaFree(dev_bdouble8);
	delete[] cdouble8;
	cudaFree(dev_cdouble8);

	double *adouble16 = new double[rozmiar * rozmiar * 2];
	double *dev_adouble16;
	cudaMalloc(&dev_adouble16, rozmiar * rozmiar * sizeof(double) * 2);
	cudaMemcpy(dev_adouble16, adouble16, rozmiar * rozmiar * sizeof(double) * 2, cudaMemcpyHostToDevice);
	double *bdouble16 = new double[rozmiar * rozmiar * 2];
	double *dev_bdouble16;
	cudaMalloc(&dev_bdouble16, rozmiar * rozmiar * sizeof(double) * 2);
	cudaMemcpy(dev_bdouble16, bdouble16, rozmiar * rozmiar * sizeof(double) * 2, cudaMemcpyHostToDevice);
	double *cdouble16 = new double[rozmiar * rozmiar * 2];
	double *dev_cdouble16;
	cudaMalloc(&dev_cdouble16, rozmiar * rozmiar * sizeof(double) * 2);
	cudaMemcpy(dev_cdouble16, cdouble16, rozmiar * rozmiar * sizeof(double) * 2, cudaMemcpyHostToDevice);
	liczbaBlokow = (rozmiar * rozmiar * 2 + rozmiar - 1) / rozmiar;
	cudaEventRecord(startGPU, 0);
	kernelDodawanieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_adouble16, dev_bdouble16, dev_cdouble16, rozmiar * 2);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas dodawania macierzy GPU (double, 16) [ms]: %f\n", czasGPU);
	cudaEventRecord(startGPU, 0);
	kernelMnozenieMacierzy << <dim3(liczbaBlokow, 1), dim3(rozmiar, 1) >> > (dev_adouble16, dev_bdouble16, dev_cdouble16, rozmiar * 2, floor(sqrt(rozmiar * 2)) - 1);
	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&czasGPU, startGPU, stopGPU);
	printf("Czas mnozenia macierzy GPU (double, 16) [ms]: %f\n", czasGPU);
	startCPU = omp_get_wtime();
	dodawaniemacierzyCPU(adouble16, bdouble16, cdouble16, ceil(rozmiar * rozmiar * 2));
	stopCPU = omp_get_wtime();
	printf("Czas dodawania macierzy CPU (double, 16) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	startCPU = omp_get_wtime();
	mnozeniemacierzyCPU(adouble16, bdouble16, cdouble16, ceil(rozmiar * rozmiar * 2));
	stopCPU = omp_get_wtime();
	printf("Czas mnozenia macierzy CPU (double, 16) [ms]: %f\n", 1000.0 * (stopCPU - startCPU));
	delete[] adouble16;
	cudaFree(dev_adouble16);
	delete[] bdouble16;
	cudaFree(dev_bdouble16);
	delete[] cdouble16;
	cudaFree(dev_cdouble16);

	cudaDeviceReset();
	return 0;
}

void dodawaniemacierzyCPU(float *a, float *b, float *c, int rozmiar)
{
	int size = floor(sqrt(rozmiar)-1);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			c[i * size + j] = a[i * size + j] + b[i * size + j];
		}
	}
}
void mnozeniemacierzyCPU(float *a, float *b, float *c, int rozmiar)
{
	int size = floor(sqrt(rozmiar)-1);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			c[i * size + j] = 0;
			for (int k = 0; k < size; k++)
			{
				c[i * size + j] += a[i * size + k] * b[k * size + j];
			}
		}
	}
}
void dodawaniemacierzyCPU(double *a, double *b, double *c, int rozmiar)
{
	int size = floor(sqrt(rozmiar)-1);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			c[i * size + j] = a[i * size + j] + b[i * size + j];
		}
	}
}
void mnozeniemacierzyCPU(double *a, double *b, double *c, int rozmiar)
{
	int size = floor(sqrt(rozmiar)-1);
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			c[i * size + j] = 0;
			for (int k = 0; k < size; k++)
			{
				c[i * size + j] += a[i * size + k] * b[k * size + j];
			}
		}
	}
}

#endif

#ifdef Zad5

#endif
