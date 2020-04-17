#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>

#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
void atomicAdd(unsigned int*, unsigned int);
#endif

#define BLOCK_SIZE 256

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) 
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// -----------------------------           Kernel FUNCTIONS           ----------------------------- //

__global__ void reduceKernel( float* reduceResult, const float* data, bool max )
{
    //indices del thread (global y local)
    int globalThread = threadIdx.x + blockDim.x*blockIdx.x;
	int localThread = threadIdx.x;

    extern __shared__ float auxData[];
    auxData[localThread] = data[globalThread];
    __syncthreads();

    for (int i = blockDim.x >> 1; i > 0; i >>= 1)
    {
        if (localThread < i)
        {
            if (max)
                auxData[localThread] = fmaxf(auxData[localThread], auxData[localThread + i]);
            else
                auxData[localThread] = fminf(auxData[localThread], auxData[localThread + i]);
        }
        __syncthreads();
    }

    if (localThread == 0)
        reduceResult[blockIdx.x] = auxData[0];
}

__global__ void histogramKernel(unsigned int* result, const float* data, int numBins, int imageSize, float min, float range)
{
    //indices del thread (global y local)
    int globalThread = threadIdx.x + blockDim.x*blockIdx.x;
	int localThread = threadIdx.x;

    if (globalThread >= imageSize)
        return;

    int bin = (data[globalThread] - min) / range * numBins; 
	//suma atomica, actua como un semaforo para evitar que los thread sumen valores erroneamente(condicion de carrera)
	atomicAdd(&(result[bin]), 1); 
}

__global__ void exclusiveScanKernel(unsigned int* result, const unsigned int* data, int numBins)
{
    //indice del thread
	int localThread = threadIdx.x;
    //variables que intercambiaran el valor de 0 a 1 viceversa
    int pout = 0, pin = 1;
    //memoria compartida
    extern __shared__ float auxData[];

    //el primer elemento le igualamos a 0, los demas copiamos el valor de entrada
    auxData[localThread] = (localThread > 0) ? data[localThread - 1] : 0;
    __syncthreads();

    for (int offset = 1; offset < numBins; offset <<=1 )
    {
        //intercambiamos los valores de pout y pin
        pout = 1 - pout;
        pin =  1 - pout;

        if (localThread >= offset)
            auxData[pout * numBins + localThread] = auxData[pin * numBins + localThread] + auxData[pin * numBins + localThread - offset];
        else
            auxData[pout * numBins + localThread] = auxData[pin * numBins + localThread];
        __syncthreads();
    }
    result[localThread] = auxData[pout * numBins + localThread];
}

// -----------------------------           CPU FUNCTIONS           ----------------------------- //
float getMaxMinLuminance(const float* const d_logLuminance, int imageSize, bool maxmin)
{
    int size = imageSize;
    float* reducedVector = NULL;
    
    //numero de threads (Block_size)
	int blocksize = BLOCK_SIZE; 
	//numero de bloques (grid_size)
	int gridsize = ceil(1.0f*size / blocksize);

    int sizeSharedMemory = blocksize * sizeof(float);
    
    while (true)
    {
        //vector para almacenar los resultados de la reduccion
		float* reducedResult;
        checkCudaErrors(cudaMalloc(&reducedResult, gridsize * sizeof(float)));

        //primera iteracion con los datos al completo
        if(reducedVector == NULL)
            reduceKernel << <gridsize, blocksize, sizeSharedMemory >> >(reducedResult, d_logLuminance, maxmin);
        //demas iteraciones con el vector de datos reducido a la mitad
        else
            reduceKernel <<<gridsize, blocksize, sizeSharedMemory>>>(reducedResult, reducedVector, maxmin);

        //sincronizamos los bloques para que terminen tod@s a la vez
        cudaDeviceSynchronize();

        if (reducedVector != NULL) 
			checkCudaErrors(cudaFree(reducedVector));
		reducedVector = reducedResult;

        //si se cumple, es que hemos llegado a la reduccion final
	    if (gridsize == 1) 
	    {
		    //copiamos el resultado de la reduccion en CPU
		    float result;
		    checkCudaErrors(cudaMemcpy(&result, reducedResult, sizeof(float), cudaMemcpyDeviceToHost));
		    //salimos del bucle y devolvemos el resultado
		    return result;
	    }
        //actualizamos el valor de gridSize disminuyendolo por el tamaño del bloque
        size = gridsize;
	    gridsize = ceil(1.0f*size / blocksize);
        //checkCudaErrors(cudaFree(reducedResult));
    }
}

unsigned int* histogram(const float* const d_logLuminance, int numBins, int imageSize, float min, float range)
{
    //numero de threads (Block_size)
	int blocksize = BLOCK_SIZE; 
	//numero de bloques (Grid_size)
	int gridsize = ceil(1.0f*imageSize / blocksize);
    unsigned int* histogramResult;
    //reservamos memoria
	checkCudaErrors(cudaMalloc(&histogramResult, numBins * sizeof(unsigned int)));
    //inicializamos a 0 los valores
	checkCudaErrors(cudaMemset(histogramResult, 0, numBins * sizeof(unsigned int)));

    //llamada al kernel
    histogramKernel <<<gridsize, blocksize >>> (histogramResult, d_logLuminance, numBins, imageSize, min, range);
    cudaDeviceSynchronize();

    return histogramResult;
}

void calculate_cdf(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  /* TODO
    1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance 
    //usaremos las funciones "fmaxf" y "fminf" que ofrece la APi matematica de cuda
	2) Obtener el rango a representar
	3) Generar un histograma de tod@s los valores del canal logLuminance usando la formula 
	bin = (Lum [i] - lumMin) / lumRange * numBins
	4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf) 
	de los valores de luminancia. Se debe almacenar en el puntero c_cdf
  */
    int imageSize = numRows * numCols;
    //1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance 
    max_logLum = getMaxMinLuminance(d_logLuminance,imageSize, true);
    min_logLum = getMaxMinLuminance(d_logLuminance,imageSize, false);

    //2) Obtener el rango a representar
    float range = max_logLum - min_logLum;

    //3) Generar un histograma de tod@s los valores del canal logLuminance usando la formula 
    unsigned int* histogramResult = histogram(d_logLuminance, numBins, imageSize, min_logLum, range);

    /*4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf) 
	de los valores de luminancia. Se debe almacenar en el puntero c_cdf*/
    exclusiveScanKernel <<<1, numBins, 2*numBins*sizeof(unsigned int) >>> (d_cdf, histogramResult, numBins);
    //liberamos memoria del histograma
    checkCudaErrors(cudaFree(histogramResult));
}

