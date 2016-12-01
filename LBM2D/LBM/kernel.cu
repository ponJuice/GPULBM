#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lbm_kernels.cuh"
#include <stdio.h>
#include <math.h>

#include "lbm_data.h"
#include "lbm_util.h"

void setConstantMemory(){
	LBM::IVector3* h_w;
	h_w = new LBM::IVector3[15];

	LBM::setVector(&h_w[0],  0,  0,  0);
	LBM::setVector(&h_w[1],  1,  0,  0);
	LBM::setVector(&h_w[2], -1,  0,  0);
	LBM::setVector(&h_w[3],  0,  1,  0);
	LBM::setVector(&h_w[4],  0, -1,  0);
	LBM::setVector(&h_w[5],  0,  0,  1);
	LBM::setVector(&h_w[6],  0,  0, -1);
	LBM::setVector(&h_w[7],  1,  1,  1);
	LBM::setVector(&h_w[8], -1, -1, -1);
	LBM::setVector(&h_w[9], -1,  1,  1);
	LBM::setVector(&h_w[10], 1, -1, -1);
	LBM::setVector(&h_w[11], 1, -1,  1);
	LBM::setVector(&h_w[12],-1,  1, -1);
	LBM::setVector(&h_w[13], 1,  1,  1);
	LBM::setVector(&h_w[14],-1, -1, -1);

	cudaMemcpyToSymbol(w,h_w,sizeof(LBM::IVector3)*15);

	delete[] h_w;
}

void allCudaFree(){
	for(int n = info.offset;n < (info.offset + info.size);n++){
		ce = cudaFree(d_point[n].a);
		if(ce != cudaSuccess){
			printf("failed cudaFree <<point.a>>\n");
			printf("error : %s\n",cudaGetErrorString(ce));
		}
	}
	ce = cudaFree(d_point);
	if(ce != cudaSuccess){
		printf("failed cudaFree <<point>>\n");
		printf("error : %s\n",cudaGetErrorString(ce));
	}
	ce = cudaFree(d_info);
	if(ce != cudaSuccess){
		printf("failed cudaFree <<info>>\n");
		printf("error : %s\n",cudaGetErrorString(ce));
	}
	cudaDeviceReset();
}

void safery(cudaError ce,char str[]){
	if(ce == cudaSuccess)
		return;
	printf("%s\n",str);
	printf("error : %s\n",cudaGetErrorString(ce));
	allCudaFree();
}

int main(){
	cudaError ce;

	int device_num = 0;	//TITANを使用
	cudaDeviceProp dev;
	cudaGetDeviceProperties(&dev,device_num);

	//コンスタントメモリに係数wをコピー
	setConstantMemory();

	//計算領域の情報
	LBM::LatticeInfo info;
	LBM::getLbmLatticeInfo(&info,1,4,4,4,15);
	LBM::printLatticeInfo(&info);

	int threads = Math::min(dev.maxThreadsPerBlock,info.size);//32 * ((info.size - 1) / 32 + 1);
	int block = (int)ceil((double)threads/(double)dev.maxThreadsPerBlock);

	printf("threads : %d\n",threads);
	printf("block : %d\n",block);

	int a_array_size = sizeof(double) * info.direct_num;
	int point_size = sizeof(LBM::LBMPoint);

	int a = sizeof(double*);

	printf("a_size : %d\n",a_array_size);
	printf("point size : %d\n",point_size);

	LBM::LBMPoint* d_point;
	LBM::LBMPoint h_point;
	LBM::LatticeInfo* d_info;

	ce = cudaMalloc((void**)&d_point,point_size*info.x_max_out*info.y_max_out*info.z_max_out);
	safery(ce,"failed cudaMalloc <<point>>");

	double* _a;
	for(int n = 0;n < info.size;n++){
		//参照領域も含めたインデックスを算出
		int index_out = info.offset 
			+ n
			+ (n / ( info.x_max_in * info.y_max_in ))
				* (info.x_max_out * info.y_max_out - info.x_max_in * info.y_max_in) 
			+ (n / info.x_max_in) % 2 
				* info.max_speed
				* 2;

		//各次元のインデックス算出
		int x = index_out % info.y_max_out;
		int y = (index_out / info.x_max_out) % info.y_max_out;
		int z = index_out % (info.x_max_out * info.y_max_out);

		//分布関数用の配列を作成
		ce = cudaMalloc((void**)&_a,a_array_size);
		safery(ce,"failed cudaMalloc <<point.a>>");

		//初期値を設定
		h_point.density = 1;
		h_point.vx = 2;
		h_point.vy = 3;
		h_point.vz = 4;
		//マスクの設定
		if(x == 0){
			h_point.mask = INFLOW;
		}else if(x == (info.x_max_out - 1)){
			h_point.mask = OUTFLOW;
		}
		if(y == 0){
			h_point.mask = OUTFLOW;
		}else if(y == (info.y_max_out - 1)){
			h_point.mask = OUTFLOW;
		}
		if(z == 0){
			h_point.mask = OUTFLOW;
		}else if(z == (info.z_max_out -1 )){
			h_point.mask = OUTFLOW;
		}

		//分布関数用配列を参照するようにする
		h_point.a = _a;
		//情報をコピー
		ce = cudaMemcpy(&d_point[index_out],&h_point,point_size,cudaMemcpyHostToDevice);
		if(ce != cudaSuccess){
			printf("failed cudaMemcpy <<point.a>>\n");
			printf("error : %s\n",cudaGetErrorString(ce));
			goto allCudaFree;
		}
	}

	int info_size = sizeof(LBM::LatticeInfo);
	ce = cudaMalloc((void**)&d_info,info_size);
	if(ce != cudaSuccess){
		printf("failed cudaMalloc <<info>>\n");
		printf("error : %s\n",cudaGetErrorString(ce));
		goto allCudaFree;
	}

	ce = cudaMemcpy(d_info,&info,info_size,cudaMemcpyHostToDevice);	
	if(ce != cudaSuccess){
		printf("failed cudaMemcpy <<info>>\n");
		printf("error : %s\n",cudaGetErrorString(ce));
		goto allCudaFree;
	}

	//流入情報の設定

	//計算実行
	lbm_calc<<<block,threads>>>(d_info,d_point);
	//lbm_test3<<<1,1>>>(d_point,_a,info.direct_num);
	cudaThreadSynchronize();
	ce = cudaGetLastError();
	if(ce != cudaSuccess){
		printf("failed karnel <<lbm_calc>>\n");
		printf("error : %s\n",cudaGetErrorString(ce));
		goto allCudaFree;
	}
	

allCudaFree:
	for(int n = info.offset;n < (info.offset + info.size);n++){
		ce = cudaFree(d_point[n].a);
		if(ce != cudaSuccess){
			printf("failed cudaFree <<point.a>>\n");
			printf("error : %s\n",cudaGetErrorString(ce));
		}
	}
	ce = cudaFree(d_point);
	if(ce != cudaSuccess){
		printf("failed cudaFree <<point>>\n");
		printf("error : %s\n",cudaGetErrorString(ce));
	}
	ce = cudaFree(d_info);
	if(ce != cudaSuccess){
		printf("failed cudaFree <<info>>\n");
		printf("error : %s\n",cudaGetErrorString(ce));
	}
	cudaDeviceReset();

	return 0;
}