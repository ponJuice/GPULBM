#pragma once
#include "lbm_kernels.cuh"
#include <stdio.h>


__device__ void calcPeq(int* peq,LBM::LBMPoint* point,int a){
	double e_dot_v = e[a].x * point->vx + e[a].y * point->vy + e[a].z * point->vz;
	double v_dot_v = point->.vx * point->vx + point->vy * point->vy + point->vz * point->vz;
	(*peq) = w[a] * point->density * (1.0
		+ ( 3.0 * e_dot_v )
		+ ( 9.0 / 2.0 ) * ( e_dot_v * e_dot_v )
		- ( 3.0 / 2.0 ) * v_dot_v );
}

__device__ void calcPe(double* pe,double peq,LBM::LBMPoint* point,int a){
	(*pe) = point->a[a] - ( 1.0 / tau) * (point->a[a] - peq);
}

__device__ void calcDensityAndVelocity(LBM::LBMPoint* target){
	
}

__global__ int calc_roop_num(int max_threadIdx_par_block,int threadIdx_in_block){
	return roop = (l_info.size - threadIdx_in_block - 1) / max_threadIdx_par_block + 1;
}

__global__ void lbm_calc(LBM::LBMPoint* points,LBM::LBMPoint* next){
	int max_threadIdx_par_block = blockDim.x * gridDim.x;
	int threadIdx_in_block = threadIdx.x + blockIdx.x * blockDim.x;

	int roop_max = calc_roop_num(max_threadIdx_par_block,threadIdx_in_block);
	
	for(int roop = 0;roop < roop_max;roop++){
		//スレッドIDとブロックIDからインデックスを算出
		int index_in = threadIdx_in_block + max_threadIdx_par_block * roop;
	
		//境界を超える場合は0
		int branch = index_in >= 0;
		branch *= index_in < l_info.size;

		//参照領域も含めたインデックスを算出
		int index_out = l_info.offset 
			+ index_in 
			+ (index_in / ( l_info.x_max_in * l_info.y_max_in ))
				* (l_info.x_max_out * l_info.y_max_out - l_info.x_max_in * l_info.y_max_in) 
			+ (index_in / l_info.x_max_in) % 2 
				* l_info.max_speed
				* 2;
		index_out = index_out * branch;

		//printf("lbm_calc in : %d  out : %d size : %d branch : %d\n",index_in,index_out,l_info->size,branch);

		//printf("index_out : %d  struct : %d\n",index_out,points[index_out].a);

		//初期化
		next[index_out].density = 0;
		next[index_out].vx = 0;
		next[index_out].vy = 0;
		next[index_out].vz = 0;

		//1ステップ計算
		for(int a = 0;a < l_info.direct_num;a++){
			//printf("	access a : %f index : %d\n",points[index_out].a[n],index_in);
			double peq;
			double pe;

			int x = index_out % l_info.y_max_out;
			int y = (index_out / l_info.x_max_out) % l_info.y_max_out;
			int z = index_out % (l_info.x_max_out * l_info.y_max_out);

			int ax = x + e[a].x;
			int ay = y + e[a].y;
			int az = z + e[a].z;

			index_a = index_out + e[a].x + l_info.x_max_out * e[a].y + l_info.x_max_out * l_info.y_max_out * e[a].z;

			calcPeq(&peq,&points[index_a],a);
			calcPe(&pe,peq,&points[index_a],a);

			int m = a % 2;
			double bounce_pe = points[index_out].a[a + (m - 1) + m];	//bounce-back境界での値

			int normal = point[index_a].mask && NORMAL;	//通常なら1
			int bounce_back = point[index_a].mask && BOUNCE;	//bounce-back境界なら1
			int inflow = point[index_a].mask && INFLOW;	//流入境界
			int outflow = point[index_a].mask && OUTFLOW; //流出境界

			next[index_out].a[a] = normal * pe + bounce_back * bounce_pe + inflow * c_inflow.[a] + outflow * point[index_out].a[a];

			next[index_out].density += next[index_out].a[a];

			next[index_out].vx += e[a].x * c * pe;
			next[index_out].vy += e[a].y * c * pe;
			next[index_out].vz += e[a].z * c * pe;
		}

		next[index_out].vx /= next[index_out].density;
		next[index_out].vy /= next[index_out].density;
		next[index_out].vz /= next[index_out].density;
	}

}

__global__ void lbm_test1(){
	int index = threadIdx.x;

	printf("test1 %d\n",index);
}

__global__ void lbm_test2(){
	int index = threadIdx.x;

	printf("test2 %d\n",index);
}

__global__ void lbm_test3(LBM::LBMPoint* point,double* a,int size){
	for(int n = 0;n < size;n++){
		a[n] = 0;
	}
	printf("array a access sucessed\n");
	printf("density : %f vx : %f vy : %f vz : %f\n",point->density,point->vx,point->vy,point->vz);
	printf("array : %d  struct : %d\n",a,point->a);
	if(a == point->a){
		printf("equal\n");
	}else{
		printf("no equal\n");
	}
	for(int n = 0;n < size;n++){
		point->a[n] = 0;
	}
	printf("struct a access sucessed\n");
}