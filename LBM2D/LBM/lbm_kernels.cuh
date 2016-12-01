#pragma once
#include "lbm_data.h"
#define NORMAL 0;
#define BOUNCE 1;
#define INFLOW 2;
#define OUTFLOW 3;

__constant__ double* w;
__constant__ LBM::IVector3* e;
__constant__ double tau;
__constant__ double c;
__constant__ LBM::LBMPoint c_inflow;
__constant__ LBM::LatticeInfo l_info;

__global__ void lbm_calc(LBM::LatticeInfo* l_info,LBM::LBMPoint* points,LBM::LBMPoint* next);

__global__ void lbm_test1();
__global__ void lbm_test2();
__global__ void lbm_test3(LBM::LBMPoint* point,double* a,int size);