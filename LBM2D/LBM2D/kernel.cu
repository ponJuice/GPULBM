#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <streambuf>
#include <string>

#define NORMAL 0
#define BOUNDARY 1
#define INFLOW 2
#define OUTFLOW 3

#define DIRECTION 9

typedef struct {
	double* a;
	double density;
	double vx,vy,vz;
	int mask;
}LBMPoint;

typedef struct{
	double vx,vy,vz;
	double density;
}LBMResult;

typedef struct {
	int x, y, z;			//�i�q�_��
	int directionNum;		//���x�x�N�g���̐�
	double deltaTime;		//���ԍ���
	double deltaLength;		//��ԍ���
	int lattice_num;		//��\�i�q�_��
	double density;			//�������x[kg/m^3]
	double lambda;			//���ώ��R�s��[nm]
	double vx,vy,vz;		//�������x[m/s]
	double cld;				//��\����[m]
	double cv;				//��\���x[m/s]
	double M;				//���q��(��C�̏ꍇ�͋�C�̕��σ�������)[kg/mol]
	double T;				//���x[K]
	double mu;				//�S���W��[Pa�Es]
	double R;				//�����C�̒萔
	LBMPoint inflow;		//����
}LBMConfig;

typedef struct{
	int x, y, z;			//�i�q�_��
	int directionNum;		//���x�x�N�g���̐�
	double deltaTime;		//���ԍ���
	double deltaLength;		//��ԍ���
	double density;			//�������x[kg/m^3]
	double lambda;			//���ώ��R�s��[nm]
	double vx,vy,vz;		//�������x[m/s]
	double cld;				//��\����[m]
	double cv;				//��\���x[m/s]
	double* w;
	double* ex;
	double* ey;
	double* ez;
	LBMPoint inflow;		//����
}LBMInfo;

typedef struct {
	double inflow_a[DIRECTION];	//�����̕��z�֐��l
	double tau;	//�ɘa����
	double c;	//�ڗ����x
	double density;	//�������x
	double vx,vy,vz;	//�������x
	int direct_num;	//������
	int max_speed;	//�ő呬�x
	int x,y,z;		//�e�����̍ő吔
	int offset;		//�ŏ��̌v�Z�_�܂ł̃I�t�Z�b�g
	long size_in;	//�v�Z�_�̐�
	long size_out;	//�i�q�_�̐�
	int x_max_out;	//�Q�Ɨ̈���܂߂�x�����̊i�q��
	int y_max_out;	//�Q�Ɨ̈���܂߂�y�����̊i�q��
	int z_max_out;	//�Q�Ɨ̈���܂߂�z�����̊i�q��
	int x_max_in;	//�v�Z�̈��x�����̊i�q��
	int y_max_in;	//�v�Z�̈��y�����̊i�q��
	int z_max_in;	//�v�Z�̈��z�����̊i�q��
}LBMData;

__constant__ LBMData c_data;
__constant__ double w[DIRECTION];
__constant__ double ex[DIRECTION];
__constant__ double ey[DIRECTION];
__constant__ double ez[DIRECTION];
__constant__ int invtert_table[DIRECTION];

namespace Math{
	int min(int a, int b) {
		int t = a <= b;
		return t*a + (1 - t)*b;
	}

	int max(int a, int b) {
		int t = a >= b;
		return t*a + (1 - t)*b;
	}
}

__device__ void calcPeq(double* peq,LBMPoint* point,int a){
	double e_dot_v = ex[a] * point->vx + ey[a] * point->vy + ez[a] * point->vz;
	double v_dot_v = point->vx * point->vx + point->vy * point->vy + point->vz * point->vz;
	(*peq) = w[a] * point->density * (1.0
		+ ( 3.0 * e_dot_v )
		+ ( 9.0 / 2.0 ) * ( e_dot_v * e_dot_v )
		- ( 3.0 / 2.0 ) * v_dot_v );
}

__device__ void calcPa(double* pe,double peq,LBMPoint* point,int a){
	(*pe) = point->a[a] - ( 1.0 / c_data.tau) * (point->a[a] - peq);
}

__device__ double calcBoundPa(LBMPoint* p,int index_out,int a){
	return p[index_out].a[invtert_table[a]];
}

__global__ void testKernel(){}

__global__ void initKernel(LBMPoint* p,double* pa){
	int max_threadIdx_par_block = blockDim.x * gridDim.x;
	int threadIdx_in_block = threadIdx.x + blockIdx.x * blockDim.x;

	int max_roop = (c_data.size_out - threadIdx_in_block - 1) / (double)max_threadIdx_par_block + 1;

	for(int roop = 0;roop < max_roop ;roop++){
		int index = threadIdx_in_block + max_threadIdx_par_block * roop;
	
		//�e�����̃C���f�b�N�X�l�Z�o
		int x = index % c_data.y_max_out;
		int y = (index / c_data.x_max_out) % c_data.y_max_out;
		int z = index / (c_data.x_max_out * c_data.y_max_out);

		//printf("init kernel (x,y,z) : (%2d,%2d,%2d)\n",x,y,z);

		//�}�X�N�̏�����
		//���̃J�[�l������x,y,z�͎Q�Ɨ̈���܂ނ̂ŗ����A���o���E�̊i�q�̌v�Z�͈ȉ��̂悤�ɂȂ�
		int under_x = x < c_data.max_speed;
		int over_x = x >= c_data.max_speed + c_data.x_max_in;
	
		int under_y = y < c_data.max_speed;
		int over_y = y >= c_data.max_speed + c_data.y_max_in;

		int under_z = z < c_data.max_speed;
		int over_z = z >= c_data.max_speed + c_data.z_max_in;

		//�}�X�N�̏�����(��Q�� �e�X�g �A�i���O)
		int cx = c_data.x_max_out / 6;
		int cy = c_data.y_max_out / 2;
		int r = 8;
		double _x = cx - x;
		double _y = cy - y;
		int in = (_x * _x + _y * _y) <= (r * r);	//�����Ă�����:1

		int is2D = c_data.z_max_in <= c_data.max_speed;

		int mask = 0;
		mask = (mask == 0) * BOUNDARY * in;
		mask += (mask == 0) * INFLOW * under_x;
		mask += (mask == 0) * OUTFLOW * over_x;
		mask += (mask == 0) * OUTFLOW * under_y;
		mask += (mask == 0) * OUTFLOW * over_y;
		mask += (mask == 0) * OUTFLOW * under_z * (1 - is2D);
		mask += (mask == 0) * OUTFLOW * over_z * (1 - is2D);
		p[index].mask = mask;

		//���x�̏�����(�ő̂̒���������0)
		p[index].density = c_data.density * (mask != BOUNDARY);;
		
		//���x�̏�����(�ő̂̒���������0)
		p[index].vx = c_data.vx * (mask != BOUNDARY);
		p[index].vy = c_data.vy * (mask != BOUNDARY);
		p[index].vz = c_data.vz * (mask != BOUNDARY);

		//���z�֐��l��������
		//�܂����z�֐��l�̔z�񂩂�g�p���镔�������o��
		int index_a = index * c_data.direct_num;
		p[index].a = &pa[index_a];
		//printf("pa : %2d\n",&p[index].a[0]);
		//�����l�v�Z
		for(int n = 0;n < c_data.direct_num;n++){
			calcPeq(&p[index].a[n],&p[index],n);
		}

		/*printf("init kernel t:%2d | b:%2d | i:%2d | x:%2d | y:%2d | z:%2d | 2D:%2d | de:%g | vx:%g | vy:%g | vx:%g | mask:%2d | a[0]:%g | a[1]:%g | a[2]:%g | a[3]:%g | a[4]:%g | a[5]:%g | a[6]:%g | a[7]:%g | a[8]:%g \n\n"
			,threadIdx.x,blockIdx.x,index,x,y,z,is2D,p[index].density,p[index].vx,p[index].vy,p[index].vz,p[index].mask
			,p[index].a[0],p[index].a[1],p[index].a[2],p[index].a[3],p[index].a[4],p[index].a[5],p[index].a[6],p[index].a[7],p[index].a[8]);*/
		
	}


}

__global__ void lbmCopyKernel(LBMPoint* dst,LBMPoint* src){
	int max_threadIdx_par_block = blockDim.x * gridDim.x;
	int threadIdx_in_block = threadIdx.x + blockIdx.x * blockDim.x;

	int max_roop = (c_data.size_out - threadIdx_in_block - 1) / (double)max_threadIdx_par_block + 1;

	for(int roop = 0;roop < max_roop ;roop++){
		int index = threadIdx_in_block + max_threadIdx_par_block * roop;
		dst[index].density = src[index].density;
		dst[index].mask = src[index].mask;
		dst[index].vx = src[index].vx;
		dst[index].vy = src[index].vy;
		dst[index].vz = src[index].vz;
		for(int n = 0;n < c_data.direct_num;n++){
			dst[index].a[n] = src[index].a[n];
		}
	}
}

__global__ void lbmKernel(LBMPoint* now,LBMPoint* next){
	int max_threadIdx_par_block = blockDim.x * gridDim.x;
	int threadIdx_in_block = threadIdx.x + blockIdx.x * blockDim.x;

	int max_roop = ((c_data.size_in - threadIdx_in_block - 1) / (double)max_threadIdx_par_block) + 1;

	//printf("lbmKernel t:%2d | d:%2d | mr:%2d | dn:%2d | csi:%2d | mtpd:%2d | tib:%2d\n",threadIdx.x,blockIdx.x,max_roop,c_data.direct_num,c_data.size_in,max_threadIdx_par_block,threadIdx_in_block);

	double pa = 0;
	double peq = 0;
	double density = 0; //���x�̃L���b�V��
	double vx = 0,vy = 0,vz = 0;	//���x�̃L���b�V��

	for(int roop = 0;roop < max_roop;roop++){
		int index_in = threadIdx_in_block + max_threadIdx_par_block * roop;
		int index_out = c_data.offset 
			+ index_in%c_data.x_max_in
			+ (index_in/c_data.x_max_in)*c_data.x_max_out;
		index_out += index_in/(c_data.x_max_in*c_data.y_max_in)*c_data.x_max_out*c_data.y_max_out;
		
		//�O�X��̃X�e�b�v�̒l���c���Ă���̂ŏ�����
		/*next[index_out].density = 0;
		next[index_out].vx = 0;
		next[index_out].vy = 0;
		next[index_out].vz = 0;*/
		/*if(now[index_out].mask == BOUNDARY){
			break;
		}*/
		double boundary = (now[index_out].mask != BOUNDARY);

		for(int a = 0;a < c_data.direct_num;a++){
			int index_a = index_out - ex[a] - c_data.x_max_out * ey[a] - c_data.x_max_out * c_data.y_max_out * ez[a];
			
			calcPeq(&peq,&now[index_a],a);
			calcPa(&pa,peq,&now[index_a],a);
			int bound_pa = calcBoundPa(now,index_out,a);

			//�L���b�V������
			int mask = now[index_a].mask;
			int normal = mask == NORMAL;
			int bound = mask == BOUNDARY;
			int inflow = mask == INFLOW;

			int outflow = mask == OUTFLOW;
		
			//�L���b�V������
			pa = normal * pa + bound * bound_pa + inflow * c_data.inflow_a[a] + outflow * now[index_out].a[a];

			next[index_out].a[a] = pa;

			//printf("lbmKernel roop t:%2d | d:%2d | r:%2d | io:%2d | in:%2d | a:%2d | ia(out):%2d | ia:%2d | pa:%f |msk:%d | n:%d(%d) | b:%d(%d) | i:%d(%d) | o:%d(%d)\n",threadIdx.x,blockIdx.x,max_roop,index_out,index_in,a,index_a,index_a,pa,mask,normal,NORMAL,bound,BOUNDARY,inflow,INFLOW,outflow,OUTFLOW);
			//printf("lbmKernel ia:%2d | pa:%f |msk:%d | n:%d(%d) | b:%d(%d) | i:%d(%d) | o:%d(%d)\n",index_a,pa,mask,normal,NORMAL,bound,BOUNDARY,inflow,INFLOW,outflow,OUTFLOW);

			density += pa;

			//�ő̂̒���������0
			vx += ex[a] * c_data.c * pa * boundary;
			vy += ey[a] * c_data.c * pa * boundary;
			vz += ez[a] * c_data.c * pa * boundary;
		}

		//���x���������x���v�Z
		//���x��0�������ꍇ�i�ő̂̒��j���ӂ݂āA���̎���1�ɒ���
		density = density + (density == 0) * 1.0;
		next[index_out].density = density * boundary;
		next[index_out].vx = vx / density;
		next[index_out].vy = vy / density;
		next[index_out].vz = vz / density;

		/*printf("lbm kernel t:%2d | b:%2d | i:%2d | de:%g | vx:%g | vy:%g | vx:%g | mask:%2d | a[0]:%g | a[1]:%g | a[2]:%g | a[3]:%g | a[4]:%g | a[5]:%g | a[6]:%g | a[7]:%g | a[8]:%g \n\n"
			,threadIdx.x,blockIdx.x,index_out,next[index_out].density,next[index_out].vx,next[index_out].vy,next[index_out].vz,next[index_out].mask
			,next[index_out].a[0],next[index_out].a[1],next[index_out].a[2],next[index_out].a[3],next[index_out].a[4],next[index_out].a[5],next[index_out].a[6],next[index_out].a[7],next[index_out].a[8]);
		*/
	}
}

__global__ void lbmResultKernel(LBMPoint* points,LBMResult* result){
	int max_threadIdx_par_block = blockDim.x * gridDim.x;
	int threadIdx_in_block = threadIdx.x + blockIdx.x * blockDim.x;

	int max_roop = (c_data.size_out - threadIdx_in_block - 1) / (double)max_threadIdx_par_block + 1;

	for(int roop = 0;roop < max_roop;roop++){
		int index = threadIdx_in_block + max_threadIdx_par_block * roop;

		//printf("lbmResultKernel io:%2d | vx:%f | vy:%f | vz:%f | de:%f\n",index,points[index].vx,points[index].vy,points[index].vz,points[index].density);

		result[index].vx = points[index].vx;
		result[index].vy = points[index].vy;
		result[index].vz = points[index].vz;
		result[index].density = points[index].density;

	}

}

void h_calcPeq(double* peq,double density,double* ex,double* ey,double* ez,double* w,double vx,double vy,double vz,int a){
	double e_dot_v = ex[a] * vx + ey[a] * vy + ez[a] * vz;
	double v_dot_v = vx * vx + vy * vy + vz * vz;
	(*peq) = w[a] * density * (1.0
		+ ( 3.0 * e_dot_v )
		+ ( 9.0 / 2.0 ) * ( e_dot_v * e_dot_v )
		- ( 3.0 / 2.0 ) * v_dot_v );
}

void printLBMData(LBMData* data){
	printf("/// Lattice Info ///\n");
	printf("direct_num : %d\n",data->direct_num);
	printf("max_speed : %d\n",data->max_speed);
	printf("x_max_out : %d\n",data->x_max_out);
	printf("y_max_out : %d\n",data->y_max_out);
	printf("z_max_out : %d\n",data->z_max_out);
	printf("x_max_in : %d\n",data->x_max_in);
	printf("y_max_in : %d\n",data->y_max_in);
	printf("z_max_in : %d\n",data->z_max_in);
	printf("size_in : %d\n",data->size_in);
	printf("size_out : %d\n",data->size_out);
	printf("offset : %d\n",data->offset);
	printf("////////////////////\n");
}

void printLBMConf(LBMConfig* conf){
	printf("/// Configulation ///\n");
	printf("�i�q�_���F(%3d,%3d,%3d)\n",conf->x,conf->y,conf->z);
	printf("���x�x�N�g�����F%2d\n",conf->directionNum);
	printf("��ԍ��݁F%f\n",conf->deltaLength);
	printf("���ԍ��݁F%f\n",conf->deltaTime);
	printf("��\�i�q�_���F%2d\n",conf->lattice_num);
	printf("�������x�F%f\n",conf->density);
	printf("�������x�F(%f,%f,%f)\n",conf->vx,conf->vy,conf->vz);
	printf("��\�����F%f\n",conf->cld);
	printf("��\���x�F%f\n",conf->cv);
	printf("���ώ��R�H���F%f\n",conf->lambda);
	printf("���q�ʁF%f\n",conf->M);
	printf("���x�F%f\n",conf->T);
	printf("�S���W���F%f\n",conf->mu);
	printf("�����C�̒萔�F%f\n",conf->R);
	printf("/////////////////////\n");
}

bool safery(cudaError_t ce,char str[]){
	if(ce == cudaSuccess){
		printf("%s : %s\n",str,cudaGetErrorString(ce));
		return false;
	}
	printf("error : %s : %s\n",cudaGetErrorString(ce),str);
	return true;
}

void preparation(LBMConfig& conf,LBMInfo& info){
	//�W��������
	info.w = new double[DIRECTION];	//9���x���f��
	info.w[0] = 4.0 / 9.0;
	info.w[1] = 1.0 / 9.0;
	info.w[2] = info.w[1];
	info.w[3] = info.w[1];
	info.w[4] = info.w[1];
	info.w[5] = 1.0 / 36.0;
	info.w[6] = info.w[5];
	info.w[7] = info.w[5];
	info.w[8] = info.w[5];

	//�����x�N�g��������
	info.ex = new double[DIRECTION];	//9���x���f��
	info.ey = new double[DIRECTION];	//9���x���f��
	info.ez = new double[DIRECTION];	//9���x���f��

	info.ex[0] = 0; info.ey[0] = 0; info.ez[0] = 0;
	info.ex[1] = 1; info.ey[1] = 0; info.ez[1] = 0;
	info.ex[2] = 0; info.ey[2] = 1; info.ez[2] = 0;
	info.ex[3] = -1; info.ey[3] = 0; info.ez[3] = 0;
	info.ex[4] = 0; info.ey[4] = -1; info.ez[4] = 0;
	info.ex[5] = 1; info.ey[5] = 1; info.ez[5] = 0;
	info.ex[6] = -1; info.ey[6] = 1; info.ez[6] = 0;
	info.ex[7] = -1; info.ey[7] = -1; info.ez[7] = 0;
	info.ex[8] = 1; info.ey[8] = -1; info.ez[8] = 0;

	info.x = conf.x;
	info.y = conf.y;
	info.z = conf.z;

	info.directionNum = DIRECTION;
	info.deltaTime = conf.deltaTime;
	info.deltaLength = conf.deltaLength;
	info.density = (101325 * conf.M)/(conf.R * conf.T);
	info.cld = conf.cld;
	info.cv = conf.cv;
	info.lambda = conf.lambda;

	info.vx = conf.vx;
	info.vy = conf.vy;
	info.vz = conf.vz;
}

void initLBMData(LBMConfig& conf,LBMInfo& info,LBMData& data){
	data.c = info.deltaLength / info.deltaTime;
	data.tau = 3.0*(conf.mu / info.density)*(info.deltaTime / (info.deltaLength*info.deltaLength)) + 0.5;
	data.direct_num = info.directionNum;
	data.max_speed = 1;
	data.x_max_out = info.x;
	data.y_max_out = info.y;
	data.z_max_out = info.z;
	data.x_max_in = Math::max(data.x_max_out - data.max_speed * 2,1);
	data.y_max_in = Math::max(data.y_max_out - data.max_speed * 2,1);
	data.z_max_in = Math::max(data.z_max_out - data.max_speed * 2,1);
	data.size_in = data.x_max_in * data.y_max_in * data.z_max_in;
	data.size_out = data.x_max_out * data.y_max_out * data.z_max_out;
	data.density = info.density;
	data.vx = info.vx;
	data.vy = info.vy;
	data.vz = info.vz;
	if(data.z_max_out - data.max_speed <= 0){
		//2D�̏ꍇ
		data.offset = data.max_speed * data.x_max_out + data.max_speed;
	}else{
		//3D�̏ꍇ
		data.offset = data.x_max_out * data.y_max_out + data.max_speed * data.x_max_out + data.max_speed;
	}
		//�����̕��z�֐��l��������
	for(int n = 0;n < DIRECTION;n++){
		h_calcPeq(&data.inflow_a[n],data.density,info.ex,info.ey,info.ez,info.w,data.vx,data.vy,data.vz,n);
	}
}

void initConstant(LBMData* data,LBMInfo* info){

	//�W�����R�s�[
	cudaMemcpyToSymbol(w,info->w,sizeof(double)*DIRECTION);
	if( safery(cudaGetLastError(),"cuda constant memcpy w") ) return;

	//�����x�N�g�����R�s�[
	cudaMemcpyToSymbol(ex,info->ex,sizeof(double)*DIRECTION);
	if( safery(cudaGetLastError(),"cuda constant memcpy ex") ) return;
	cudaMemcpyToSymbol(ey,info->ey,sizeof(double)*DIRECTION);
	if( safery(cudaGetLastError(),"cuda constant memcpy ey") ) return;
	cudaMemcpyToSymbol(ez,info->ez,sizeof(double)*DIRECTION);
	if( safery(cudaGetLastError(),"cuda constant memcpy ez") ) return;

	//Bounce-Back�����p�̔��]�e�[�u�����R�s�[
	int h_inv_table[DIRECTION];
	h_inv_table[0] = 0;
	h_inv_table[1] = 3;
	h_inv_table[2] = 4;
	h_inv_table[3] = 1;
	h_inv_table[4] = 2;
	h_inv_table[5] = 7;
	h_inv_table[6] = 8;
	h_inv_table[7] = 5;
	h_inv_table[8] = 6;
	cudaMemcpyToSymbol(invtert_table,&h_inv_table,sizeof(int)*DIRECTION);
	if( safery(cudaGetLastError(),"cuda constant memcpy invert_table") ) return;

	//LBMData�̏����R���X�^���ƃ������ɃR�s�[
	cudaMemcpyToSymbol(c_data,data,sizeof(LBMData));
	if( safery(cudaGetLastError(),"cuda constant memcpy LBMData") ) return;

	delete[] info->w; info->w = nullptr;
	delete[] info->ex; info->ex = nullptr;
	delete[] info->ey; info->ey = nullptr;
	delete[] info->ez; info->ez = nullptr;
}

void writeData(LBMResult* result,LBMData data,LBMInfo info,int num){
	std::string fName = "test_"+ std::to_string(num)+".dat";
	std::ofstream ofs(fName);

	int writeRate = 1;

	for(int m = 0;m < data.y_max_out;m += writeRate){
		for(int n = 0;n < data.x_max_out;n += writeRate){
			int index = n + m * data.x_max_out;
			double _x = n*info.deltaLength;
			double _y = m*info.deltaLength;
			ofs << _x << " " << _y << " " << result[index].vx << " " << result[index].vy << std::endl;
		}
		ofs << std::endl;
	}

}

void main(){
	LBMConfig l_conf;

	l_conf.x = 256;
	l_conf.y = 256;
	l_conf.z = 1;
	l_conf.density = 1.2041516978905051;
	l_conf.vx = 0.2;
	l_conf.vy = 0;
	l_conf.vz = 0;
	l_conf.directionNum = DIRECTION;
	l_conf.cld = 0.01;	//1[m]
	l_conf.cv = l_conf.vx;	//1 [m/s]
	l_conf.lattice_num = 16;
	l_conf.deltaLength = l_conf.cld / (double)l_conf.lattice_num;
	l_conf.deltaTime = l_conf.deltaLength;
	l_conf.lambda = 68e-9;
	l_conf.M = 0.028966;
	l_conf.R = 8.3144598;
	l_conf.T = 293.15;
	l_conf.mu = 1.83e-5;

	LBMInfo l_info;
	preparation(l_conf,l_info);

	LBMData l_data;	//�J�[�l���ł̋��L�f�[�^(Constant�������Ɋi�[)
	initLBMData(l_conf,l_info,l_data);	//LBM�̏�񂩂�f�[�^��������

	printLBMConf(&l_conf);	//LBMConfig���o��
	printLBMData(&l_data);	//LBMData���o��
	
	initConstant(&l_data,&l_info);	//�R���X�^���g�������ɋ��L�f�[�^��]��

	//�f�[�^�錾
	LBMPoint* d_p = nullptr;		//�J�[�l�����̊i�q�_�f�[�^�z��
	double* d_pa = nullptr;	//�J�[�l�����̊i�q�_���x�z��
	LBMPoint* n_d_p = nullptr;		//�J�[�l�����̊i�q�_�f�[�^�z��
	double* n_d_pa = nullptr;	//�J�[�l�����̊i�q�_���x�z��

	//�f�[�^�̗p��(now)
	cudaMalloc((void**)&d_p,l_info.x * l_info.y * l_info.z * sizeof(LBMPoint));
	if( safery(cudaGetLastError(),"cuda malloc d_p") ) goto cudaAllFree;
	cudaMalloc((void**)&d_pa,l_info.x * l_info.y * l_info.z * DIRECTION * sizeof(double));
	if( safery(cudaGetLastError(),"cuda malloc d_pa") ) goto cudaAllFree;

	//�f�[�^�̗p��(next)
	cudaMalloc((void**)&n_d_p,l_info.x * l_info.y * l_info.z * sizeof(LBMPoint));
	if( safery(cudaGetLastError(),"cuda malloc n_d_p") ) goto cudaAllFree;
	cudaMalloc((void**)&n_d_pa,l_info.x * l_info.y * l_info.z * DIRECTION * sizeof(double));
	if( safery(cudaGetLastError(),"cuda malloc n_d_pa") ) goto cudaAllFree;

	//�������p�X���b�h���ƃu���b�N���̎Z�o
	dim3 init_block(256);
	dim3 init_thread(256);

	//�������p�J�[�l���Ăяo��(now)
	initKernel<<<init_block,init_thread>>>(d_p,d_pa);
	//testKernel<<<1,1>>>();
	cudaThreadSynchronize();
	if( safery(cudaGetLastError(),"cuda kernel initKernel") ) goto cudaAllFree;

	///�������p�J�[�l���Ăяo��(next)�B���̌�now���R�s�[����
	initKernel<<<init_block,init_thread>>>(n_d_p,n_d_pa);
	cudaThreadSynchronize();
	if( safery(cudaGetLastError(),"cuda kernel initKernel") ) goto cudaAllFree;
	//testKernel<<<1,1>>>();
	lbmCopyKernel<<<init_block,init_thread>>>(n_d_p,d_p);
	cudaThreadSynchronize();
	if( safery(cudaGetLastError(),"cuda kernel lbmCopyKernel") ) goto cudaAllFree;

	cudaThreadSynchronize();

	printf("\n--------------------------------------------lbmKernel-------------------------------------------------------\n");

	//���ʕۑ��p�i�f�o�C�X�j
	LBMResult* d_result = nullptr;
	cudaMalloc((void**)&d_result,sizeof(LBMResult)*l_data.x_max_out*l_data.y_max_out*l_data.z_max_out);
	if( safery(cudaGetLastError(),"cuda malloc") ) goto cudaAllFree;

	//���ʕۑ��p�i�z�X�g�j
	LBMResult* h_result = new LBMResult[l_data.x_max_out*l_data.y_max_out*l_data.z_max_out];

	//�v�Z�p�X���b�h���ƃu���b�N���̎Z�o
	dim3 block(256);
	dim3 thread(256);

	int step_roop = 10;
	int write_rate = 1;	//�S�X�e�b�v�ŋL�^���邩
	for(int n = 0;n < step_roop;n++){

		//�v�Z�J�n
		lbmKernel<<<block,thread>>>(d_p,n_d_p);
		//if( safery(cudaGetLastError(),"cuda kernel lbmKernel") ) goto cudaAllFree;
		cudaThreadSynchronize();
		//if( safery(cudaGetLastError(),"cuda kernel lbmKernel") ) goto cudaAllFree;
		LBMPoint* t = d_p;
		d_p = n_d_p;
		n_d_p = t;

		if(n % write_rate == 0){
			//���ʂ��R�s�[
			//���x�Ɩ��x�����𕪗��i�J�[�l���ŕ�����s�j
			lbmResultKernel<<<block,thread>>>(d_p,d_result);
			cudaThreadSynchronize();
			if( safery(cudaGetLastError(),"cuda result kernel") ) goto cudaAllFree;

			//�z�X�g���������ɃR�s�[
			cudaMemcpy(h_result,d_result,sizeof(LBMResult)*l_data.x_max_out*l_data.y_max_out*l_data.z_max_out,cudaMemcpyDeviceToHost);
			if( safery(cudaGetLastError(),"cuda memcpy") ) goto cudaAllFree;

			//�t�@�C���ɏ�������
			writeData(h_result,l_data,l_info,n);
		}
	}

	//�v��Ȃ��̂�����
	if(n_d_p != nullptr)	cudaFree(n_d_p); n_d_p = nullptr;
	if(n_d_pa != nullptr)	cudaFree(n_d_pa); n_d_pa = nullptr;

	//���ʂ��R�s�[
	//���x�Ɩ��x�����𕪗��i�J�[�l���ŕ�����s�j
	lbmResultKernel<<<block,thread>>>(d_p,d_result);
	cudaThreadSynchronize();
	if( safery(cudaGetLastError(),"cuda result kernel") ) goto cudaAllFree;

	//�z�X�g���������ɃR�s�[
	cudaMemcpy(h_result,d_result,sizeof(LBMResult)*l_data.x_max_out*l_data.y_max_out*l_data.z_max_out,cudaMemcpyDeviceToHost);
	if( safery(cudaGetLastError(),"cuda memcpy") ) goto cudaAllFree;

	//�t�@�C���ɏ�������
	writeData(h_result,l_data,l_info,step_roop);

cudaAllFree:
	cudaThreadSynchronize();
	if( safery(cudaGetLastError(),"cuda kernel lbmKernel") ) goto cudaAllFree;

	if(d_p != nullptr)	cudaFree(d_p);
	if(d_pa != nullptr)	cudaFree(d_pa);
	if(n_d_p != nullptr)	cudaFree(n_d_p);
	if(n_d_pa != nullptr)	cudaFree(n_d_pa);
	if(d_result != nullptr) cudaFree(d_result);

	delete[] h_result;

	cudaDeviceReset();

	printf("END\n");
	return;
}
