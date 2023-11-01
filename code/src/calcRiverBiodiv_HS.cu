
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cstdio>

#define MAX_LOOP 5000
#define MAX_VEC_LEN 10
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define INDEX2D_YX(ncol, idy, idx) ((idy) * (ncol) + (idx))

#define DEFAULT_PIX_LEN 0.09
#define PI 3.1415926

typedef unsigned char uchar;

__device__ int findIndex(int value, const int* arr, int num) {
	int index = -1;
#pragma unroll
	for (int i = 0; i < num; i++) {
		if (arr[i] == value) {
			index = i;
			break;
		}
	}
	return index;
}

__device__ bool updPosition(int2& point, float& flowdist, int2* dirvecs, int2 dst_point, const uchar* FD, const float* DX, const float* DY,
	int2 i_start, int2 i_end, const int nrow, const int ncol)
{
	bool bool_catch = false;
	if (point.x == dst_point.x && point.y == dst_point.y) {
		//here this point is already in the dst point
		bool_catch = true;
	}
	else {
		int point_index = INDEX2D_YX(ncol, point.y, point.x);
		uchar FD_value = FD[point_index];
		int vec_idx = 0;
		if (FD_value == 0) {
			vec_idx = 0;
		}
		else {
			vec_idx = (int)log2f((float)FD_value) + 1;
		}
		//to update the point with current flow direction, then to fix out-of-boundary problem
		int2 point_upd, point_diff;
		float2 point_d;
		//point_upd.x = point.x + tex1D(dirvecs_txt, vec_idx * 2 + 1);
		point_upd.x = point.x + dirvecs[vec_idx].x;
		point_upd.x = MAX(i_start.x, point_upd.x);
		point_upd.x = MIN(point_upd.x, i_end.x - 1);
		//point_upd.y = point.y + tex1D(dirvecs_txt, vec_idx * 2);
		point_upd.y = point.y + dirvecs[vec_idx].y;
		point_upd.y = MAX(i_start.y, point_upd.y);
		point_upd.y = MIN(point_upd.y, i_end.y - 1);

		point_diff.x = point_upd.x - point.x;
		point_diff.y = point_upd.y - point.y;
		point_d.x = DX[point.y];
		point_d.y = DY[point.x];
		flowdist += sqrtf((point_diff.y * point_d.y) * (point_diff.y * point_d.y) + (point_diff.x * point_d.x) * (point_diff.x * point_d.x));
		point = point_upd;
	}
	return bool_catch;
}

__device__ float calcLocalCatchFlowDist(int2 point, int2 dst_point, const uchar* FD, const float* DX, const float* DY,
	int2* dirvecs, int2 i_start, int2 i_end, const int nrow, const int ncol)
{
	//set up the intial conditions
	int2 point_prev, point_now;
	point_prev = point; point_now = point;
	float flowdist = (float)DEFAULT_PIX_LEN;  //default value
	float result = -1.0;
	bool bool_catch = false;
	//to track water flow of the point and to check whether it is in the catchment
	for (int i = 0; i < MAX_LOOP; i++) {
		bool_catch = updPosition(point_now, flowdist, dirvecs, dst_point, FD, DX, DY, i_start, i_end, nrow, ncol);
		if (bool_catch) {
			result = flowdist;
			break;
		}
		//if the flow point is repeating (reached a local minimum)
		else if (point_now.x == point_prev.x && point_now.y == point_prev.y) {
			break;
		}
		point_prev = point_now;
	}
	return result;
}

__device__ float calcContribBiodiv(int2 point, float flowdist, const uchar* LULC, const float* DX, const float* DY, const int decay_func, const float* param_vec,
	const float* LC_values, const int* LC_types, const int num_LC_types, const int nrow, const int ncol) {
	float contrib = 0.0;
	int point_index = INDEX2D_YX(ncol, point.y, point.x);
	int LC_type = (int)LULC[point_index];
	float pixel_area = DX[point.y] * DY[point.x];
	int i_type = findIndex(LC_type, LC_types, num_LC_types);

	if (i_type != -1) {
		//decay function: exponential (exp)
		if (decay_func == 1) {
			float alpha = 3 / param_vec[i_type];
			contrib = alpha * pixel_area * LC_values[i_type] * expf(-1 * alpha * flowdist);
		}
		//decay function: spherical (sph)
		else if (decay_func == 2) {
			float theta1 = param_vec[i_type];
			float theta2 = (8.0 / 3 / theta1) * pixel_area;
			if (flowdist <= theta1) {
				contrib = theta2 * LC_values[i_type] * (1 - 1.5 * (flowdist / theta1) + 0.5 * (flowdist / theta1) * (flowdist / theta1) * (flowdist / theta1));
			}
		}
		//decay function: matern15
		else if (decay_func == 3) {
			float alpha = 3 / param_vec[i_type];
			float beta = sqrtf(3) * alpha / 2;
			contrib = pixel_area * beta * (1 + sqrtf(3) * alpha * flowdist) * expf(-1 * sqrtf(3) * alpha * flowdist);
		}
		//decay function: matern25
		else if (decay_func == 4) {
			float alpha = 3 / param_vec[i_type];
			float beta = 3 * sqrtf(5) * alpha / 8;
			contrib = pixel_area * beta * (1 + sqrtf(5) * alpha * flowdist + 5.0 / 3 * alpha * alpha * flowdist * flowdist) *
				expf(-1 * sqrtf(5) * alpha * flowdist);
		}
		//decay function: gaussian
		else if (decay_func == 5) {
			float alpha = 3 / param_vec[i_type];
			float beta = 2 * sqrtf(alpha * alpha / PI / 3);
			contrib = pixel_area * beta * expf(-1 * alpha * alpha * flowdist * flowdist);
		}
	}
	return contrib;
}

//decay_func = 1: exp; decay_func = 2: sph.
__global__ void calcRiverBiodivEnv(float* B, const int* FA, const uchar*  FD, const uchar* LULC, const float* DX, const float* DY,
	const float* param_vec, const float* LC_values, const int* LC_types,	const int decay_func, const int num_LC_types, 
	const float FA_thres, const int radius_pix, const int nrow, const int ncol)
{
	int2 dst_point, i_point;
	dst_point.x = threadIdx.x + blockDim.x * blockIdx.x;
	dst_point.y = threadIdx.y + blockDim.y * blockIdx.y;
	int dst_index = INDEX2D_YX(ncol, dst_point.y, dst_point.x);
	if (dst_point.x >= ncol || dst_point.y >= nrow || FA[dst_index] < FA_thres)
		return;
	else {
		//set up flow direction vectors & distance vectors
		int2 dirvecs[9];
		//case in HydroSHEDS v1.0
		dirvecs[0].y = 0; dirvecs[0].x = 0; dirvecs[1].y = 0; dirvecs[1].x = 1; dirvecs[2].y = 1; dirvecs[2].x = 1; dirvecs[3].y = 1; dirvecs[3].x = 0;
		dirvecs[4].y = 1; dirvecs[4].x = -1; dirvecs[5].y = 0; dirvecs[5].x = -1; dirvecs[6].y = -1; dirvecs[6].x = -1; dirvecs[7].y = -1; dirvecs[7].x = 0;
		dirvecs[8].y = -1; dirvecs[8].x = 1;

		float biodiv = 0.0, flowdist = 0.0;
		//each river pixel: to search for local catchment 
		int2 i_start, i_end;
		i_start.y = MAX(dst_point.y - radius_pix, 0);
		i_end.y = MIN(dst_point.y + radius_pix + 1, nrow);
		i_start.x = MAX(dst_point.x - radius_pix, 0);
		i_end.x = MIN(dst_point.x + radius_pix + 1, ncol);

		for (i_point.y = i_start.y; i_point.y < i_end.y; i_point.y++) {
			for (i_point.x = i_start.x; i_point.x < i_end.x; i_point.x++) {
				if (((i_point.y - dst_point.y) * (i_point.y - dst_point.y) + (i_point.x - dst_point.x) * (i_point.x - dst_point.x)) > (radius_pix * radius_pix))
					continue;
				else {
					flowdist = calcLocalCatchFlowDist(i_point, dst_point, FD, DX, DY, dirvecs, i_start, i_end, nrow, ncol);
					if (abs(flowdist + 1) < 1e-6)  //not in the catchment
						continue;
					else {
						biodiv += calcContribBiodiv(i_point, flowdist, LULC, DX, DY, decay_func, param_vec, LC_values, LC_types, num_LC_types, nrow, ncol);
					}
				}
			}
		}
		//here to calculate the baseline biodiversity value; B[xxx] is filled with river discharge (Q) data.
		biodiv += LC_values[num_LC_types] + LC_values[num_LC_types + 1] * logf(B[dst_index] + 1e-3);
		B[dst_index] = biodiv;
	}
}

