
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#define MAX_LOOP 60000
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

__device__ bool updPosition(int2& point, float& flowdist, int2* dirvecs, const uchar* FD, const float* DX, const float* DY, float max_dist, int nrow, int ncol)
{
	bool bool_catch = false;
	int point_idx = point.y * ncol + point.x;
	if (flowdist > max_dist) {
		//already reach the effective distance
		bool_catch = true;
	}
	else {
		uchar FD_value = FD[point_idx];
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
		point_upd.x = point.x + dirvecs[vec_idx].x;
		point_upd.x = MAX(0, point_upd.x);
		point_upd.x = MIN(point_upd.x, ncol - 1);
		point_upd.y = point.y + dirvecs[vec_idx].y;
		point_upd.y = MAX(0, point_upd.y);
		point_upd.y = MIN(point_upd.y, nrow - 1);

		point_diff.x = point_upd.x - point.x;
		point_diff.y = point_upd.y - point.y;
		point_d.x = DX[point.y];
		point_d.y = DY[point.x];
		flowdist += sqrtf((point_diff.y * point_d.y) * (point_diff.y * point_d.y) + (point_diff.x * point_d.x) * (point_diff.x * point_d.x));
		//flowdist += calcLonLatDist(point, point_upd, GeoTransform);
		point = point_upd;
	}
	return bool_catch;
}

__device__ float calcContribBiodiv(int2 point, float flowdist, const uchar* LULC, const float unit_area, const int decay_func, const float* param_vec,
	const float* LC_values, const int* LC_types, const int num_LC_types, const int nrow, const int ncol) {
	float contrib = 0.0;
	int point_index = INDEX2D_YX(ncol, point.y, point.x);
	int LC_type = (int)LULC[point_index];
	float pixel_area = unit_area;
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

__global__ void calcAccumLULCEffect(float* LULC_Effect, const int* FA, const uchar* FD, const uchar* LULC, const float* DX, const float* DY,
	const float* param_vec, const float* LC_values, const int* LC_types, const int decay_func, const int num_LC_types, const float FA_thres, const float max_dist, const float unit_area, const int nrow, const int ncol)

{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= ncol || y >= nrow)
		return;
	//set up flow direction vectors
	int2 dirvecs[9];
	//flow directions in HydroSHEDS
	dirvecs[0].y = 0; dirvecs[0].x = 0; dirvecs[1].y = 0; dirvecs[1].x = 1; dirvecs[2].y = 1; dirvecs[2].x = 1; dirvecs[3].y = 1; dirvecs[3].x = 0;
	dirvecs[4].y = 1; dirvecs[4].x = -1; dirvecs[5].y = 0; dirvecs[5].x = -1; dirvecs[6].y = -1; dirvecs[6].x = -1; dirvecs[7].y = -1; dirvecs[7].x = 0;
	dirvecs[8].y = -1; dirvecs[8].x = 1;

	//set up the intial conditions
	int2 point_idx, point, point_prev;
	point.x = x;	point.y = y;
	point_prev = point;
	point_idx = point;
	int idx = INDEX2D_YX(ncol, y, x);
	float flowdist, flowdist_prev = 0.0;
	float lulc_effect = 0.0;
	bool bool_catch = false;


	for (int i = 0; i < MAX_LOOP; i++) {
		//to track water flow of the point and to check whether it is within the effective distance.
		bool_catch = updPosition(point, flowdist, dirvecs, FD, DX, DY, max_dist, nrow, ncol);
		if (bool_catch) {
			//already reach the max effective range (max_dist)
			break;
		}
		else if (point.x == point_prev.x && point.y == point_prev.y) {
			break;
		}
		else if (FA[INDEX2D_YX(ncol, point.y, point.x)] >= FA_thres) {
			lulc_effect += calcContribBiodiv(point_idx, flowdist, LULC, unit_area, decay_func, param_vec, LC_values, LC_types, num_LC_types, nrow, ncol) * (flowdist - flowdist_prev);
		}
		point_prev = point;
		flowdist_prev = flowdist;
	}
	LULC_Effect[idx] = lulc_effect;
}
