
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define PI 3.1415926

typedef unsigned char uchar;

/*
__device__ float* calcHaversineDist(float* geo_trans, int2 point, int nrow, int ncol)
{
	float dlat = geo_trans[1] * PI / 180;
	float dlon = geo_trans[5] * PI / 180;
	float p1_lonlat_y = (geo_trans[3] + point.y * geo_trans[5]) * PI / 180;
	float p2_lonlat_y = (geo_trans[3] + (point.y + 1) * geo_trans[5]) * PI / 180;
	float a_dlat = sinf(dlat / 2) * sinf(dlat / 2) + cosf(p1_lonlat_y) * cosf(p2_lonlat_y) * sinf(0.0 / 2) * sinf(0.0 / 2);
	float a_dlon = sinf(0.0 / 2) * sinf(0.0 / 2) + cosf(p1_lonlat_y) * cosf(p2_lonlat_y) * sinf(dlon / 2) * sinf(dlon / 2);
	float c_dlat = atan2f(sqrtf(a_dlat), sqrtf(1 - a_dlat));
	float c_dlon = atan2f(sqrtf(a_dlon), sqrtf(1 - a_dlon));
	float pix_dists[2] = {0.0};
	pix_dists[0] = 6371 * c_dlat;
	pix_dists[1] = 6371 * c_dlon;
	return pix_dists;
}
*/

__device__ bool updPosition(int2& point, float& flowdist, int2* dirvecs, float* catchdist, uchar* FD, float* DX, float* DY, float default_dist_value, int nrow, int ncol)
{
	bool bool_catch = false;
	int point_idx = point.y * ncol + point.x;
	if (fabsf(catchdist[point_idx]- default_dist_value)<1e-3) {
		//here this point is already in the catchment outlet
		bool_catch = true;
	}
	else {
		uchar FD_value = FD[point_idx];
		uchar vec_idx = 0;
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

// catchdist == default_dist_value (radius_pix * DEFAULT_PIX_LEN): the outlet of catchment (should be pre-defined in Python code); 
__global__ void calcFlowDist(float* catchdist, uchar* FD, float* DX, float* DY, float default_dist_value, int max_trace_steps, int nrow, int ncol)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	if (x >= ncol || y >= nrow)
		return;
	//set up flow direction vectors & distance vectors
	int2 dirvecs[9];
	//case in HydroSHEDS v1.0
	dirvecs[0].y = 0; dirvecs[0].x = 0; dirvecs[1].y = 0; dirvecs[1].x = 1; dirvecs[2].y = 1; dirvecs[2].x = 1; dirvecs[3].y = 1; dirvecs[3].x = 0;
	dirvecs[4].y = 1; dirvecs[4].x = -1; dirvecs[5].y = 0; dirvecs[5].x = -1; dirvecs[6].y = -1; dirvecs[6].x = -1; dirvecs[7].y = -1; dirvecs[7].x = 0;
	dirvecs[8].y = -1; dirvecs[8].x = 1;

	//set up the intial conditions
	int2 point, point_prev;
	point.x = x;	point.y = y;
	point_prev = point;
	int idx = y * ncol + x;
	float flowdist = default_dist_value;
	bool bool_catch = false;


	for (int i = 0; i < max_trace_steps; i++) {
		//to track water flow of the point and to check whether it is in the catchment
		bool_catch = updPosition(point, flowdist, dirvecs, catchdist, FD, DX, DY, default_dist_value, nrow, ncol);
		if (bool_catch && fabsf(catchdist[idx] - default_dist_value) >= 1e-3) {
			//we can only update the non-outlet pixels
			catchdist[idx] = flowdist;
			break;
		}
		//if the flow point is repeating (reached a local minimum)
		else if (point.x == point_prev.x && point.y == point_prev.y) {
			break;
		}
		point_prev = point;
	}
}
