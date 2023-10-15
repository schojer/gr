/*  
                            ***********************
								  by Jie SHAO
                       The Hong Kong Polytechnic University
                         e-mail: jie.shao@polyu.edu.hk
                            ***********************
*/

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <Windows.h>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace pcl;
using namespace cv;

#define GRIDSIZE 3.0	// setting grid size for calculating roof area

// calculating plane area
float computerArea(PointCloud<PointXYZ>::Ptr roof)
{
	// projecting roof points onto horizontal plane
	PointXYZ min, max;
	getMinMax3D(*roof, min, max);
	int rows = (max.x - min.x) / GRIDSIZE + 1;
	int cols = (max.y - min.y) / GRIDSIZE + 1;

	Mat img = Mat(rows, cols, CV_8UC1, Scalar(255));
	for (int i = 0; i < roof->size(); i++)
	{
		int u = (roof->points[i].x - min.x) / GRIDSIZE;
		int v = (roof->points[i].y - min.y) / GRIDSIZE;
		img.at<uchar>(u, v) = 66;
	}

	// median smooth
	Mat m_img;
	medianBlur(img, m_img, 3);

	// counting the number of effective pixel points
	int number = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (m_img.at<uchar>(i, j) < 255)
				number += 1;
		}
	}
	// calculating area
	float area = number * GRIDSIZE * GRIDSIZE;

	return area;
}

// clustering each roof points and calculating roof area
void EuclideanCluster(PointCloud<PointXYZ>::Ptr input, PointCloud<PointXYZRGB>::Ptr green_area)
{
	pcl::search::KdTree<PointXYZ>::Ptr tree(new pcl::search::KdTree<PointXYZ>);
	tree->setInputCloud(input);
	vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<PointXYZ> ec;
	ec.setClusterTolerance(1.5);
	ec.setMinClusterSize(10);
	ec.setMaxClusterSize(999999);
	ec.setSearchMethod(tree);
	ec.setInputCloud(input);
	ec.extract(cluster_indices);

	PointCloud<PointXYZ>::Ptr area_cloud(new PointCloud<PointXYZ>());
	float g_area = 0;
	float r_area = 0;
	for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{

		for (vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
		{
			PointXYZ pt;
			pt.x = input->points[*pit].x;
			pt.y = input->points[*pit].y;
			pt.z = input->points[*pit].z;
			area_cloud->push_back(pt);
		}

		// calculating roof area
		float area = computerArea(area_cloud);
		if (area >= 10)
		{
			PointXYZRGB pt;
			pt.r = rand() % 255;
			pt.g = rand() % 255;
			pt.b = rand() % 255;
			for (int i = 0; i < area_cloud->size(); i++)
			{
				pt.x = area_cloud->points[i].x;
				pt.y = area_cloud->points[i].y;
				pt.z = area_cloud->points[i].z;
				green_area->push_back(pt);
			}
			g_area += area;
		}
		area_cloud->clear();
	}
}

// fitting roof plane and calculating roof slope
float PlaneFittingandSlope(PointCloud<PointXYZ>::Ptr input)
{
	// Normalization processing
	PointXYZ min, max;
	getMinMax3D(*input, min, max);
	PointCloud<PointXYZ>::Ptr segs(new PointCloud<PointXYZ>());
	for (int i = 0; i < input->size(); i++)
	{
		max.x = input->points[i].x - min.x;
		max.y = input->points[i].y - min.y;
		max.z = input->points[i].z - min.z;
		segs->push_back(max);
	}

	Vec4f coeff;
	SACSegmentation<PointXYZ> sac;
	ModelCoefficients::Ptr coefficients(new ModelCoefficients);
	PointIndices::Ptr inliers(new PointIndices);

	sac.setInputCloud(segs);
	sac.setMethodType(SAC_RANSAC);
	sac.setModelType(SACMODEL_PLANE);
	sac.setDistanceThreshold(0.5);
	sac.segment(*inliers, *coefficients);
	segs->clear();

	for (int i = 0; i < 4; i++)
		coeff[i] = coefficients->values[i];

	// calculating angle based on normal vector
	float angle = acos(coeff[2] / sqrt(coeff[0] * coeff[0] + coeff[1] * coeff[1] + coeff[2] * coeff[2]));
	angle = angle * 180.0 / M_PI;
	if (angle > 90)
		angle = 180 - angle;

	return angle;
}

// roof point cloud segmentation based on region growing
void RoofbySlope(PointCloud<PointXYZ>::Ptr temp_input, PointCloud<PointXYZ>::Ptr green_roofs)
{
	// Normalization processing
	PointCloud<PointXYZ>::Ptr input(new PointCloud<PointXYZ>());
	PointXYZ min, max;
	getMinMax3D(*temp_input, min, max);
	for (int i = 0; i < temp_input->size(); i++)
	{
		max.x = temp_input->points[i].x - min.x;
		max.y = temp_input->points[i].y - min.y;
		max.z = temp_input->points[i].z - min.z;
		input->push_back(max);
	}

	// setting search strategy
	pcl::search::Search<PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<PointXYZ> >(new pcl::search::KdTree<pcl::PointXYZ>);
	// calculating normal
	PointCloud<Normal>::Ptr normals(new PointCloud<Normal>);
	NormalEstimation<PointXYZ, Normal> normal_estimator;
	normal_estimator.setSearchMethod(tree);
	normal_estimator.setInputCloud(input);
	normal_estimator.setKSearch(50);
	normal_estimator.compute(*normals);
	pcl::IndicesPtr indices(new vector<int>);
	pcl::PassThrough<PointXYZ> pass;
	pass.setInputCloud(input);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0.0, 1.0);
	pass.filter(*indices);

	// clustering
	pcl::RegionGrowing<PointXYZ, Normal> reg;
	reg.setMinClusterSize(5);
	reg.setMaxClusterSize(999999);
	reg.setSearchMethod(tree);
	reg.setNumberOfNeighbours(30);
	reg.setInputCloud(input);
	reg.setInputNormals(normals);
	reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);	
	reg.setCurvatureThreshold(1);	

	vector<PointIndices> clusters;
	reg.extract(clusters);
	input->clear();
	normals->clear();

	for (vector<pcl::PointIndices>::const_iterator it = clusters.begin(); it != clusters.end(); ++it)
	{
		PointXYZRGB pt;
		pt.r = rand() % 255;
		pt.g = rand() % 255;
		pt.b = rand() % 255;

		// calculating slope of roof plane 
		PointCloud<PointXYZ>::Ptr roof_plane(new PointCloud<PointXYZ>());
		for (vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
		{
			pt.x = temp_input->points[*pit].x;
			pt.y = temp_input->points[*pit].y;
			pt.z = temp_input->points[*pit].z;
			PointXYZ seg;
			seg.x = pt.x;
			seg.y = pt.y;
			seg.z = pt.z;
			roof_plane->push_back(seg);
		}
		float angle = PlaneFittingandSlope(roof_plane);

		// determining candidate roofs through roof slope (15бу)
		if (angle < 15)
			*green_roofs += *roof_plane;

		roof_plane->clear();
	}
}

int main()
{
	cout << "Determining potential green roofs..." << endl;

	// determining potential roofs through roof slope and area
	string dirpath = "...\\fig1_figS2_roof_points\\raw_roofs\\";		// setting raw roof point cloud file path

	// detecting potential green roofs from each raw roof point cloud file in sequence
	for (int i = 1; i <= 42; i++)
	{
		// read raw roof point cloud
		string roof_path = dirpath + std::to_string(i) + ".pcd";
		PointCloud<PointXYZ>::Ptr roof_cloud(new PointCloud<PointXYZ>);
		pcl::io::loadPCDFile(roof_path, *roof_cloud);
		cout << roof_cloud->size() << endl;

		// determining candidate roofs through roof slope
		PointCloud<PointXYZ>::Ptr slope_green(new PointCloud<PointXYZ>());
		RoofbySlope(roof_cloud, slope_green);
		roof_cloud->clear();
		cout << "Slope green: " << slope_green->size() << endl;

		// basing on roof slope, determining candidate green roofs through roof area
		PointCloud<PointXYZRGB>::Ptr green_roofs(new PointCloud<PointXYZRGB>());
		EuclideanCluster(slope_green, green_roofs);
		slope_green->clear();

		// output potential green roofs
		string save_path = dirpath + std::to_string(i) + "_green_roofs.pcd";
		pcl::io::savePCDFileBinary(save_path, *green_roofs);
		green_roofs->clear();
	}

	return 0;
}