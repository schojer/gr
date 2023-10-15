#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace pcl;

int main()
{
	string ground_path = "ground.pcd";
	PointCloud<PointXYZ>::Ptr grounds(new PointCloud<PointXYZ>());
	pcl::io::loadPCDFile(ground_path, *grounds);
	PointCloud<PointXYZ>::Ptr xy_grounds(new PointCloud<PointXYZ>());

	for (int i = 0; i < grounds->size(); i++)
	{
		PointXYZ pt;
		pt.x = grounds->points[i].x;
		pt.y = grounds->points[i].y;
		pt.z = 0;
		xy_grounds->push_back(pt);
	}

	KdTreeFLANN<PointXYZ> kdtree;
	kdtree.setInputCloud(xy_grounds);

	string lidar_path = "all_roofs.pcd";
	PointCloud<PointXYZ>::Ptr buildings(new PointCloud<PointXYZ>());
	pcl::io::loadPCDFile(lidar_path, *buildings);

	PointCloud<PointXYZI>::Ptr buildings_height(new PointCloud<PointXYZI>());

	for (int i = 0; i < buildings->size(); i++)
	{
		PointXYZ seed;
		seed.x = buildings->points[i].x;
		seed.y = buildings->points[i].y;
		seed.z = 0;
		vector<int> indexs;
		vector<float> diss;

		float height = 0;
		if (kdtree.nearestKSearch(seed, 1, indexs, diss) > 0)
		{
			int id = indexs[0];
			// calculating building height based on the elevation difference between roof point and its nearest ground point
			height = fabs(buildings->points[i].z - grounds->points[id].z);
		}

		PointXYZI pti;
		pti.x = seed.x;
		pti.y = seed.y;
		pti.z = buildings->points[i].z;
		pti.intensity = height;
		buildings_height->push_back(pti);
	}

	// save the caculation results
	string save_path = "building_height.txt";
	ofstream outfile(save_path, ios::trunc);
	float average_acc = 0;
	for (int i = 0; i < buildings_height->size(); i++)
	{
		outfile << setiosflags(ios::fixed) << setprecision(3) <<
			buildings_height->at(i).x << "," << buildings_height->at(i).y << "," << buildings_height->at(i).z << "," << buildings_height->at(i).intensity << endl;
	}
	outfile.close();
	buildings_height->clear();

	return 0;
}