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
#include <pcl/kdtree/kdtree_flann.h>
#include <string>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#define RES 5.0		// projection size
#define RADIUS 500  // search radius

using namespace std;
using namespace pcl;
using namespace cv;

void readpop(string path, PointCloud<PointXYZI>::Ptr pops)
{
	std::ifstream file(path, std::ios::in);
	std::string line;
	std::vector<std::string> pt_inf;
	getline(file, line);
	cout << line << endl;

	// Æ½¾ùÖµ
	while (getline(file, line))
	{
		stringstream ss(line);
		string tmp;
		vector<float> v;
		while (getline(ss, tmp, ','))
		{
			v.push_back(atof(tmp.c_str()));
		}

		PointXYZI pt;
		pt.x = v[1];
		pt.y = v[2];
		pt.z = 0;
		pt.intensity = v[3];

		pops->push_back(pt);
		v.clear();
	}
	file.close();
}

// calculating human exposure to greenspace after roof greening in Hong Kong
int main()
{
	cout << "Human exposure to greenspace..." << endl;

	// read plant point cloud
	string plant_path = "plants.pcd";
	PointCloud<PointXYZ>::Ptr plants(new PointCloud<PointXYZ>());
	pcl::io::loadPCDFile(plant_path, *plants);
	cout << plants->size() << endl;
	PointCloud<PointXYZ>::Ptr xy_plants(new PointCloud<PointXYZ>());
	for (int i = 0; i < plants->size(); i++)
	{
		PointXYZ pt;
		pt.x = plants->points[i].x;
		pt.y = plants->points[i].y;
		pt.z = 0;
		xy_plants->push_back(pt);
	}
	plants->clear();

	// assuming all potential roofs are greened, then adding roof points into plant point cloud
	string groof_path = "potential_green_roofs.pcd";
	PointCloud<PointXYZ>::Ptr roofs(new PointCloud<PointXYZ>());
	pcl::io::loadPCDFile(groof_path, *roofs);
	cout << roofs->size() << endl;
	for (int i = 0; i < roofs->size(); i++)
	{
		PointXYZ pt;
		pt.x = roofs->points[i].x;
		pt.y = roofs->points[i].y;
		pt.z = 0;
		xy_plants->push_back(pt);
	}
	roofs->clear();

	KdTreeFLANN<PointXYZ> kdtree;
	kdtree.setInputCloud(xy_plants);

	// read population data of Hong Kong
	string pop_path = "constrained_HK_Population_city_hk1980.txt";
	PointCloud<PointXYZI>::Ptr pops(new PointCloud<PointXYZI>());
	readpop(pop_path, pops); 

	int size = RADIUS * 2.0 / RES + 1;
	PointCloud<PointXYZI>::Ptr green_pops(new PointCloud<PointXYZI>());
	PointCloud<PointXYZI>::Ptr un_greenarea(new PointCloud<PointXYZI>());
	// defining greenspace exposure
	vector<float> exposures(pops->size(), 0);

	for (int i = 0; i < pops->size(); i++)
	{
		PointXYZ seed;
		seed.x = pops->points[i].x;
		seed.y = pops->points[i].y;
		seed.z = 0;
		vector<int> indexs;
		vector<float> diss;

		PointXYZI pti;
		pti.x = pops->points[i].x;
		pti.y = pops->points[i].y;
		pti.z = 0;
		pti.intensity = 0;

		PointXYZI area_pt;
		area_pt.x = pops->points[i].x;
		area_pt.y = pops->points[i].y;
		area_pt.z = 0;
		area_pt.intensity = 0;

		if (kdtree.radiusSearch(seed, RADIUS, indexs, diss) > 0)
		{
			// projecting plant point onto the horizontal plane
			Mat img = Mat(size, size, CV_8UC1, Scalar(255));
			Point2f min_pt;
			min_pt.x = seed.x - RADIUS;
			min_pt.y = seed.y - RADIUS;
			int num = 0;
			for (int j = 0; j < indexs.size(); j++)
			{
				// calculating pixel coordinate of each laser scanning point
				int id = indexs[j];
				int px = (xy_plants->points[id].x - min_pt.x) / RES;
				int py = (xy_plants->points[id].y - min_pt.y) / RES;
				if (img.at<uchar>(px, py) == 255)
					num += 1;
				img.at<uchar>(px, py) = 66;
			}
			// greenspace area per people
			pti.intensity = num * RES * RES * 1.0 / pops->points[i].intensity;
			area_pt.intensity = num * RES * RES * 1.0 / 10000;

			// greenspace exposure area
			float expos = num * RES * RES * 1.0 / (M_PI * RADIUS * RADIUS);
			exposures[i] = expos;
		}
		indexs.clear();
		diss.clear();
		green_pops->push_back(pti);

		un_greenarea->push_back(area_pt);
	}
	xy_plants->clear();

	// calculating human exposure to greenspace index value
	float green_expos = 0.0, all_human = 0, all_gr = 0;
	for (int i = 0; i < exposures.size(); i++)
	{
		all_human = all_human + pops->points[i].intensity;
		all_gr = all_gr + pops->points[i].intensity * exposures[i];
	}
	green_expos = all_gr / all_human;
	cout << setiosflags(ios::fixed) << setprecision(3) << "   ; human exposure to greenspace£º" << green_expos << endl;

	exposures.clear();
	pops->clear();

	return 0;
}