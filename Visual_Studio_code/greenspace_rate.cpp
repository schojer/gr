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
#define RADIUS 500  // search radius for calculating greenspace coverage rate

using namespace std;
using namespace pcl;
using namespace cv;

int main()
{
	cout << "Calculating greenspace coverage ratio ..." << endl;
	
	// load plant point cloud
	string plant_path = "plants.pcd";
	PointCloud<PointXYZ>::Ptr plantCloud(new PointCloud<PointXYZ>());
	pcl::io::loadPCDFile(plant_path, *plantCloud);
	cout << "Plant points: " << plantCloud->size() << endl;

	// projecting plant points onto the horizontal plane
	PointCloud<PointXYZ>::Ptr h_plant(new PointCloud<PointXYZ>());
	for (unsigned int i = 0; i < plantCloud->size(); i++)
	{
		PointXYZ pt;
		pt.x = plantCloud->points[i].x;
		pt.y = plantCloud->points[i].y;
		pt.z = 0;
		h_plant->push_back(pt);
	}
	plantCloud->clear();

	KdTreeFLANN<PointXYZ> kdtree;
	kdtree.setInputCloud(h_plant);
	int size = RADIUS * 2.0 / RES + 1;

	// read the location of each building, here we used the footprint center to describe building location
	string path = "building_footprint_centers.txt";
	std::ifstream file(path, std::ios::in);
	std::string line;
	getline(file, line);

	vector<pair<int, float> > ratios;
	while (getline(file, line))
	{
		stringstream ss(line);
		string tmp;
		vector<string> v;
		while (getline(ss, tmp, ','))
		{
			v.push_back(tmp);
		}

		PointXYZ seed;
		seed.x = stod(v[1]);
		seed.y = stod(v[2]);
		seed.z = 0;

		float pt_ratio;
		vector<int> indexs;
		vector<float> diss;

		if (kdtree.radiusSearch(seed, RADIUS, indexs, diss) > 0)
		{
			// projecting plant points onto the horizontal plane and calculating area
			Mat img = Mat(size, size, CV_8UC1, Scalar(255));
			Point2f min_pt;
			min_pt.x = seed.x - RADIUS;
			min_pt.y = seed.y - RADIUS;
			int num = 0;
			for (int j = 0; j < indexs.size(); j++)
			{
				// computing the pixel coordinate of each laser scanning point
				int id = indexs[j];
				int px = (h_plant->points[id].x - min_pt.x) / RES;
				int py = (h_plant->points[id].y - min_pt.y) / RES;
				if (img.at<uchar>(px, py) == 255)
					num += 1;
				img.at<uchar>(px, py) = 66;
			}
			//  calculating rate
			pt_ratio = num * RES * RES * 1.0 / (M_PI * RADIUS * RADIUS);
		}
		indexs.clear();
		diss.clear();

		pair<int, float> rate;
		rate.first = stoi(v[0]);
		rate.second = pt_ratio;
		v.clear();
		ratios.push_back(rate);
	}
	file.close();

	// save greenspace coverage rate
	string save_path = "greenspace_rate.txt";
	ofstream outfile(save_path, ios::trunc);
	for (int i = 0; i < ratios.size(); i++)
	{
		// The first column represents the building ID, and the second represents greenspace rate
		outfile << setiosflags(ios::fixed) << setprecision(3) <<
			ratios[i].first << "," << ratios[i].second << endl;			
	}
	ratios.clear();
	outfile.close();

	return 0;
}