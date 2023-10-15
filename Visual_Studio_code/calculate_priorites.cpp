#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <utility>

using namespace std;

// defining building attributes
struct ATt {
	int id;					// building ID in footprint file
	bool isgreen;			// is greening?
	float year;				// building age
	float category;			// building category (private, public, and miscellaneous)
	float temp_spring;		// temperature: spring
	float temp_summer;		// summer
	float temp_autumn;		// autumn
	float temp_winter;		// winter
	float precipitation;	// annual precipitation
	float greenratio;		// greenspace coverage rate around buildings
	float distance;			// distance between building and its nearest main road
	float income;			// median monthly income
	float height;			// building height
};

// standardizing all indicators
void normalization(const int id, vector<float> input, vector<float>& output)
{
	// retrieving maximun and minmum values
	float max_a = *max_element(input.begin(), input.end());
	float min_a = *min_element(input.begin(), input.end());

	if (id == 1)
	{
		// positive indicators
		for (unsigned int i = 0; i < input.size(); i++)
			output[i] = (input[i] - min_a) / (max_a - min_a);
	}
	else
	{
		// negative indicators
		for (unsigned int i = 0; i < input.size(); i++)
			output[i] = (max_a - input[i]) / (max_a - min_a);
	}
}

// read attribute file 'indicators_and_priority.txt'
void readAttributes(string path, vector<ATt>& atts)
{
	std::ifstream file(path, std::ios::in);
	std::string line;
	std::vector<std::string> pt_inf;
	getline(file, line);
	cout << line << endl;

	while (getline(file, line))
	{
		stringstream ss(line);
		string tmp;
		vector<float> v;
		while (getline(ss, tmp, ','))
		{
			v.push_back(atof(tmp.c_str()));
		}

		ATt inf;
		inf.id = int(v[0]);
		inf.distance = v[1];
		inf.category = int(v[2]);
		inf.income = v[3];
		inf.precipitation = v[4];

		inf.year = int(v[5]);

		inf.temp_spring = v[6];
		inf.temp_summer = v[7];
		inf.temp_autumn = v[8];
		inf.temp_winter = v[9];

		inf.greenratio = v[10];

		inf.height = v[11];

		atts.push_back(inf);
		v.clear();
	}
	file.close();
}

// calculating roof greening priorities by using an equally weighted strategy
void CalculateAverage(string path, string save_path)
{
	// First, read attribute information
	vector<ATt> temp_atts;
	readAttributes(path, temp_atts);

	vector<float> ori_distance;		// distance indicator
	vector<float> ori_precipitati;	// precipitation indicator
	vector<float> ori_tempspring;	// temperature
	vector<float> ori_tempsummer;
	vector<float> ori_tempautumn;
	vector<float> ori_tempwinter;
	vector<float> ori_income;		// median income

	for (int i = 0; i < temp_atts.size(); i++)
	{
		// buildings located more than 500 meters from the main road were assigned a distance indicator of 0
		if (temp_atts[i].distance > 500)
			ori_distance.push_back(500);
		else
			ori_distance.push_back(temp_atts[i].distance);

		ori_precipitati.push_back(temp_atts[i].precipitation);
		ori_tempspring.push_back(temp_atts[i].temp_spring);
		ori_tempsummer.push_back(temp_atts[i].temp_summer);
		ori_tempautumn.push_back(temp_atts[i].temp_autumn);
		ori_tempwinter.push_back(temp_atts[i].temp_winter);

		ori_income.push_back(temp_atts[i].income);
	}

	// Second, standardizing all indicators
	// for greenspace coverage rate and building category
	vector<float> norm_greens(temp_atts.size());
	vector<float> norm_category(temp_atts.size());
	for (int i = 0; i < temp_atts.size(); i++)
	{
		norm_greens[i] = 1 - temp_atts[i].greenratio;

		if (temp_atts[i].category == 1 || temp_atts[i].category == 2)	// private building
			norm_category[i] = 0.5;
		else if (temp_atts[i].category == 3)	// public
			norm_category[i] = 1.0;
		else
			norm_category[i] = 0.75;	// miscellaneous
	}


	// for distance indicator
	vector<float> norm_distance(temp_atts.size());
	normalization(2, ori_distance, norm_distance);
	ori_distance.clear();

	// for precipitation indicator
	vector<float> norm_precipitation(temp_atts.size());
	normalization(1, ori_precipitati, norm_precipitation);
	ori_precipitati.clear();

	// for temperature indicator
	// spring
	vector<float> norm_spring(temp_atts.size());
	normalization(1, ori_tempspring, norm_spring);
	ori_tempspring.clear();
	// summer
	vector<float> norm_summer(temp_atts.size());
	normalization(1, ori_tempsummer, norm_summer);
	ori_tempsummer.clear();
	// autumn
	vector<float> norm_autumn(temp_atts.size());
	normalization(1, ori_tempautumn, norm_autumn);
	ori_tempautumn.clear();
	// winter
	vector<float> norm_winter(temp_atts.size());
	normalization(1, ori_tempwinter, norm_winter);
	ori_tempwinter.clear();

	// calculating temperature weight by combing temperature values of four seasons
	vector<float> norm_temperature(temp_atts.size());
	for (int i = 0; i < temp_atts.size(); i++)
	{
		norm_temperature[i] = 0.1 * norm_spring[i] + 0.4 * norm_summer[i] + 0.4 * norm_autumn[i] + 0.1 * norm_winter[i];
	}
	norm_spring.clear();
	norm_summer.clear();
	norm_autumn.clear();
	norm_winter.clear();

	// for income indicator
	vector<float> norm_income(temp_atts.size());
	normalization(2, ori_income, norm_income);
	ori_income.clear();

	// calculating roof greening priorities
	vector<pair<int, float> > weights;

	for (int i = 0; i < temp_atts.size(); i++)
	{
		pair<int, float> wei;
		wei.first = temp_atts[i].id;
		wei.second = norm_precipitation[i] * 1.0 / 6 + norm_temperature[i] * 1.0 / 6 + norm_greens[i] * 1.0 / 6 +
			norm_distance[i] * 1.0 / 6 + norm_income[i] * 1.0 / 6 + norm_category[i] * 1.0 / 6;
		weights.push_back(wei);
	}
	norm_precipitation.clear();
	norm_temperature.clear();
	norm_greens.clear();
	norm_distance.clear();
	norm_income.clear();
	norm_category.clear();

	// save results
	ofstream outfile(save_path, ios::trunc);
	for (int i = 0; i < weights.size(); i++)
	{
		outfile << weights[i].first << "," << weights[i].second << endl;
	}

	weights.clear();
	temp_atts.clear();
	outfile.close();
}

int main()
{
	string path = "greening_indicators.txt";
	string save_path = "final_greening_priorities.txt";
	CalculateAverage(path, save_path);

	return 0;
}