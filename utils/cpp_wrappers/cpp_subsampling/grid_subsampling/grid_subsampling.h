

#include "../../cpp_utils/cloud/cloud.h"

#include <set>
#include <cstdint>

using namespace std;

class SampledData
{
public:

	// Elements
	// ********

	int count;
	PointXYZ point;
	vector<float> features;
	vector<unordered_map<int, int>> labels;
	vector<unordered_map<int, int>> ins_labels;


	// Methods
	// *******

	// Constructor
	SampledData() 
	{ 
		count = 0; 
		point = PointXYZ();
	}

	SampledData(const size_t fdim, const size_t ldim)
	{
		count = 0;
		point = PointXYZ();
	    features = vector<float>(fdim);
	    labels = vector<unordered_map<int, int>>(ldim);
	    ins_labels = vector<unordered_map<int, int>>(ldim);
	}

	// Method Update
	void update_all(const PointXYZ p, vector<float>::iterator f_begin, vector<int>::iterator l_begin, vector<int>::iterator il_begin)
	{

		count += 1;
		point += p;
		transform (features.begin(), features.end(), f_begin, features.begin(), plus<float>());
		int i = 0;
		for(vector<int>::iterator it = l_begin; it != l_begin + labels.size(); ++it)
		{
		    labels[i][*it] += 1;
		    i++;
		}

        int j = 0;
		for(vector<int>::iterator it = il_begin; it != il_begin + ins_labels.size(); ++it)
		{
		    ins_labels[j][*it] += 1;
		    j++;
		}
		return;
	}
	void update_features(const PointXYZ p, vector<float>::iterator f_begin)
	{
		count += 1;
		point += p;
		transform (features.begin(), features.end(), f_begin, features.begin(), plus<float>());
		return;
	}
	void update_classes(const PointXYZ p, vector<int>::iterator l_begin, vector<int>::iterator il_begin)
	{
		count += 1;
		point += p;
		int i = 0;
		for(vector<int>::iterator it = l_begin; it != l_begin + labels.size(); ++it)
		{
		    labels[i][*it] += 1;
		    i++;
		}
		for(vector<int>::iterator it = il_begin; it != il_begin + ins_labels.size(); ++it)
		{
		    ins_labels[i][*it] += 1;
		    i++;
		}
		return;
	}
	void update_points(const PointXYZ p)
	{
		count += 1;
		point += p;
		return;
	}
};



void grid_subsampling(vector<PointXYZ>& original_points,
                      vector<PointXYZ>& subsampled_points,
                      vector<float>& original_features,
                      vector<float>& subsampled_features,
                      vector<int>& original_classes,
                      vector<int>& subsampled_classes,
                      vector<int>& original_ins_labels,
                      vector<int>& subsampled_ins_labels,
                      float sampleDl,
                      int verbose);

