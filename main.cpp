#include <string>
#include <cstring>
#include <iostream>
#include <sys/time.h>

#include "fast_als.cuh"

using namespace std;

int main(int argc, char *argv[])
{
	string output_file_name;
	string likes_file_name;
	int features_size = 50;
	int csimples = 0;
	int cit = 10;
	int likes_format = 0;
	float als_alfa = 5;
	int samples_for_calc_error_users = 0;
	int samples_for_calc_error_items = 0;
	int count_gpu = 0;

	for(int i = 1; i <  argc; i++)
	{
		std::string sarg = argv[i];
		if( sarg == "--likes")
		{
			i++;
			likes_file_name = argv[i];
		}
		else
		if( sarg == "--f_size")
		{
			i++;
			features_size = atoi(argv[i]);
			std::cerr << " Count features:  " << features_size << std::endl;
		}
		else
		if( sarg == "--csamples")
		{
			i++;
			csimples = atoi(argv[i]);
		}
		else
		if( sarg == "--it")
		{
			i++;
			cit = atoi(argv[i]);
		}else
		if( sarg == "--out")
		{
			i++;
			output_file_name = argv[i];
		}else
		if( sarg == "--likes-format")
		{
			i++;
			likes_format = atoi(argv[i]);
		}else
		if( sarg == "--als-alfa")
		{
			i++;
			als_alfa = atof(argv[i]);
		}else
		if( sarg == "--als-error")
		{
			i++;
			std::string samples(argv[i]);
			size_t pos = samples.find(":");
			if(pos == std::string::npos)
				samples_for_calc_error_users  =  samples_for_calc_error_items =  atoi(argv[i]);
			else
			{
				samples_for_calc_error_users = atoi(samples.substr(0,pos).c_str());
				samples_for_calc_error_items = atoi(samples.substr(pos+1).c_str());
			}
		}else
		if( sarg == "--gpu-count")
		{
			i++;
			count_gpu = atoi(argv[i]);
		}
	}

	std::ifstream f_stream(likes_file_name.c_str() );
	std::istream& in((likes_file_name.length() == 0) ? std::cin : f_stream);

	std::cerr << " Count ALS iteration " << cit << std::endl;
	std::cerr << " Start Matrix Factorization - Fast ALS " << std::endl;
	std::cerr << " Input file format -  " << likes_format << std::endl;
	std::cerr << " ALS alfa -  " << als_alfa << std::endl;
	std::cerr << " ALS count GPU -  " << count_gpu << std::endl;

	fast_als als_alg(in, features_size, als_alfa, 30, csimples, likes_format, samples_for_calc_error_users, samples_for_calc_error_items, count_gpu);

	struct timeval t1;
	struct timeval t2;

	gettimeofday(&t1, NULL);
	als_alg.calculate(cit);
	gettimeofday(&t2, NULL);

	std::cout << "als calc time: " << t2.tv_sec - t1.tv_sec << std::endl;

	std::ofstream fout_users((output_file_name+".ufea").c_str());
	als_alg.serialize_users(fout_users);
	fout_users.close();

	std::ofstream fout_items((output_file_name+".ifea").c_str());
	als_alg.serialize_items(fout_items);
	fout_items.close();

	std::ofstream fout_umap((output_file_name+".umap").c_str());
	als_alg.serialize_users_map(fout_umap);
	fout_umap.close();

	std::ofstream fout_imap((output_file_name+".imap").c_str());
	als_alg.serialize_items_map(fout_imap);
	fout_imap.close();

	return 0;
}
