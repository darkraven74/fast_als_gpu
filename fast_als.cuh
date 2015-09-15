#ifndef FAST_ALS_CUH_
#define FAST_ALS_CUH_

#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
//#include <random>

#include <cula.h>
#include <cula_blas.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>

class fast_als {
public:
	///
	/// Definition of features vector
	///
	typedef std::vector<float> features_vector;
	typedef std::vector< std::vector<int> >	 likes_vector;
	typedef std::vector< std::vector<float> >   likes_weights_vector;
	typedef std::vector<int> likes_vector_item;
	typedef std::vector<float> likes_weights_vector_item;

	typedef thrust::device_vector<float> d_features_vector;
	typedef thrust::device_vector<int>	 d_likes_vector;
	typedef thrust::device_vector<float> d_likes_weights_vector;
	///
	/// Ctor
	/// Inputs are:
	/// stream with triplets:
	/// count_features - count latent features
	/// format of likes
	/// 0 - old
	/// 1 - simple
	/// <user> <item> <weight>
	///
	fast_als(std::istream& tuples_stream,
			int count_features,
			float alfa,
			float gamma,
			int count_samples,
			int likes_format,
			int count_error_samples_for_users,
			int count_error_samples_for_items);

	virtual ~fast_als();

	///
	/// Calculate als (Matrix Factorization)
	/// in
	/// count_iterations - count iterations
	///
	virtual void calculate(int count_iterations);

	virtual void MSE();
	virtual float hit_rate();

	///
	/// Get Items features vector
	///
	const features_vector& get_features_items() const { return _features_items; }
	int get_count_items() const { return _count_items; }

	///
	/// Get Users features vector
	///
	const features_vector& get_features_users() const { return _features_users; }
	int get_count_users() const { return _count_users; }

	void serialize_map(std::ostream& out, std::map<unsigned long, int>& out_map);
	void serialize_matrix(std::ostream& out, const float* mat, int crow, int ccol, bool id = false);
	void serialize_users(std::ostream& out);
	void serialize_items(std::ostream& out);
	void serialize_users_map(std::ostream& out);
	void serialize_items_map(std::ostream& out);

protected:
	///
	/// Read likes from stream
	/// if format == 0
	/// user group item
	/// if format == 1
	/// user item weight
	///
	void read_likes(std::istream& tuples_stream, int count_simples, int format);

	///
	/// fill random values to features matrix
	///
	void fill_rnd(features_vector& in_v, int in_size);

	///
	/// solve one iteration of als
	///
	void solve(
			const likes_vector& likes,
			const likes_weights_vector& weights,
			features_vector& in_v,
			int in_size,
			features_vector& out_v,
			int out_size,
			int _count_features,
			std::vector<int>& likes_offsets,
			std::vector<int>& likes_sizes);

	void mulYxY(features_vector& in_v, int in_size, std::vector<float>& ans);

	fast_als::features_vector calc_g(features_vector& in_v, int in_size, int _count_features);

	void calc_ridge_regression_gpu(
				const likes_vector& likes,
				const likes_weights_vector& weights,
				const features_vector& in_v,
				features_vector& out_v,
				int out_size,
				int _count_features,
				features_vector& g,
				std::vector<int>& likes_offsets,
				std::vector<int>& likes_sizes);

	void generate_test_set();

	void init_helper_vectors();

private:
	///
	/// features vectors, for users and items
	///
	features_vector _features_users;
	int _count_users;
	features_vector _features_items;
	int _count_items;

	int _count_features;
	int _count_samples;


	///
	/// Internal data
	///
	std::map<unsigned long, int> _users_map;
	std::map<unsigned long, int> _items_map;
	likes_vector                 _user_likes;
	likes_weights_vector         _user_likes_weights;
	likes_vector                 _item_likes;
	likes_weights_vector         _item_likes_weights;

	std::vector<int> d_user_offsets;
	std::vector<int> d_item_offsets;

	std::vector<int> d_user_sizes;
	std::vector<int> d_item_sizes;

	float _als_alfa;
	float _als_gamma;

	///
	/// Count samples for calculate error
	///
	int _count_error_samples_for_users;
	std::vector<int>   users_for_error;
	int _count_error_samples_for_items;
	std::vector<int>   items_for_error;

	std::vector<std::pair<int, int> > test_set;
	likes_weights_vector _user_likes_weights_temp;

	culaStatus status;
    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status;
};

#endif /* FAST_ALS_CUH_ */
