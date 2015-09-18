#include <cmath>
#include <algorithm>
#include <set>
#include <map>
#include <ctime>
#include <thrust/functional.h>

#include "fast_als.cuh"

#define BLOCK_SIZE 32

void checkStatus(culaStatus status)
{
	char buf[256];

	if(!status)
		return;

	culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
	printf("cula error! %s\n", buf);

	culaShutdown();
	exit(EXIT_FAILURE);
}

fast_als::fast_als(std::istream& tuples_stream,
		int count_features,
		float alfa,
		float gamma,
		int count_samples,
		int likes_format,
		int count_error_samples_for_users,
		int count_error_samples_for_items) :
		_count_users(0),
		_count_items(0),
		_count_features(count_features),
		_als_alfa(alfa),
		_als_gamma(gamma),
		_count_error_samples_for_users(count_error_samples_for_users),
		_count_error_samples_for_items(count_error_samples_for_items),
		_count_samples(count_samples),
		cublas_status(cublasCreate(&cublas_handle))
{
	status = culaInitialize();
	checkStatus(status);

//	srand(time(NULL));
	srand(34);

	read_likes(tuples_stream, count_samples, likes_format);

	//generate_test_set();

	_features_users.assign(_count_users * _count_features, 0 );
	_features_items.assign(_count_items * _count_features, 0 );
}

fast_als::~fast_als()
{
	culaShutdown();
	cublas_status = cublasDestroy(cublas_handle);
}

void fast_als::read_likes(std::istream& tuples_stream, int count_simples, int format)
{
	std::string line;
	char const tab_delim = '\t';
	int i = 0;

	while(getline(tuples_stream, line))
	{
		std::istringstream line_stream(line);
		std::string value;
		getline(line_stream, value, tab_delim);
		unsigned long uid = atol(value.c_str());

		if (_users_map.find(uid) == _users_map.end())
		{
			_users_map[uid] = _count_users;
			_count_users++;
			_user_likes.push_back(std::vector<int>());
			_user_likes_weights.push_back(std::vector<float>());
			_user_likes_weights_temp.push_back(std::vector<float>());
		}

		int user = _users_map[uid];

		if( format == 0 )
		{
			getline(line_stream, value, tab_delim);
//			unsigned long gid = atol(value.c_str());
		}

		getline(line_stream, value, tab_delim);
		unsigned long iid = atol(value.c_str());
		float weight = 1;

		float weight_temp = 1;

		if(format == 1)
		{
			getline(line_stream, value, tab_delim);
			weight_temp = atof( value.c_str() );
		}

		if (_items_map.find(iid) == _items_map.end())
		{
			_items_map[iid] = _count_items;
			_item_likes.push_back(std::vector<int>());
			_item_likes_weights.push_back(std::vector<float>());
			_count_items++;
		}

		int item = _items_map[iid];
		///
		/// adding data to user likes
		/// and to item likes
		///
		_user_likes[user].push_back( item );
		_user_likes_weights[user].push_back( weight );
		_user_likes_weights_temp[user].push_back( weight_temp );
		_item_likes[item].push_back( user );
		_item_likes_weights[item].push_back( weight );

		if (i % 10000 == 0) std::cout << i << " u: " << _count_users << " i: " << _count_items << "\r";

		///std::cout << "u:" << user << " -> " << item << std::endl;
		///std::cout << "i:" << item << " -> " << user << std::endl;

		i++;
		if(count_simples && i >= count_simples) break;
	}

	if (!count_simples)
	{
		_count_samples = i;
	}

	std::cout.flush();

//	std::cout << "csamples: " << _count_samples << std::endl;

	std::cout << "\ntotal:\n u: " << _count_users << " i: " << _count_items << std::endl;
}

void fast_als::generate_test_set()
{
	for (int i = 0; i < _count_users; i++)
	{
		int size = _user_likes[i].size();
		for (int j = 0; j < size / 2;)
		{
			int id = rand() % _user_likes[i].size();

			if (_user_likes_weights_temp[i][id] < 4)
			{
				continue;
			}

			test_set.push_back(std::make_pair(i, _user_likes[i][id]));

			for (unsigned int k = 0; k < _item_likes[_user_likes[i][id]].size(); k++)
			{
				if (_item_likes[_user_likes[i][id]][k] == i)
				{
					_item_likes[_user_likes[i][id]].erase(_item_likes[_user_likes[i][id]].begin() + k);
					_item_likes_weights[_user_likes[i][id]].erase(_item_likes_weights[_user_likes[i][id]].begin() + k);
				}
			}

			_user_likes[i].erase(_user_likes[i].begin() + id);
			_user_likes_weights[i].erase(_user_likes_weights[i].begin() + id);
			_user_likes_weights_temp[i].erase(_user_likes_weights_temp[i].begin() + id);
			_count_samples--;
			break;
		}
	}
}

void fast_als::fill_rnd(features_vector& in_v, int in_size)
{
	std::cerr << "Generate random features.. ";
//	std::default_random_engine generator(time(NULL));
//	std::default_random_engine generator(34);
//	std::normal_distribution<float> distribution(0, 1);

	for (int i = 0; i < in_size * _count_features; i++)
	{
//		in_v[i] = distribution(generator);
		in_v[i] = ((float)rand()/(float)RAND_MAX);
	}

	std::cerr << "done" << std::endl;
}

void fast_als::calculate(int count_iterations)
{
	fill_rnd(_features_users, _count_users);
	init_helper_vectors();

	std::ofstream hr10("hr10.txt");

	for(int i = 0; i < count_iterations; i++)
	{
		time_t start =  time(0);
		std::cerr << "ALS Iteration: " << i << std::endl;

		std::cerr << "Items." << std::endl;
		solve(_item_likes, _item_likes_weights, _features_users, _count_users, _features_items, _count_items,
				_count_features, d_item_offsets, d_item_sizes);
		std::cerr << "Users." << std::endl;
		solve(_user_likes, _user_likes_weights, _features_items, _count_items, _features_users, _count_users,
				_count_features, d_user_offsets, d_user_sizes);

		time_t end =  time(0);
		std::cerr << "==== Iteration time : " << end - start << std::endl;

		//MSE();
		//hr10 << hit_rate() << std::endl;

	}

	hr10.close();
}

void fast_als::solve(
		const likes_vector& likes,
		const likes_weights_vector& weights,
		features_vector& in_v,
		int in_size,
		features_vector& out_v,
		int out_size,
		int _count_features,
		std::vector<int>& likes_offsets,
		std::vector<int>& likes_sizes)
{
	time_t start = time(0);
	fast_als::features_vector g = calc_g(in_v, in_size, _count_features);
//	fast_als::features_vector g(_count_features * _count_features);
//	fill_rnd(g, _count_features);
	cudaDeviceSynchronize();
	start = time(0) - start;
	std::cerr << "g calc: " << start << std::endl;

	if ( cudaSuccess != cudaPeekAtLastError() )
				std::cerr <<  "!WARN - Cuda error (g calc) : "  << cudaGetErrorString(cudaGetLastError()) << std::endl;

	start = time(0);
	calc_ridge_regression_gpu(likes, weights, in_v, out_v, out_size, _count_features, g, likes_offsets, likes_sizes);
	start = time(0) - start;
	std::cerr << "regression calc: " << start << std::endl;

}

#define RESERVED_MEM 0xA00000

void fast_als::mulYxY(features_vector& in_v, int in_size, std::vector<float>& ans)
{
	thrust::device_vector<float> device_YxY(_count_features * _count_features, 0);
	ans.assign(_count_features * _count_features, 0);
	float alpha = 1;
	float beta = 1;
	///
	/// Calculate size of block for input matrix
	/// input matreix is Y matrix
	///
	size_t cuda_free_mem = 0;
	size_t cuda_total_mem = 0;

	cudaMemGetInfo(&cuda_free_mem, &cuda_total_mem);
	cuda_free_mem -= RESERVED_MEM;
	std::cerr << "Cuda memory YxY free: " << cuda_free_mem << std::endl;


	///
	/// detect size of block of Y matrix
	///
	int count_rows = cuda_free_mem / (_count_features *sizeof(float));

	count_rows = count_rows >= in_size? in_size:count_rows;
	int parts_size = in_size / count_rows + ( (in_size  % count_rows != 0)? 1 : 0);
	thrust::device_vector<float> x_device(count_rows * _count_features, 0);

	for(int part=0; part < parts_size; part++)
	{
		int actual_part_size = ( part == parts_size-1 && in_size  % count_rows != 0) ?  in_size  % count_rows : count_rows;

		/// copy to memory
		/*for(int i=0; i < _count_features; ++i)
		{
			size_t offset = i* in_size + part * count_rows;
			thrust::copy(in_v.begin()+ offset,  in_v.begin()+ offset + actual_part_size, x_device.begin() + i * actual_part_size) ;
		}*/

		size_t offset = part * _count_features * count_rows;
		thrust::copy(in_v.begin()+ offset,  in_v.begin()+ offset + actual_part_size * _count_features, x_device.begin());


		/*cublas_status = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, _count_features, _count_features, actual_part_size , &alpha,
							 thrust::raw_pointer_cast(&x_device[0]), actual_part_size , thrust::raw_pointer_cast(&x_device[0]),
							 actual_part_size, &beta, thrust::raw_pointer_cast(&device_YxY[0]), _count_features);*/

		cublas_status = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, _count_features, _count_features, actual_part_size , &alpha,
									 thrust::raw_pointer_cast(&x_device[0]), _count_features , thrust::raw_pointer_cast(&x_device[0]),
									 _count_features, &beta, thrust::raw_pointer_cast(&device_YxY[0]), _count_features);

		/*status = culaSgemm('N', 'T', _count_features, _count_features, in_size, 1.0f, &in_v[0], _count_features,
					&in_v[0], _count_features, 0.0f, &A[0], _count_features);
		checkStatus(status);*/



		if ( cublas_status != 0 )
			std::cerr <<  "!WARN - Cuda error (als::mulYxY -> cublasSgemm) : "  << cublas_status << std::endl;
	}

	thrust::copy(device_YxY.begin(), device_YxY.end(), ans.begin());
}

fast_als::features_vector fast_als::calc_g(features_vector& in_v, int in_size, int _count_features)
{
	std::vector<float> A(_count_features * _count_features);
	std::vector<float> U(_count_features * _count_features);
	std::vector<float> G(_count_features * _count_features);
	std::vector<float> S(_count_features);

	/*status = culaSgemm('N', 'T', _count_features, _count_features, in_size, 1.0f, &in_v[0], _count_features,
			&in_v[0], _count_features, 0.0f, &A[0], _count_features);
	checkStatus(status);*/


	mulYxY(in_v, in_size, A);


	/*std::cout << "====== YxY === " << std::endl;
	for (int i = 0; i < _count_features; i++)
	{
		for (int j = 0; j < _count_features; j++)
		{
			std::cout << A[i * _count_features + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "End of ====== YxY === " << std::endl;
	*/




	status = culaSgesvd('N', 'S', _count_features, _count_features, &A[0], _count_features, &S[0], NULL,
			_count_features, &U[0], _count_features);
	checkStatus(status);

	std::vector<float> lam_sqrt(_count_features * _count_features, 0.0);

	for (int i = 0; i < _count_features; i++)
	{
		lam_sqrt[i * _count_features + i] = sqrt(S[i]);
	}

	status = culaSgemm('N', 'N', _count_features, _count_features, _count_features, 1.0f, &lam_sqrt[0], _count_features,
			&U[0], _count_features, 0.0f, &G[0], _count_features);
	checkStatus(status);

	//G transpose
	/*for (int i = 0; i < _count_features; i++)
	{
		for (int j = 0; j < i; j++)
		{
			std::iter_swap(G.begin() + i * _count_features + j, G.begin() + j * _count_features + i);
		}
	}*/

	/*for (int i = 0; i < _count_features; i++)
	{
		for (int j = 0; j < _count_features; j++)
		{
			std::cout << G[i * _count_features + j] << "\t";
		}
		std::cout << std::endl;
	}*/

	return G;
}

__global__ void ridge_regression_kernel(const float* weights, const float* in_v, float* out_v, int out_size,
		int _count_features, float* g, int* likes_sizes, int* likes_offsets, float _als_alfa, float _als_gamma/*, float* errors*/)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < out_size)
	{
		int id = x;
		int in_size = likes_sizes[id];
		int in_offset = likes_offsets[id];
		float local_errors[14000]; //TODO: array size

		int out_offset = id * _count_features;
		int in_offset_features = in_offset * _count_features;

		for (int i = 0; i < in_size; i++)
		{
			int in_cur_off = in_offset_features + i;
			float sum = 0;
			for (int j = 0; j < _count_features; j++)
			{
				sum += out_v[out_offset + j] * in_v[in_cur_off + j * in_size];
			}
			float c = 1 + _als_alfa * weights[in_offset + i];
			local_errors[i] = (c / (c - 1)) - sum;
		}

		for (int i = 0; i < _count_features; i++)
		{
			float sum = 0;
			for (int j = 0; j < _count_features; j++)
			{
				sum += out_v[out_offset + j] * g[j * _count_features + i];
			}
			local_errors[in_size + i] = -sum;
		}

		for (int k = 0; k < _count_features; k++)
		{
			float out_v_cur = out_v[out_offset + k];

			float a = 0;
			float d = 0;
			for (int i = 0; i < in_size; i++)
			{
				float c = _als_alfa * weights[in_offset + i];
				float in_v_cur =  in_v[in_offset_features + k * in_size + i];
				a += c * in_v_cur * in_v_cur;
				d += c * in_v_cur * (local_errors[i] + out_v_cur * in_v_cur);
			}
			for (int i = 0; i < _count_features; i++)
			{
				float g_cur = g[k * _count_features + i];
				a += g_cur * g_cur;
				d += g_cur * (local_errors[in_size + i] + out_v_cur * g_cur);
			}

			float out_v_cur_new = d / (_als_gamma + a);

			out_v[out_offset + k] = out_v_cur_new;

			float out_diff = out_v_cur - out_v_cur_new;

			for (int i = 0; i < in_size; i++)
			{
				local_errors[i] += out_diff * in_v[in_offset_features + k * in_size + i];
			}
			for (int i = 0; i < _count_features; i++)
			{
				local_errors[in_size + i] += out_diff * g[k * _count_features + i];
			}
		}
	}
}

void fast_als::calc_ridge_regression_gpu(
		const likes_vector& likes,
		const likes_weights_vector& weights,
		const features_vector& in_v,
		features_vector& out_v,
		int out_size,
		int _count_features,
		features_vector& g,
		std::vector<int>& likes_offsets,
		std::vector<int>& likes_sizes)
{
	d_features_vector d_g(g);

	int count_rows = 500000; //TODO: fix count_rows
	count_rows = count_rows >= out_size ? out_size : count_rows;
	int parts_size = out_size / count_rows + ((out_size % count_rows != 0) ? 1 : 0);

	for(int part = 0; part < parts_size; part++)
	{
		int actual_part_size = (part == parts_size - 1 && out_size % count_rows != 0) ? out_size % count_rows : count_rows;
		thrust::device_vector<float> d_weights;
		thrust::device_vector<int> d_likes_offsets(likes_offsets.begin() + part * count_rows, likes_offsets.begin() + part * count_rows + actual_part_size);
		int sub = *(likes_offsets.begin() + part * count_rows);
		thrust::for_each(d_likes_offsets.begin(), d_likes_offsets.end(), thrust::placeholders::_1 -= sub);

		thrust::device_vector<int> d_likes_sizes(likes_sizes.begin() + part * count_rows, likes_sizes.begin() + part * count_rows + actual_part_size);
		d_features_vector d_in_v;

		std::vector<float> h_weights;
		features_vector h_in_v;


		size_t offset = part * _count_features * count_rows;
		d_features_vector d_out_v(out_v.begin() + offset, out_v.begin() + offset + actual_part_size * _count_features);
		for (int i = 0; i < actual_part_size; i++)
		{
			h_weights.insert(h_weights.end(), (weights.begin() + part * count_rows + i)->begin(), (weights.begin() + part * count_rows + i)->end());
			for (int k = 0; k < _count_features; k++)
			{
				for (int j = 0; j < likes_sizes[part * count_rows + i]; j++)
				{
					int id = likes[part * count_rows + i][j];
					h_in_v.insert(h_in_v.end(), *(in_v.begin() + id * _count_features + k));
				}
			}
		}

		if ( cudaSuccess != cudaPeekAtLastError() )
					std::cerr <<  "!WARN - Cuda error (thrust in regression) : "  << cudaGetErrorString(cudaGetLastError()) << std::endl;

		d_weights = h_weights;
		d_in_v = h_in_v;

		dim3 block(BLOCK_SIZE, 1);
		dim3 grid(1 + actual_part_size / BLOCK_SIZE, 1);

		ridge_regression_kernel<<<grid, block>>>(thrust::raw_pointer_cast(&d_weights[0]),
				thrust::raw_pointer_cast(&d_in_v[0]), thrust::raw_pointer_cast(&d_out_v[0]), actual_part_size, _count_features,
				thrust::raw_pointer_cast(&d_g[0]), thrust::raw_pointer_cast(&d_likes_sizes[0]), thrust::raw_pointer_cast(&d_likes_offsets[0]),
				_als_alfa, _als_gamma);

		cudaDeviceSynchronize();
		cudaError_t lastErr = cudaGetLastError();
		if (lastErr != cudaSuccess)
		{
			std::cout << "cuda error in kernel! " << cudaGetErrorString(lastErr) <<  std::endl;
		}

		thrust::copy(d_out_v.begin(), d_out_v.end(), out_v.begin() + offset);
		if ( cudaSuccess != cudaPeekAtLastError() )
							std::cerr <<  "!WARN - Cuda error (thrust in regression 2) : "  << cudaGetErrorString(cudaGetLastError()) << std::endl;
	}
}

void fast_als::MSE()
{

}

float fast_als::hit_rate()
{
	std::vector<float> predict(_count_users * _count_items);

	//predict = P * Q^t
	//predict = P^t * Q
	status = culaSgemm('T', 'N', _count_users, _count_items, _count_features, 1.0f, &_features_users[0], _count_features,
			&_features_items[0], _count_features, 0.0f, &predict[0], _count_users);
	checkStatus(status);

	for (int i = 0; i < _count_users; i++)
	{
		for (unsigned int j = 0; j < _user_likes[i].size(); j++)
		{
			int item_id = _user_likes[i][j];
			predict[item_id * _count_users + i] = -1000000;
		}
	}

	std::set<std::pair<int, int> > test_set_set(test_set.begin(), test_set.end());

	std::set<std::pair<int, int> > recs;
	for (int i = 0; i < _count_users; i++)
	{
		std::vector<float> v;
		for (int j = 0; j < _count_items; j++)
		{
			v.push_back(predict[j * _count_users + i]);
		}

		for (int j = 0; j < 10; j++)
		{
			std::vector<float>::iterator it = std::max_element(v.begin(), v.end());
			int item = std::distance(v.begin(), it);
			v[item] = -1000000;
			recs.insert(std::make_pair(i, item));
		}
	}

	float tp = 0;
	for (std::set<std::pair<int, int> >::iterator it = test_set_set.begin(); it != test_set_set.end(); it++)
	{
		if (recs.count(*it))
		{
			tp++;
		}
	}
	float hr10 = tp * 1.0 / test_set_set.size();

	std::cout << hr10 << std::endl;

	return hr10;
}


void fast_als::serialize_users_map(std::ostream& out)
{
	serialize_map(out, _users_map);
}

void fast_als::serialize_items_map(std::ostream& out)
{
	serialize_map(out, _items_map);
}

void fast_als::serialize_map(std::ostream& out, std::map<unsigned long, int>& out_map)
{
	std::map<unsigned long, int>::iterator it = out_map.begin();
	for( ; it != out_map.end(); it++)
	{
		out << it->first << "\t" << it->second << std::endl;
	}
}

void fast_als::serialize_items(std::ostream& out)
{
	const fast_als::features_vector& items = get_features_items();
	serialize_matrix(out, &items.front(), _count_features, _count_items, true);
}

void fast_als::serialize_users(std::ostream& out)
{
	const fast_als::features_vector& users = get_features_users();
	serialize_matrix(out, &users.front(), _count_features, _count_users, true);
}

void fast_als::serialize_matrix(std::ostream& out, const float* mat, int crow, int ccol, bool id)
{
	char* buf = (char*)malloc(10 * sizeof(char));
	for(int i = 0; i < ccol; i++)
	{
		if(id) out << i << "\t";

		for(int j = 0; j < crow;  j++)
		{
			sprintf(buf, "%.1f", mat[i * crow + j]);
			out << buf << (( j == crow-1)? "" : "\t");
		}
		out << std::endl;
	}
	free(buf);
}

void fast_als::init_helper_vectors()
{
	d_user_offsets.push_back(0);
	d_user_sizes.push_back(_user_likes[0].size());

	for (int i = 1; i < _count_users; i++)
	{
		d_user_offsets.push_back(d_user_offsets.back() + d_user_sizes[i - 1]);
		d_user_sizes.push_back(_user_likes[i].size());
	}

	d_item_offsets.push_back(0);
	d_item_sizes.push_back(_item_likes[0].size());

	for (int i = 1; i < _count_items; i++)
	{
		d_item_offsets.push_back(d_item_offsets.back() + d_item_sizes[i - 1]);
		d_item_sizes.push_back(_item_likes[i].size());
	}
}
