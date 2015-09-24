#include <cmath>
#include <algorithm>
#include <set>
#include <map>
#include <ctime>
#include <thrust/functional.h>
#include <omp.h>

#include "fast_als.cuh"

#define BLOCK_SIZE 32
#define COUNT_ROWS_START 10

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
		int count_error_samples_for_items,
		int count_gpus) :
		_count_users(0),
		_count_items(0),
		_count_features(count_features),
		_als_alfa(alfa),
		_als_gamma(gamma),
		_count_error_samples_for_users(count_error_samples_for_users),
		_count_error_samples_for_items(count_error_samples_for_items),
		_count_samples(count_samples),
		_count_gpus(count_gpus)
{
	int cout_dev=0;
	cudaGetDeviceCount(&cout_dev);
	for (int i = 0; i < cout_dev; i++)
	{
		std::cerr << "=== CUDA Device: " <<  i << std::endl;
		cudaGetDeviceProperties(&prop, i);
		std::cerr << "Cuda Device: " << prop.name << std::endl;
		std::cerr << "Total gloabl mem: " << prop.totalGlobalMem << std::endl;
		std::cerr << "Total shared mem per block: " << prop.sharedMemPerBlock << std::endl;
		std::cerr << "Multi processors: " << prop.multiProcessorCount << std::endl;
		std::cerr << "Warp size: " <<  prop.warpSize << std::endl;
		std::cerr << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
		std::cerr << "maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
	}

	if (_count_gpus == 0 || _count_gpus > cout_dev)
	{
		_count_gpus = cout_dev;
	}

//	srand(time(NULL));
	srand(34);

	read_likes(tuples_stream, count_samples, likes_format);

	//generate_test_set();

	_features_users.assign(_count_users * _count_features, 0 );
	_features_items.assign(_count_items * _count_features, 0 );
	YxY.assign(_count_features * _count_features, 0);
}

fast_als::~fast_als()
{

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
			weight = weight_temp;
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
	time_t start =  time(0);

	for (int i = 0; i < in_size * _count_features; i++)
	{
		in_v[i] = ((float)rand()/(float)RAND_MAX);
	}

	start = time(0) - start;
	std::cerr << "done. time: " << start << std::endl;
}

void fast_als::calculate(int count_iterations)
{
	fill_rnd(_features_users, _count_users);
	init_helper_vectors();

	if (_count_gpus > 1)
	{
		calculate_multiple_gpus(count_iterations);
	}
	else
	{
		calculate_one_gpu(count_iterations);
	}
}

void fast_als::calculate_one_gpu(int count_iterations)
{
	std::ofstream hr10("hr10.txt");
	cudaSetDevice(0);

	for (int i = 0; i < count_iterations; i++)
	{
		time_t start =  time(0);
		std::cerr << "ALS Iteration: " << i << std::endl;

		std::cerr << "Items." << std::endl;
		solve(_item_likes.begin(), _item_likes_weights.begin(), _features_users, _count_users, _features_items, _count_items,
						_count_features, d_item_offsets);

		std::cerr << "Users." << std::endl;
		solve(_user_likes.begin(), _user_likes_weights.begin(), _features_items, _count_items, _features_users, _count_users,
						_count_features, d_user_offsets);

		time_t end =  time(0);
		std::cerr << "==== Iteration time : " << end - start << std::endl;

		//MSE();
		//hr10 << hit_rate() << std::endl;
	}

	hr10.close();
}

void fast_als::calculate_multiple_gpus(int count_iterations)
{
	int _count_features_first_part = _count_features / _count_gpus;
	int _count_features_last_part = _count_features - _count_features_first_part * (_count_gpus - 1);

	int _count_items_first_part = _count_items / _count_gpus;
	int _count_items_last_part = _count_items - _count_items_first_part * (_count_gpus - 1);

	int _count_users_first_part = _count_users / _count_gpus;
	int _count_users_last_part = _count_users - _count_users_first_part * (_count_gpus - 1);

	std::vector<int> _count_features_parts(_count_gpus, _count_features_first_part);
	_count_features_parts.back() = _count_features_last_part;

	std::vector<int> _count_items_parts(_count_gpus, _count_items_first_part);
	_count_items_parts.back() = _count_items_last_part;

	std::vector<int> _count_users_parts(_count_gpus, _count_users_first_part);
	_count_users_parts.back() = _count_users_last_part;

	std::vector<int> features_offsets(_count_gpus, 0);
	std::vector<int> items_offsets(_count_gpus, 0);
	std::vector<int> users_offsets(_count_gpus, 0);

	for (int i = 1; i < _count_gpus; i++)
	{
		features_offsets[i] = features_offsets[i - 1] + _count_features_parts[i - 1];
		items_offsets[i] = items_offsets[i - 1] + _count_items_parts[i - 1];
		users_offsets[i] = users_offsets[i - 1] + _count_users_parts[i - 1];
	}

	omp_set_dynamic(0);
	omp_set_num_threads(_count_gpus);

	std::ofstream hr10("hr10.txt");

	for (int i = 0; i < count_iterations; i++)
	{
		time_t start =  time(0);
		std::cerr << "ALS Iteration: " << i << std::endl;

		//Items
		#pragma omp parallel num_threads(_count_gpus)
		{
			int thread_id = omp_get_thread_num();
			int gpu_id;
			cudaSetDevice(thread_id);
			cudaGetDevice(&gpu_id);
			std::cerr << "Items. Thread: " << thread_id << " GPU: " << gpu_id << std::endl;

			solve(_item_likes.begin() + items_offsets[thread_id], _item_likes_weights.begin() + items_offsets[thread_id], _features_users, _count_users,
					_features_items, _count_items_parts[thread_id], _count_features_parts[thread_id], d_item_offsets,
					features_offsets[thread_id], items_offsets[thread_id]);
		}


		cudaDeviceSynchronize();
		// Users
		#pragma omp parallel num_threads(_count_gpus)
		{
			int thread_id = omp_get_thread_num();
			int gpu_id;
			cudaSetDevice(thread_id);
			cudaGetDevice(&gpu_id);
			std::cerr << "Users. Thread: " << thread_id << " GPU: " << gpu_id << std::endl;

			solve(_user_likes.begin() + users_offsets[thread_id], _user_likes_weights.begin() + users_offsets[thread_id], _features_items, _count_items,
					_features_users, _count_users_parts[thread_id], _count_features_parts[thread_id], d_user_offsets,
					features_offsets[thread_id], users_offsets[thread_id]);
		}

		cudaDeviceSynchronize();

		time_t end =  time(0);
		std::cerr << "==== Iteration time : " << end - start << std::endl;

		//MSE();
		//hr10 << hit_rate() << std::endl;
	}
	hr10.close();
}


void fast_als::solve(
		const likes_vector::const_iterator& likes,
		const likes_weights_vector::const_iterator& weights,
		const features_vector& in_v,
		int in_size,
		features_vector& out_v,
		int out_size,
		int _count_features_local,
		std::vector<int>& likes_offsets,
		int features_local_offset,
		int out_offset)
{
	culaStatus cula_status = culaInitialize();
	checkStatus(cula_status);
	cublasHandle_t cublas_handle;
	cublasStatus_t cublas_status(cublasCreate(&cublas_handle));

	time_t start = time(0);
	fast_als::features_vector g = calc_g(in_v, in_size, cublas_handle, cublas_status, cula_status,
			_count_features_local, features_local_offset);
//	fast_als::features_vector g(_count_features * _count_features);
//	fill_rnd(g, _count_features);
	cudaDeviceSynchronize();
	start = time(0) - start;
	std::cerr << "g calc: " << start << std::endl;

	if ( cudaSuccess != cudaPeekAtLastError() )
				std::cerr <<  "!WARN - Cuda error (g calc) : "  << cudaGetErrorString(cudaGetLastError()) << std::endl;

	start = time(0);
	calc_ridge_regression_gpu(likes, weights, in_v, out_v, out_size, g, likes_offsets, out_offset);
	start = time(0) - start;
	std::cerr << "regression calc: " << start << std::endl;

	culaShutdown();
	cublas_status = cublasDestroy(cublas_handle);

}

#define RESERVED_MEM 0xA00000

void fast_als::mulYxY(const features_vector& in_v, int in_size,
		cublasHandle_t& cublas_handle, cublasStatus_t& cublas_status, int _count_features_local, int features_local_offset)
{
	thrust::device_vector<float> device_YxY(_count_features * _count_features_local, 0);
	float alpha = 1;
	float beta = 1;
	///
	/// Calculate size of block for input matrix
	/// input matrix is Y matrix
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

		size_t offset = part * _count_features * count_rows;
		thrust::copy(in_v.begin()+ offset,  in_v.begin()+ offset + actual_part_size * _count_features, x_device.begin());


		cublas_status = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, _count_features, _count_features_local, actual_part_size , &alpha,
											 thrust::raw_pointer_cast(&x_device[0]), _count_features,
											 thrust::raw_pointer_cast(&x_device[0])+ features_local_offset,
											 _count_features, &beta, thrust::raw_pointer_cast(&device_YxY[0]),
											 _count_features);

		if ( cublas_status != 0 )
			std::cerr <<  "!WARN - Cuda error (als::mulYxY -> cublasSgemm) : "  << cublas_status << std::endl;
	}

	thrust::copy(device_YxY.begin(), device_YxY.end(), YxY.begin() + _count_features * features_local_offset);
}

fast_als::features_vector fast_als::calc_g(const features_vector& in_v, int in_size,
		cublasHandle_t& cublas_handle, cublasStatus_t& cublas_status, culaStatus& cula_status,
		int _count_features_local, int features_local_offset)
{
	std::vector<float> U(_count_features * _count_features);
	std::vector<float> G(_count_features * _count_features);
	std::vector<float> S(_count_features);

	mulYxY(in_v, in_size, cublas_handle, cublas_status, _count_features_local, features_local_offset);

	cudaDeviceSynchronize();
	#pragma omp barrier

	cula_status = culaSgesvd('N', 'S', _count_features, _count_features, &YxY[0], _count_features, &S[0], NULL,
			_count_features, &U[0], _count_features);
	checkStatus(cula_status);

	std::vector<float> lam_sqrt(_count_features * _count_features, 0.0);

	for (int i = 0; i < _count_features; i++)
	{
		lam_sqrt[i * _count_features + i] = sqrt(S[i]);
	}

	cula_status = culaSgemm('N', 'N', _count_features, _count_features, _count_features, 1.0f, &lam_sqrt[0], _count_features,
			&U[0], _count_features, 0.0f, &G[0], _count_features);
	checkStatus(cula_status);

	return G;
}

__global__ void ridge_regression_kernel(const float* weights, const float* in_v, float* out_v, int out_size,
		int _count_features, float* g, int* likes_offsets, float _als_alfa, float _als_gamma, float* errors)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < out_size)
	{
		int id = x;
		int in_offset = likes_offsets[id];
		int in_size = likes_offsets[id + 1] - in_offset;
		int error_offset = in_offset + id * _count_features;
		int error_g_offset = error_offset + in_size;
		int out_offset = id * _count_features;
		int in_offset_features = in_offset * _count_features;

		for (int i = 0; i < in_size; i++)
		{
			int in_cur_off = in_offset_features + i * _count_features;
			float sum = 0;
			for (int j = 0; j < _count_features; j++)
			{
				sum += out_v[out_offset + j] * in_v[in_cur_off + j];
			}
			float c = 1 + _als_alfa * weights[in_offset + i];
			errors[error_offset + i] = (c / (c - 1)) - sum;
		}

		for (int i = 0; i < _count_features; i++)
		{
			float sum = 0;
			for (int j = 0; j < _count_features; j++)
			{
				sum += out_v[out_offset + j] * g[j * _count_features + i];
			}
			errors[error_g_offset + i] = -sum;
		}

		for (int k = 0; k < _count_features; k++)
		{
			float out_v_cur = out_v[out_offset + k];

			float a = 0;
			float d = 0;
			for (int i = 0; i < in_size; i++)
			{
				float c = _als_alfa * weights[in_offset + i];
				float in_v_cur =  in_v[in_offset_features + i * _count_features + k];
				a += c * in_v_cur * in_v_cur;
				d += c * in_v_cur * (errors[error_offset + i] + out_v_cur * in_v_cur);
			}
			for (int i = 0; i < _count_features; i++)
			{
				float g_cur = g[k * _count_features + i];
				a += g_cur * g_cur;
				d += g_cur * (errors[error_g_offset + i] + out_v_cur * g_cur);
			}

			float out_v_cur_new = d / (_als_gamma + a);

			out_v[out_offset + k] = out_v_cur_new;

			float out_diff = out_v_cur - out_v_cur_new;

			for (int i = 0; i < in_size; i++)
			{
				errors[error_offset + i] += out_diff * in_v[in_offset_features + i * _count_features + k];
			}
			for (int i = 0; i < _count_features; i++)
			{
				errors[error_g_offset + i] += out_diff * g[k * _count_features + i];
			}
		}
	}
}

void fast_als::calc_ridge_regression_gpu(
		const likes_vector::const_iterator& likes,
		const likes_weights_vector::const_iterator& weights,
		const features_vector& in_v,
		features_vector& out_v,
		int out_size,
		features_vector& g,
		std::vector<int>& likes_offsets,
		int out_offset)
{
	d_features_vector d_g(g);

	time_t start = time(0);

	int count_rows = COUNT_ROWS_START;
//	int count_rows = 1000;
	count_rows = count_rows >= out_size ? out_size : count_rows;
	int parts_size = out_size / count_rows + ((out_size % count_rows != 0) ? 1 : 0);

	if (count_rows != out_size)
	{
		std::cout << "out size: " << out_size << std::endl;
		float used_mem = 0;
		int prev_count_rows = count_rows;
		size_t cuda_free_mem = 0;
		size_t cuda_total_mem = 0;
		cudaMemGetInfo(&cuda_free_mem, &cuda_total_mem);
		while ((used_mem < cuda_free_mem * 1.0) && (count_rows < out_size))
		{
			prev_count_rows = count_rows;
			count_rows *= 2;
			count_rows = count_rows >= out_size ? out_size : count_rows;
			parts_size = out_size / count_rows + ((out_size % count_rows != 0) ? 1 : 0);
			float max_sum = 0;
			for (int i = 0; i < parts_size; i++)
			{
				int actual_part_size = (i == parts_size - 1 && out_size % count_rows != 0) ? out_size % count_rows : count_rows;
				float sum = likes_offsets[out_offset + i * count_rows + actual_part_size] - likes_offsets[out_offset + i * count_rows];
				if (sum > max_sum)
				{
					max_sum = sum;
				}
			}
			used_mem = (max_sum * (2 + _count_features) + count_rows * (_count_features * 2 + 1) + _count_features * _count_features) * 4.0;
			/*std::cout << "free: " << cuda_free_mem / 1024.0 / 1024.0 << " used: " << used_mem / 1024.0 / 1024.0 << " count_rows: "
								<< count_rows << std::endl;*/
		}
		if (used_mem >= cuda_free_mem * 1.0)
		{
			count_rows = prev_count_rows;
		}
		count_rows = count_rows >= out_size ? out_size : count_rows;
		parts_size = out_size / count_rows + ((out_size % count_rows != 0) ? 1 : 0);
	}


	start = time(0) - start;

	std::cout << "calc size time: " << start << std::endl;
	std::cout << "count_rows: " << count_rows << std::endl;


	time_t prepare = 0;
	time_t kernel = 0;

	for(int part = 0; part < parts_size; part++)
	{
		cudaDeviceSynchronize();
		time_t start = time(0);

		int actual_part_size = (part == parts_size - 1 && out_size % count_rows != 0) ? out_size % count_rows : count_rows;
		thrust::device_vector<float> d_weights;
		thrust::device_vector<int> d_likes_offsets(likes_offsets.begin() + out_offset + part * count_rows,
				likes_offsets.begin() + out_offset + part * count_rows + actual_part_size + 1);
		int sub = *(likes_offsets.begin() + out_offset + part * count_rows);
		thrust::for_each(d_likes_offsets.begin(), d_likes_offsets.end(), thrust::placeholders::_1 -= sub);

		std::vector<float> h_weights;
		features_vector h_in_v;

		size_t offset = out_offset * _count_features + part * _count_features * count_rows;
		d_features_vector d_in_v;
		d_features_vector d_out_v(out_v.begin() + offset, out_v.begin() + offset + actual_part_size * _count_features);


		int err_size = 0;
		for (int i = 0; i < actual_part_size; i++)
		{
			h_weights.insert(h_weights.end(), (weights + part * count_rows + i)->begin(), (weights + part * count_rows + i)->end());
			err_size += (*(likes + part * count_rows + i)).size();
			for (int j = 0; j < (*(likes + part * count_rows + i)).size(); j++)
			{
				int id = (*(likes + part * count_rows + i))[j];
				h_in_v.insert(h_in_v.end(), in_v.begin() + id * _count_features, in_v.begin() + (id + 1) * _count_features);
			}
		}


		d_weights = h_weights;
		d_in_v = h_in_v;



		cudaDeviceSynchronize();
		start = time(0) - start;
		prepare += start;
		start = time(0);
		if ( cudaSuccess != cudaPeekAtLastError() )
					std::cerr <<  "!WARN - Cuda error (thrust in regression) : "  << cudaGetErrorString(cudaGetLastError()) << std::endl;

		thrust::device_vector<float> errors(err_size + _count_features * actual_part_size);

		dim3 block(BLOCK_SIZE, 1);
		dim3 grid(1 + actual_part_size / BLOCK_SIZE, 1);


		float size_mbytes = ( err_size * (2 + _count_features ) + actual_part_size * (2 * _count_features + 1)  + _count_features * _count_features)
				* 4.0 / 1024 / 1024;
		std::cout << "*****GPU data size in MB: " << size_mbytes << std::endl;

		size_t cuda_free_mem = 0;
		size_t cuda_total_mem = 0;
		cudaMemGetInfo(&cuda_free_mem, &cuda_total_mem);
		std::cerr << "Cuda memory used in MB: " << (cuda_total_mem - cuda_free_mem) / 1024.0 / 1024.0 << std::endl;




		ridge_regression_kernel<<<grid, block>>>(thrust::raw_pointer_cast(&d_weights[0]),
				thrust::raw_pointer_cast(&d_in_v[0]), thrust::raw_pointer_cast(&d_out_v[0]), actual_part_size, _count_features,
				thrust::raw_pointer_cast(&d_g[0]), thrust::raw_pointer_cast(&d_likes_offsets[0]),
				_als_alfa, _als_gamma, thrust::raw_pointer_cast(&errors[0]));

		cudaDeviceSynchronize();
		start = time(0) - start;
		kernel += start;
		start = time(0);
		cudaError_t lastErr = cudaGetLastError();
		if (lastErr != cudaSuccess)
		{
			std::cout << "cuda error in kernel! " << cudaGetErrorString(lastErr) <<  std::endl;
		}

		thrust::copy(d_out_v.begin(), d_out_v.end(), out_v.begin() + offset);
		cudaDeviceSynchronize();
		start = time(0) - start;
		prepare += start;
		if ( cudaSuccess != cudaPeekAtLastError() )
							std::cerr <<  "!WARN - Cuda error (thrust in regression 2) : "  << cudaGetErrorString(cudaGetLastError()) << std::endl;
	}
	std::cout << "prepare time, s: " << prepare << std::endl;
	std::cout << "kernel time, s: " << kernel << std::endl;
}

void fast_als::MSE()
{

}

float fast_als::hit_rate()
{
	std::vector<float> predict(_count_users * _count_items);

	//predict = P * Q^t
	//predict = P^t * Q
	culaStatus cula_status = culaInitialize();
	checkStatus(cula_status);
	cula_status = culaSgemm('T', 'N', _count_users, _count_items, _count_features, 1.0f, &_features_users[0], _count_features,
			&_features_items[0], _count_features, 0.0f, &predict[0], _count_users);
	checkStatus(cula_status);
	culaShutdown();

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
	int size = _user_likes[0].size();
	for (int i = 1; i < _count_users; i++)
	{
		d_user_offsets.push_back(d_user_offsets.back() + size);
		size = _user_likes[i].size();
	}
	d_user_offsets.push_back(d_user_offsets.back() + size);

	d_item_offsets.push_back(0);
	size = _item_likes[0].size();
	for (int i = 1; i < _count_items; i++)
	{
		d_item_offsets.push_back(d_item_offsets.back() + size);
		size = _item_likes[i].size();
	}
	d_item_offsets.push_back(d_item_offsets.back() + size);
}
