#pragma once

#include <numeric>
#include <vector>

template<class T>
struct TensorView {
public:
	explicit TensorView(const std::size_t &size)
		: size_(size),
		  owner_ship_(true) {
		data_ = new T[size];
	}

	TensorView(T *data, const std::size_t &size)
		: data_(data),
		  size_(size),
		  owner_ship_(false) {
	}

	template<class S>
	TensorView(T *data, const std::vector<S> &sizes)
		: data_(data),
		  owner_ship_(false) {
		size_ = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<S>());
	}

	TensorView(const TensorView &rhs)
		: size_(rhs.size_),
		  owner_ship_(rhs.owner_ship_) {
		if (owner_ship_) {
			data_ = new T[rhs.size_];
			memcpy(data_, rhs.data_, rhs.size_ * sizeof(T));
			owner_ship_ = true;
			return;
		}

		data_ = rhs.data_;
	}

	TensorView(TensorView &&rhs) noexcept
		: data_(rhs.data_),
		  size_(rhs.size_),
		  owner_ship_(rhs.owner_ship_) {
		rhs.owner_ship_ = false;
		rhs.data_ = nullptr;
		rhs.size_ = 0;
	}

	T *data() { return data_; }

	const T *data() const { return data_; }


	~TensorView() {
		if (owner_ship_)
			delete[] data_;
	}

	T *data_ = nullptr;       //< 数据指针
	std::size_t size_ = 0;    //< 数据大小
	bool owner_ship_ = false; //< 数据所有权
};
