#ifndef SIMPLE_NETWORK_H
#define SIMPLE_NETWORK_H

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <boost/python.hpp>
#include <numpy.hpp>

void reseed(int val)
{
	unsigned short seed[3];

	seed[0] = val;
	seed[1] = val + 1;
	seed[2] = val + 2;
	seed48(seed);
}

class Network
{
public:
	int numInput, numHidden, numOutput;
	int numWeights;
	bool useAC, recurrent;
	float *weightArray, *inputUnits, *hiddenUnits, *outputUnits, *recurrentUnits;
	double *activityCounter;
	bool *hiddenMask;

	Network(const int _i, const int _h, const int _o, const bool _useAC = false,
	        const bool _recurrent = false):
		numInput(_i),
		numHidden(_h),
		numOutput(_o),
		useAC(_useAC),
		recurrent(_recurrent)
	{
		if (this->recurrent)
		{
			this->numInput += this->numHidden;
		}
		this->numWeights = (this->numInput + this->numOutput) * this->numHidden;

		this->inputUnits = new float[this->numInput];
		this->hiddenUnits = new float[this->numHidden];
		this->outputUnits = new float[this->numOutput];
		this->recurrentUnits = new float[this->numHidden];

		this->activityCounter = new double[this->numHidden];
		this->hiddenMask = new bool[this->numHidden];
		this->weightArray = new float[this->numWeights];

		for (int i = 0; i < this->numWeights; ++i)
		{
			this->weightArray[i] = 0.0;
		}

		for (int i = 0; i < this->numHidden; ++i)
		{
			this->activityCounter[i] = 0.00001;
			this->hiddenMask[i] = true;
		}
	}

	~Network()
	{
		delete[] this->inputUnits;
		delete[] this->hiddenUnits;
		delete[] this->outputUnits;
		delete[] this->recurrentUnits;

		delete[] this->activityCounter;
		delete[] this->weightArray;
		delete[] this->hiddenMask;
	}

	float inline Sigmoid(const float _input) const
	{
		return 1.0 / (1.0 + exp(-_input));
	}

	float inline TanH(const float _input) const
	{
		return tanh(_input);
	}

	void SetWeightAndActivityCounter(boost::numpy::ndarray &_weights,
	                                 boost::numpy::ndarray &_activity,
	                                 const bool _reverse = false)
	{
		char* arr_data = _weights.get_data();
		int stride = _weights.strides(0);
		for (int i = 0; i < _weights.shape(0); ++i)
		{
			this->weightArray[i] =
			  static_cast<float>(*(reinterpret_cast<double*>(arr_data + i * stride)));
			// std::cout << this->weightArray[i] << " ";
		}
		// std::cout << std::endl;

		arr_data = _activity.get_data();
		stride = _activity.strides(0);
		for (int i = 0; i < _activity.shape(0); ++i)
		{
			this->activityCounter[i] =
			  *(reinterpret_cast<double*>(arr_data + i * stride));
			// std::cout << this->activityCounter[i] << " ";
		}
		// std::cout << std::endl;

		this->SetHiddenMask(_reverse);
	}

	// if reversed is set to true, then most active hidden neurons have highest
	// chance of being enabled, otherwise opposite is true
	void SetHiddenMask(const bool _reverse)
	{
		timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);
		reseed(ts.tv_nsec);

		float sum = 0.0;
		for (int i = 0; i < numHidden; ++i)
		{
			sum += static_cast<float>(this->activityCounter[i]);
		}
		for (int i = 0; i < numHidden; ++i)
		{
			const float prob = static_cast<float>(this->activityCounter[i]) / sum;
			const float r = static_cast <float> (lrand48()) /
			                static_cast <float> (RAND_MAX);

			if (r < prob)
			{
				this->hiddenMask[i] = _reverse;
			}
			else
			{
				this->hiddenMask[i] = !_reverse;
			}
			// std::cout << this->hiddenMask[i] << " ";
		}
		// std::cout << std::endl;
	}

	void UpdateActivityCounter()
	{
		int maxIndex = -1;
		float maxValue = -1.0;
		for (int i = 0; i < this->numHidden; ++i)
		{
			const float temp = fabs(this->hiddenUnits[i]);
			if (temp > maxValue)
			{
				maxIndex = i;
				maxValue = temp;
			}
		}
		this->activityCounter[maxIndex] += 0.00001;
	}

	void Activate()
	{
		float sum;

		for (int i = 0; i < numHidden; ++i)
		{
			if (this->useAC && !this->hiddenMask[i])
			{
				this->hiddenUnits[i] = 0.0;
			}
			else
			{
				sum = 0.0;
				for (int j = 0; j < this->numInput; ++j)
				{
					sum += this->weightArray[j + i * this->numInput] *
					       this->inputUnits[j];
				}
				this->hiddenUnits[i] = this->TanH(sum);
			}
		}

		if (this->recurrent)
		{
			for (int i = (this->numInput - this->numHidden); i < this->numInput; ++i)
			{
				this->inputUnits[i] = this->hiddenUnits[i];
			}
		}

		// we find out which hidden unit is most active
		// (ie it has the largest output value)
		// and increment its counter
		if (this->useAC)
		{
			this->UpdateActivityCounter();
		}

		const int offset = this->numHidden * this->numInput;
		for (int i = 0; i < this->numOutput; ++i)
		{
			sum = 0.0;
			for (int j = 0; j < this->numHidden; ++j)
			{
				sum += this->weightArray[j + (i * this->numHidden) + offset]
				       * this->hiddenUnits[j];
				// std::cout << j + (i * this->numHidden) + offset << std::endl;
			}
			this->outputUnits[i] = this->TanH(sum);
		}
	}
};

#endif