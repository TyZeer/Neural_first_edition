#include "NeuralNetwork.h"
#include "NeuralNetworkParametres.h"
#include "NNData.h" 
#include <time.h>
#include <math.h>
#include <vector>
#include<iomanip>

	

int main()
{
	NNParametres config;
	NeuralNetwork network(config);
	network.ShowNeuralNetworkParametres(std::cout);
	NNData Dataset;
	Dataset.Read("C:/Users/ִלטענטי/source/repos/Neural/x64/Debug/Folder_train/optdigits.tra");
	Dataset.Prepare();
	srand(time(NULL));
	std::vector<int > neur_vect{ 64,50,50,10 };
	for (size_t i = 1; i < 4; i++)
	{
		network.InitSynapseInLayer(i, [&]()->float {return( static_cast <float> (rand()) / static_cast <float> (RAND_MAX) * sqrt(2.0f / neur_vect[i - 1])); });
	}

	for (size_t i = 0; i < 64; i++)
	{
		network.SetInputNeuron(i, Dataset.icon_vect_[0].data_[i]);
	}
	network.Calculate([](float val)->float {return 1.0F / (1.0f + (float)pow(M_E, val * -1.0f)); });
	for (int i = 0; i < 10; i++)
	{
		if (i == Dataset.icon_vect_[i].value_)
			network.CalcOutputError(i, 1.0f, [](float val)->float {return val * (1.0F - val); });
		else
			network.CalcOutputError(i, 0.0f, [](float val)->float {return val * (1.0F - val); });
	}
	for (size_t i = 0; i < Dataset.icon_vect_.size(); i++)
	{
		for (size_t j = 0; j < 64; j++)
		{
			network.SetInputNeuron(j, Dataset.icon_vect_[i].data_[j]);
		}
		network.Calculate([](float val)->float {return 1.0F / (1.0f + (float)pow(M_E, val * -1.0f)); });
		for (int k = 0; k < 10; k++)
		{
			if (k == Dataset.icon_vect_[k].value_)
				network.CalcOutputError(k, 1.0f, [](float val)->float {return val * (1.0F - val); });
			else
				network.CalcOutputError(k, 0.0f, [](float val)->float {return val * (1.0F - val); });

		}
		if (i % 200 == 0)
			std::cout << "MSE = " << std::setprecision(6) << network.MediumSquareError() << std::endl;
		network.BackPropogationError([](float val)->float {return val * (1.0F - val); });

	}
	network.ShowOutputNeuronValue(std::cout, 10);



}