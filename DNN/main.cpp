#include <iostream>
#include <vector>
#include <chrono>

#define numDNN 100
#define numInputs 3
#define numOutputs 1

using std::cout;
using std::endl;
using std::find;
using std::vector;
using std::distance;

using std::chrono::seconds;
using std::chrono::microseconds;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

uint32_t m_z = (uint32_t)duration_cast<seconds>(high_resolution_clock::now().time_since_epoch()).count();
uint32_t m_w = (uint32_t)duration_cast<microseconds>(high_resolution_clock::now().time_since_epoch()).count();

uint32_t intRand()
{
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);

	return (m_z << 16) + m_w;
}

double doubleRand() { return (intRand() + 1.0) * 2.328306435454494e-10; }

class DNN
{
public:
	vector<double> bias;
	vector<double> output;
	vector<double> pastOutput;
	vector<vector<double>> weights;
	vector<vector<int>> connections;
	vector<bool> empty;

	double nonlinear(double x) { return (x > 0) * x; }

	DNN()
	{
		for (int i = 0; i < numInputs + numOutputs; i++)
		{
			output.push_back(0);
			empty.push_back(false);
			pastOutput.push_back(0);
			bias.push_back(doubleRand() * 2 - 1);
		}

		for (int i = 0; i < numInputs; i++)
		{
			weights.push_back({});
			connections.push_back({});

			for (int j = numInputs; j < numInputs + numOutputs; j++)
			{
				connections[i].push_back(j);
				weights[i].push_back(doubleRand() * 2 - 1);
			}
		}

		for (int i = numInputs; i < numInputs + numOutputs; i++)
		{
			weights.push_back({});
			connections.push_back({});

			for (int j = 0; j < numInputs; j++)
			{
				connections[i].push_back(j);
				weights[i].push_back(doubleRand() * 2 - 1);
			}
		}
	}

	void print()
	{
		cout << "output: ";

		for (int n = 0; n < output.size(); n++)
		{
			cout << output[n] << " ";
		}

		cout << endl << endl;

		cout << "pastOutput: ";

		for (int n = 0; n < pastOutput.size(); n++)
		{
			cout << pastOutput[n] << " ";
		}

		cout << endl << endl;

		cout << "empty: ";

		for (int n = 0; n < empty.size(); n++)
		{
			cout << empty[n] << " ";
		}

		cout << endl << endl;

		cout << "bias: ";

		for (int n = 0; n < bias.size(); n++)
		{
			cout << bias[n] << " ";
		}

		cout << endl << endl;

		cout << "weights: " << endl;

		for (int n = 0; n < weights.size(); n++)
		{
			for (int i = 0; i < weights[n].size(); i++)
			{
				cout << weights[n][i] << " ";
			}

			cout << endl;
		}

		cout << endl;

		cout << "connections: " << endl;

		for (int n = 0; n < connections.size(); n++)
		{
			for (int i = 0; i < connections[n].size(); i++)
			{
				cout << connections[n][i] << " ";
			}

			cout << endl;
		}

		cout << endl;
	}

	void resetMemory()
	{
		for (int p = 0; p < bias.size(); p++)
			output[p] = 0;
	}

	void forwardPropagate(double* inputs)
	{
		for (int p = 0; p < bias.size(); p++)
			pastOutput[p] = output[p];

		for (int i = 0; i < numInputs; i++)
			pastOutput[i] += inputs[i];

		for (int p = 0; p < bias.size(); p++)
		{
			output[p] = bias[p];

			for (int c = 0; c < connections[p].size(); c++)
				output[p] += pastOutput[connections[p][c]] * weights[p][c];

			if (!(nonlinear(output[p]) >= 0))
			{
				cout << output[p] << endl;

				for (int c = 0; c < connections[p].size(); c++)
				{
					cout << pastOutput[connections[p][c]] << endl;
					cout << weights[p][c] << endl;
				}
			}
			output[p] = nonlinear(output[p]);
		}
	}

	void mutateData()
	{
		for (int n = 0; n < bias.size(); n++)
		{
			bias[n] += (doubleRand() * 2 - 1) * 1;

			for (int i = 0; i < weights[n].size(); i++)
				weights[n][i] += (doubleRand() * 2 - 1) * (doubleRand() * 0.1 + 0.01);
		}
	}

	void addNode()
	{
		int node1 = doubleRand() * bias.size();
		int connection = doubleRand() * connections[node1].size();
		vector<bool>::iterator findNode2 = find(empty.begin(), empty.end(), true);

		if (findNode2 == empty.end())
		{
			bias.push_back(0);
			output.push_back(0);
			pastOutput.push_back(0);
			weights.push_back({ 1 });
			connections.push_back({ connections[node1][connection] });
			empty.push_back(false);
			connections[node1][connection] = bias.size() - 1;
		}
		else
		{
			int node2 = distance(empty.begin(), findNode2);
			bias[node2] = 0;
			output[node2] = 0;
			pastOutput[node2] = 0;
			weights[node2] = { 1 };
			connections[node2] = { connections[node1][connection] };
			empty[node2] = false;
			connections[node1][connection] = node2;
		}
	}

	void addConnection()
	{
		int node1 = doubleRand() * bias.size();
		int node2 = doubleRand() * bias.size();
		vector<int>::iterator findNode2 = find(connections[node1].begin(), connections[node1].end(), node2);

		if (findNode2 == connections[node1].end())
		{
			connections[node1].push_back(node2);
			weights[node1].push_back(1);
		}
	}

	void getReverseStructure(vector<vector<int>>* reverseConnections)
	{
		for (int n = 0; n < bias.size(); n++)
			reverseConnections->push_back({});

		for (int n = 0; n < bias.size(); n++)
			for (int i = 0; i < connections[n].size(); i++)
				(*reverseConnections)[connections[n][i]].push_back(n);
	}

	void deleteNode()
	{
		if (bias.size() > numInputs + numOutputs)
		{
			vector<vector<int>> reverseConnections;
			getReverseStructure(&reverseConnections);

			int node = doubleRand() * (bias.size() - numInputs - numOutputs) + numInputs + numOutputs;
			empty[node] = true;

			for (int c = 0; c < reverseConnections[node].size(); c++)
			{
				int node1 = reverseConnections[node][c];
				vector<int>::iterator findNode2 = find(connections[node1].begin(), connections[node1].end(), node);
				int connection = distance(connections[node1].begin(), findNode2);

				connections[node1].erase(connections[node1].begin() + connection);
				weights[node1].erase(weights[node1].begin() + connection);

				for (int p = 0; p < connections[node].size(); p++)
				{
					int node2 = connections[node][p];
					vector<int>::iterator findNode2 = find(connections[node1].begin(), connections[node1].end(), node2);

					if (findNode2 == connections[node1].end())
					{
						connections[node1].push_back(node2);
						weights[node1].push_back(1);
					}
				}
			}
		}
	}

	int numberOfNodesReached(int givenNode, int givenDeletedConnection, bool* givenReached)
	{
		if (givenReached[givenNode] == true) { return 0; }

		int sum = 1;
		givenReached[givenNode] = true;

		for (int i = 0; i < connections[givenNode].size(); i++)
			if (i != givenDeletedConnection)
				sum += numberOfNodesReached(connections[givenNode][i], -1, givenReached);

		return sum;
	}

	void deleteConnection()
	{
		bool* reached = new bool[bias.size()]{};
		int node1 = doubleRand() * bias.size();
		int connection = doubleRand() * connections[node1].size();

		if (numberOfNodesReached(node1, connection, reached) == bias.size())
		{
			weights[node1].erase(weights[node1].begin() + connection);
			connections[node1].erase(connections[node1].begin() + connection);
		}

		delete[] reached;
	}

	void mutateStructure()
	{
		if (doubleRand() > 0.9)
			switch (int(doubleRand() * 4))
			{
			case 0: addNode(); break;
			case 1: addConnection(); break;
			case 2: deleteNode(); break;
			case 3: deleteConnection(); break;
			}
	}
};

int main()
{
	DNN agents[numDNN];
	double score[numDNN]{};
	bool empty[numDNN]{};
	double input[numInputs]{};

	for (int i = 0; i < numDNN; i++)
	{
		agents[i] = DNN();
	}

	while (true)
	{
		for (int i = 0; i < numDNN; i++)
		{
			score[i] = 0;
			empty[i] = false;
			agents[i].resetMemory();
		}

		for (int i = 0; i < 100; i++)
		{
			double sum = 0;

			for (int j = 0; j < numInputs; j++)
			{
				input[j] = doubleRand() * 2 - 1;
				sum += input[j];
			}

			for (int t = 0; t < 100; t++)
			{
				for (int a = 0; a < numDNN; a++)
				{
					agents[a].forwardPropagate(input);
				}
			}

			for (int a = 0; a < numDNN; a++)
			{
				score[a] = abs(agents[a].output[numInputs] - sum);
			}
		}

		for (int i = 0; i < 50; i++)
		{
			int minIndex = 0;
			int minScore = 0;

			for (int j = 0; j < numDNN; j++)
			{
				if (!empty[j] && score[j] > minScore)
				{
					minIndex = j;
					minScore = score[j];
				}
			}

			empty[minIndex] = true;
			int random = int(doubleRand() * (numDNN - 1));
			agents[minIndex] = agents[random + (random >= minIndex)];
			agents[minIndex].mutateData();
			agents[minIndex].mutateStructure();
		}

		int maxIndex = 0;
		int maxScore = 100000;
		for (int i = 0; i < numDNN; i++)
		{
			if (score[i] < maxScore)
			{
				maxIndex = i;
				maxScore = score[i];
			}
		}

		agents[maxIndex].print();
	}
}