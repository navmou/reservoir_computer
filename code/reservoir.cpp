#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>


using namespace std;

struct timeval tv;

typedef vector<double> oneDvec;
typedef vector<vector<double>> twoDvec;
typedef vector<vector<vector<double>>> triDvec;


oneDvec generate_trajectory()
{
  //setting constant values as Lorenz paper
  double b=8.0/3.0 , sigma=10.0 , r=28.0;
  oneDvec traj;
  int T = 50.0;
  double dt = 0.02 , t = 0.0;

  double x0 , y0 , z0 , x , y , z;
  //initial conditions are chosen to be on the attractor
  x0 = -1.76015 ; y0 = -2.87819 ; z0 = 15.7505;
  
  while (t <= T) {
    x = x0 + dt*sigma*(-x0+y0);
    y = y0 + dt*(-x0*z0 + r*x0 - y0);
    z = z0 + dt*(x0*y0 - b*z0);
    traj.push_back(y0);
    x0 = x ; y0 = y ; z0 = z;

    t+=dt;
  }
  return traj;
}


twoDvec get_adjacency(int n_neurons , double P)
{
  twoDvec adjacency;
  oneDvec from;
  for (int i = 0; i < n_neurons; ++i)
    {
      from.clear();
      for (int j = 0 ; j < n_neurons; ++j)
	{
	  if (i == j)
	    {
	      from.push_back(0.0);
	    }
	  else
	    {
	      if(rand()/(double) RAND_MAX < P){from.push_back(1.0);}
	      else{from.push_back(0.0);}
	    }
	}

    }

  return adjacency;
}



int main(){
  oneDvec traj , training_set , test_set;
  double P = 0.4;
  int n_neurons = 100, n_in = 20 , n_out = 50;
  //generating a trajectory using Lorenz model
  traj = generate_trajectory();
  //using 80% of the trajectory for training set and remaining 20% for test set.
  for (int i = 0 ; i < 2000; ++i) {
    training_set.push_back(traj[i]);
  }
  for (int i = 2000; i <= int(traj.size()); ++i) {
    test_set.push_back(traj[i]);
  }

  twoDvec adjacency = get_adjacency(n_neurons , P);  
  //Choosing the neurons that are connected to input
  int inputs[n_in];
  for (int i = 0; i < n_in; ++i) {
    inputs[i] = (rand()%50);
  }
  //Choosing the neurons connected to output
  int outputs[n_out];
  int neuron;
  for (int i = 0; i < n_out; ++i) {
    neuron = rand()%100;
    for (int j = 0 ; j < n_in; ++j) {
      if(neuron == inputs[j])
	{

	}
    }

    outputs.push_back();
  }


  





  ofstream file;
  file.open("traj1.txt");
  for (int i = 0; i < int(training_set.size()); ++i) {
    file << training_set[i] << endl;
  }
  file.close();


  
  return 0;
}
