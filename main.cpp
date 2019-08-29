#include "network.h"

#include <fstream>
#include <math.h>

#include <parameters.h>
#include <utilities.h>
#include <threadpool.h>

parameters *p;   /* The global parameter pointer */
int verbose;

extern void queue_data( void * );
int run_over_data( network * );

int main ( int argc, char ** argv ) {

  randomize();

  p = new parameters("network.rcp");
  verbose = p->getInt("VERBOSE");

  network * n = new network();

  /*********************************** matt mazur structure test
  float vec[2] = {0.05,0.10};
  float target[2] = {0.01, 0.99};
  float error = 0.0f;

  for ( uint j=0; j<p->getUInt("TRAINING_CYCLES"); j++ ) {
    error = n->train( vec, target );
    cout << "network error (cycle " << j+1 << ") = " << error << "\n\n";
  }
  n->dump_weights();

  return 0;
 ***********************************/


  if ( p->getBool("TRAIN") )
    return n->train_network();

  /*
  if ( p->getBool("NORMALIZE") ) {
    int return_value = n->normalize(p->getString("INPUT_FILE"));
    if ( return_value )
      return return_value;
  }
  */
  n->set_weights();
  n->dump_weights();

  float vec[4][2] = { {0.05,0.10},
		      {0.10,0.15},
		      {0.15,0.20},
		      {0.20,0.25}};

  threadpool * pool = new threadpool(&queue_data,2,verbose);

  pool->enqueue( (void *)vec[0] );
  pool->enqueue( (void *)vec[1] );
  pool->enqueue( (void *)vec[2] );
  pool->enqueue( (void *)vec[3] );

  //pool->dump_queue();
  pool->wait_until_empty();

  //run_over_data(n);

  if ( n )
    delete n;
  if ( pool )
    delete pool;

  return 0;
}

int run_over_data(network *n) {

  const int number_of_inputs  = (const int) n->getNumberOfInputs();
  const int number_of_outputs = (const int) n->getNumberOfOutputs();

  ifstream infile;
  infile.open(p->getString("INPUT_FILE"));
  if ( !infile.is_open() ) {
    cerr << "Unable to open input file " << p->getString("INPUT_FILE") << "\n";
    return -1;
  }

  string line;
  string delimiter(" ");
  float inVec[number_of_inputs];
  float * result = new float [number_of_outputs];

  while ( getline( infile, line ) ) {
    size_t pos = 0;
    string token;
    int counter = 0;
 
    while ((pos = line.find(delimiter)) != std::string::npos) {
      token = line.substr(0, pos);

      if ( counter < number_of_inputs )
	inVec[counter] = atof(token.c_str());
      counter++;
      line.erase(0, pos + delimiter.length());
    }
    if ( counter < number_of_inputs )
      inVec[counter] = atof(line.c_str());

    float * result = n->input(inVec);

    cout << "NN(";
    for (int i=0; i<number_of_inputs; i++)
      cout << inVec[i] << ", ";
    cout << "\b\b) = (";
    for ( int i=0; i<number_of_outputs; i++ ) 
      cout << result[i] << ", ";
    cout << "\b\b)\n";
  }

  delete [] result;

  return 0;
}
