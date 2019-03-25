#include "mex.h"
#include "matrix.h"


static std::vector<RootNode> rn_instances = {};
static std::vector<StructuredTensor> st_instances = {};


// Handle RootNode class functions
void rnFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{

}

// Handle StructuredTensor class functions
void stFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{

}

// Main entry point of mex program
// Args:
//  nlhs: Number of outputs (left hand side)
//  plhs: Outputs
//  nrhs: Number of inputs (right hand side)
//  prhs: Inputs
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  // First input is always the class identifier as string
  mxArray* class_name_mx = prhs[0];
  size_t n_chars = mxGetN(fcn_name_mx);
  char class_name[n_chars+1];
  mxGetString(class_name_mx, class_name);

  if (strcmp(class_name,"ST") == 0) {
    stFunction(nlhs, plhs, nrhs-1, prhs++);
  } else if (strcmp(class_name,"RN") == 0) {
    rnFunction(nlhs, plhs, nrhs-1, prhs++);
  }
}
