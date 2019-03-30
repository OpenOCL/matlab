//
// MIT License
// Copyright (c) 2019 Jonas Koenemann

#include <vector>
#include <iostream>
#include <cassert>
#include <map>

#include "mex.h"
#include "matrix.h"

#include "ocl/tensor/tensor.h"

static std::vector<Ocl::Tensor> tensor_storage;
static int num_tensors = 0;

bool isFunctionHandle(const mxArray* in_arr) {
  return mxIsClass(in_arr,"function_handle");
}

void display(const std::string str)
{
  mexPrintf("%s\n", str.c_str());
}

void mexError(std::string str)
{
  mexErrMsgIdAndTxt("casadi_mex:mexError", str.c_str());
}

void mexAssert(bool cond, std::string str)
{
  if (!cond)
  {
    mexError(str);
  }
}

std::string mxToString(const mxArray* in_arr)
{
  mexAssert(mxIsChar(in_arr), "Argument must be string.");

  size_t n_chars = mxGetN(in_arr);
  char fcn_name[n_chars+1];
  mxGetString(in_arr, fcn_name, n_chars+1);
  return std::string(fcn_name);
}

std::vector<double> mxToNumericVec(const mxArray* in_arr)
{
  mexAssert(mxIsDouble(in_arr) && !mxIsComplex(in_arr), "Invalid convertion from mx type to vector.");

  int nv = (int) mxGetNumberOfElements(in_arr);
  std::vector<double> vec(nv);

  double* data = mxGetPr(in_arr);
  for(int i=0; i<nv; i++)
  {
    vec[i] = *data++;
  }
  return vec;
}

mxArray* callCallbackFunction(const mxArray* fcn_handle)
{
  int status;

  mexAssert(isFunctionHandle(fcn_handle), "No function handle given.");

  // create symbolic object of class CasadiSym
  mxArray* idx = mxCreateDoubleScalar((double)num_symbolics);
  casadi::SX var = casadi::SX::sym("vv", 5, 5);
  symbolics[num_symbolics] = var;
  num_symbolics++;

  mxArray* fh = mxDuplicateArray(fcn_handle);

  mxArray* inputs[2];
  inputs[0] = fh;
  inputs[1] = idx;

  mxArray* outputs[1];
  status = mexCallMATLAB(1, outputs, 2, inputs, "feval");
  mexAssert(status==0, "Error in calling function handle.");

  // Free dynamically allocated memory
  mxDestroyArray(fh);
  mxDestroyArray(idx);

  return outputs[0];
}

void processTensor(int nlhs, mxArray *plhs[],
                   int nrhs, const mxArray *prhs[])
{

  mexAssert(nrhs >=  1, "Not enough input arguments.");

  // First input is always the function identifier as string
  std::string fcn_name = mxToString(prhs[0]);

  if (fcn_name.compare("construct") == 0) {

    // call args are:
    //  prhs[0] "construct" : string
    //  prhs[1] value : double
    mexAssert(nrhs >= 2, "Not enough input arguments.");
    std::vector<double> value = mxToNumericVec(prhs[1]);

    mxAssert(value.size() == 1, "Only scalar values.");

    // create tensor and insert list of tensors, remember index
    ocl::Tensor tensor(value[0]);
    tensor_storage[num_tensors] = tensor;
    int idx = num_tensors;
    num_tensors++;

    // return index of symbolic
    mexAssert(nlhs >= 1, "Output argument for index required");
    plhs[0] = mxCreateDoubleScalar((double)idx);
  }

  else if (fcn_name.compare("uplus") == 0)
  {
    // call args are:
    //  prhs[0] "uplus" : string
    //  prhs[1] idx : int
    mexAssert(nrhs >= 2, "Not enough input arguments.");

    int idx = (int)mxGetScalar(prhs[1]);
    ocl::Tensor tensor = tensor_storage[idx];

    ocl::Tensor new_tensor = uplus(tensor);
    tensor_storage[num_tensors] = new_tensor;
    int new_idx = num_tensors;
    num_tensors++;

    // return index of symbolic
    mexAssert(nlhs >= 1, "Output argument for index required");
    plhs[0] = mxCreateDoubleScalar((double)new_idx);
  }
  else if (fcn_name.compare("full") == 0)
  {
    // call args are:
    //  prhs[0] "uplus" : string
    //  prhs[1] idx : int
    mexAssert(nrhs >= 2, "Not enough input arguments.");

    int idx = (int)mxGetScalar(prhs[1]);
    ocl::Tensor tensor = tensor_storage[idx];

    std::vector<std::vector<double>> data = ocl::full(tensor);

    // return index of symbolic
    mexAssert(nlhs >= 3, "Output argument for index required");
    plhs[0] = mxCreateDoubleScalar((double)data.size());
    plhs[1] = mxCreateDoubleScalar((double)data[0].size());
    plhs[3] = mxCreateDoubleScalar(data);
  }
  else
  {
    mexError("Method not recognized.");
  }

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

  mexAssert(nrhs >=  2, "Not enough input arguments.");

  std::string class_name = mxToString(prhs[1]);
  // tensor class
  if (class_name.compare("T") == 0) {
    processTensor(--nrhs, ++prhs);
  }
  else {
    mexError("Class not recognized.");
  }
}
