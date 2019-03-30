function self = Tensor(value)
  if nargin == 1
    self.idx = ocl_mex_program('T','construct', value);
  end
end
