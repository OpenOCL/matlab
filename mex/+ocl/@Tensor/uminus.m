function t = uminus(self)
  new_idx = ocl_mex_program('T', 'uminus', self.idx);
  t = createTensor(new_idx);
end
