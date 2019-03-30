function t = inverse(self)
  new_idx = ocl_mex_program('T', 'inverse', self.idx);
  t = createTensor(new_idx);
end
