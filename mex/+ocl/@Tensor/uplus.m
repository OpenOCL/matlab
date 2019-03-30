function t = uplus(self)
  new_idx = ocl_mex_program('T', 'uplus', self.idx);
  t = createTensor(new_idx);
end
