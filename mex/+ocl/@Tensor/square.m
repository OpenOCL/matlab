function t = square(self)
  new_idx = ocl_mex_program('T', 'square', self.idx);
  t = createTensor(new_idx);
end
