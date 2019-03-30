function t = abs(self)
  new_idx = ocl_mex_program('T', 'abs', self.idx);
  t = createTensor(new_idx);
end
