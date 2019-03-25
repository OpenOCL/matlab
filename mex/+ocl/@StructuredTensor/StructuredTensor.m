classdef StructuredTensor < handle
% STRUCTUREDTENSOR
% Shortcut: ST

  properties (Static)
  end

  properties
    instanceID % index number to identify instance
  end

  methods (Static)
    function r = create(structureID, value)
      instanceID = oclMex('ST', 'create', structureID, value);
      r = StructuredTensor(instanceID);
    end

  end
  methods

    function self = StructuredTensor(instanceID)
      self.instanceID = instanceID;
    end

    function r = uplus(self)
      instanceID = oclMex('ST', 'uplus', self.instanceID);
      r = StructuredTensor(instanceID);
    end

    function r = plus(self, other)
      instanceID = oclMex('ST', 'plus', self.instanceID, other.instanceID);
      r = StructuredTensor(instanceID);
    end

    function r = pow(self, exponent)
      instanceID = oclMex('ST', 'pow', self.instanceID, exponent);
      r = StructuredTensor(instanceID);
    end

    function r = reshape(self, cols, rows)
      instanceID = oclMex('ST', 'reshape', self.instanceID, cols, rows);
      r = StructuredTensor(instanceID);
    end

    function r = str(self)
      r = oclTensorMex('ST', 'str', self.instanceID);
    end

  end

end
