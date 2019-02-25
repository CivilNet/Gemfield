import sys
import coremltools  
from coremltools.models.neural_network import flexible_shape_utils  
spec = coremltools.utils.load_spec(sys.argv[1])
print(spec)
