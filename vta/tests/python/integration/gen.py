
# Modified by contributors from Intel Labs

for i in range(330):
  print( f'class TensorGemmJsonTest{i:04d} extends GenericTest( "TensorGemmJson", (p:Parameters) => new TensorGemmPipelined()(p), (c:TensorGemmPipelined) => new TensorGemmJsonTester(c, "jsons/simLOG{i:04d}.json"))')
