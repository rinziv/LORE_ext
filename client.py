import grpc
import model_pb2
import model_pb2_grpc

port = 55053
host = 'localhost'

channel = grpc.insecure_channel("{}:{}".format(host,port))

stub = model_pb2_grpc.LoreTabularExplainerStub(channel)

x = [38,28887,7,0,0,
            50,0,0,0,1,0,0,
            0,0,0,1,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,1,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,
            1,0,0,1,0,0,0,0,0,0,0,0,
            0,1,0,1,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,1,0,0]

requestTuple = model_pb2.TupleInstance()
requestTuple.field[:] = x

print(len(requestTuple.field))


responseExplanation = stub.explain_tuple(requestTuple)
print('Explanation:' + responseExplanation.rule)