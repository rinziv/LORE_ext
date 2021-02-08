import grpc
from concurrent import futures
import time

# import generated classes
import model_pb2
import model_pb2_grpc

# import the model
import model as explainer

port = 55053

class LoreTabularExplainerServicer(model_pb2_grpc.LoreTabularExplainerServicer):
    def explain_tuple(self, request, context):
        response = model_pb2.TupleExplanation()
        response.rule = explainer.explain_tuple(request.field)

        return response


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), )

model_pb2_grpc.add_LoreTabularExplainerServicer_to_server(LoreTabularExplainerServicer(), server)

print('Starting server. Listening on port: '+ str(port))
server.add_insecure_port("[::]:{}".format(port))
server.start()

try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)