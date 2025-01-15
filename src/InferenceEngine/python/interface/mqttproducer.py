import paho.mqtt.client as mqtt
import json

class MQTTProducer:

    def __init__(self, server, client_id):
        self.server = server
        self.client_id = client_id
        self.client = mqtt.Client(self.client_id)
    
    def __del__(self):
        self.stop()
    
    def start(self):
        self.client.connect(self.server["address"], self.server["port"])

    def stop(self):
        self.client.disconnect()

    def produce(self, topic, msg):
        msg_str = json.dumps(msg)
        self.client.publish(topic, msg_str)