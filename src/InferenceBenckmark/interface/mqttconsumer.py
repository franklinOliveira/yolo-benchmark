import paho.mqtt.client as mqtt
from queue import Queue
import json

class MQTTConsumer:

    def __init__(self, server, client_id, topics):
        self.server = server
        self.client_id = client_id
        self.topics = topics
        self.client = mqtt.Client(self.client_id)

    def __del__(self):
        self.stop()

    def on_connect(self, client, userdata, flags, rc):
        for topic in self.topics:
            self.client.subscribe(topic) 

    def on_message(self, client, userdata, msg):
        self.msgs.put([msg.topic, msg.payload.decode()])

    def start(self):
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.msgs = Queue(5)

        self.client.connect(self.server["address"], self.server["port"])
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    def consume(self):
        if self.msgs.qsize() == 0:
            return None, None
        else:
            topic, msg_str = self.msgs.get()
            msg = json.loads(msg_str)
            return topic, msg