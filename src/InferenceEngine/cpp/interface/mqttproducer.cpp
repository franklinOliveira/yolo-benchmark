#include "mqttproducer.hpp"

MQTTProducer::MQTTProducer(const std::string& server, const std::string& clientID)
    : serverAddress(server), clientId(clientID), client(serverAddress, clientId, mqtt::create_options(MQTTVERSION_5)) {}

MQTTProducer::~MQTTProducer() { this->stop(); }

void MQTTProducer::start() { this->client.connect(); }

void MQTTProducer::stop() { this->client.disconnect(); }

void MQTTProducer::produce(const std::string& topic, const nlohmann::json& msg) {

    std::string msgStr = msg.dump();
    mqtt::message_ptr msgPointer = mqtt::make_message(topic, msgStr);
    msgPointer->set_payload(msgStr);
    this->client.publish(msgPointer);

}