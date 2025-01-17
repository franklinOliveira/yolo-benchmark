#include <iostream>
#include <string>
#include <mqtt/client.h>
#include <nlohmann/json.hpp>

class MQTTProducer {
private:
    const std::string serverAddress;
    const std::string clientId;
    mqtt::client client;

public:
    MQTTProducer(const std::string& server, const std::string& clientID);
    ~MQTTProducer();

    void start();
    void stop();
    void produce(const std::string& topic, const nlohmann::json& msg);
};