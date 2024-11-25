#include "postprocessing.hpp"

namespace ImagePostprocessing
{
    std::vector<Detection> applyNMS(std::vector<int> predictedClasses, std::vector<float> predictedScores, std::vector<cv::Rect> predictedBoxes, const float confidenceThresh, const float iouThresh)
    {
        std::vector<Detection> detections;
        std::vector<int> nmsResult;

        cv::dnn::NMSBoxes(predictedBoxes, predictedScores, confidenceThresh, iouThresh, nmsResult);
        for (unsigned long i = 0; i < nmsResult.size(); ++i)
        {
            int idx = nmsResult[i];
            float score = predictedScores[idx];
            int classId = predictedClasses[idx];
            cv::Rect box = predictedBoxes[idx];
            int location[4] = {
                box.x,
                box.y,
                (box.x + box.width),
                (box.y + box.height)};
            
            Detection detection(classId, score, location);
            detections.push_back(detection);

            std::cout << detection.getClassId() << " at (" << detection.getBoundingBox().xMin << ", " << detection.getBoundingBox().yMin << ") with " << detection.getScore() << std::endl;
        }

        return detections;
    }
}